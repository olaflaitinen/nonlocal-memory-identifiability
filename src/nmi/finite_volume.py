# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Positivity-preserving finite-volume solver for model (1), following
# the first order upwind construction of Carrillo, Chertock and Huang (2015).
# The scheme is used to validate the pseudo-spectral solver, to verify the
# stationary identity on a bounded interval with no-flux boundaries, and to
# confirm the fitted steady states of the wolf application on a masked grid.
# Manuscript: Section 4.1; Proposition 4 for the bounded domain variant.
# Inputs: model parameters, initial data and a resolution. Outputs: recorded
# densities, the final state and solver diagnostics.

import time  # Wall clock measurement of the cost of a forward solve.

import numpy as np  # Numerical arrays and fast Fourier transforms.

from nmi.grids import wavenumbers_1d  # Wavenumbers used by the periodic convolution.
from nmi.kernels import kernel_hat_1d, kernel_values_1d  # Kernel transforms and profiles.


# Convolution of a field with the detection kernel on a bounded interval.
# The field is extended by zero outside the interval, as required by the
# bounded domain formulation of Proposition 4.
# Arguments:
#   field (numpy.ndarray): cell averages on the bounded interval.
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   cell (float): uniform cell width of the grid.
# Returns:
#   numpy.ndarray: cell averages of G_R * field on the same grid.
def convolve_bounded_1d(field, radius, kernel, cell):
    n_cells = field.shape[0]  # Number of cells of the bounded interval.
    offsets = (np.arange(-(n_cells - 1), n_cells)) * cell  # Displacements between cell centres.
    weights = kernel_values_1d(offsets, radius, kernel) * cell  # Quadrature weights of the kernel.
    full = np.convolve(field, weights, mode="full")  # Discrete convolution with zero extension.
    return full[n_cells - 1 : 2 * n_cells - 1]  # Entries that correspond to the cells of the grid.


# Convolution of a periodic field with the detection kernel.
# Arguments:
#   field (numpy.ndarray): cell averages on the periodic grid.
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
# Returns:
#   numpy.ndarray: cell averages of G_R * field on the same grid.
def convolve_periodic_grid(field, radius, kernel, domain_length):
    n_cells = field.shape[0]  # Number of cells of the periodic grid.
    wavenumbers = wavenumbers_1d(n_cells, domain_length)  # Wavenumbers of the real transform.
    multiplier = kernel_hat_1d(radius * wavenumbers, kernel)  # Multipliers of equation (3).
    return np.fft.irfft(np.fft.rfft(field) * multiplier, n=n_cells)  # Exact periodic convolution.


# Upwind numerical flux of the finite-volume scheme at the cell interfaces.
# Arguments:
#   density (numpy.ndarray): cell averages of the density.
#   velocity (numpy.ndarray): advective velocity at the interfaces.
#   diffusion (float): diffusion rate d of the density equation.
#   cell (float): uniform cell width of the grid.
#   periodic (bool): True for the periodic variant, False for no-flux boundaries.
# Returns:
#   numpy.ndarray: numerical fluxes at the interfaces, one more than the cells
#   in the no-flux case and exactly as many as the cells in the periodic case.
def upwind_flux_1d(density, velocity, diffusion, cell, periodic):
    if periodic:  # Interfaces wrap around the torus, so neighbours are cyclic.
        left = density  # Cell average on the left of each interface.
        right = np.roll(density, -1)  # Cell average on the right of each interface.
    else:  # Interior interfaces only, with zero flux imposed at the boundary.
        left = density[:-1]  # Cell average on the left of each interior interface.
        right = density[1:]  # Cell average on the right of each interior interface.
    diffusive = -diffusion * (right - left) / cell  # Central difference of the diffusive flux.
    advective = np.maximum(velocity, 0.0) * left + np.minimum(velocity, 0.0) * right  # Upwind.
    interior = diffusive + advective  # Total numerical flux at the interfaces.
    if periodic:  # The periodic variant needs no boundary treatment.
        return interior  # Fluxes at the N interfaces of the torus.
    return np.concatenate(([0.0], interior, [0.0]))  # Zero flux through the two boundaries.


# Positivity-preserving finite-volume solve of model (1) on an interval.
# Arguments:
#   params (dict): keys "diffusion", "alpha", "beta", "memory_decay", "radius".
#   kernel (str): "tophat" or "gaussian".
#   u0 (numpy.ndarray): initial cell averages of the density.
#   k0 (numpy.ndarray or float): initial cell averages of the cognitive map.
#   t_final (float): end of the observation window.
#   dt (float): time step of the explicit scheme.
#   domain_length (float): length L of the interval or of the periodic domain.
#   boundary (str): "periodic" or "noflux".
#   record_times (sequence): times at which the density is recorded.
#   use_numba (bool): whether the compiled flux update is used when available.
# Returns:
#   dict: recorded times and densities, the final state and diagnostics.
def simulate_fv_1d(
    params,  # Model parameters of equation (1).
    kernel,  # Detection kernel family.
    u0,  # Initial cell averages of the density.
    k0,  # Initial cell averages of the cognitive map.
    t_final,  # End of the observation window.
    dt,  # Time step of the explicit scheme.
    domain_length=1.0,  # Length L of the interval or of the periodic domain.
    boundary="periodic",  # Boundary condition, "periodic" or "noflux".
    record_times=None,  # Times at which the density is recorded.
    use_numba=False,  # Whether the compiled flux update is used when available.
):  # End of the argument list.
    started = time.perf_counter()  # Wall clock reading at the start of the solve.
    diffusion = float(params["diffusion"])  # Diffusion rate d of the density equation.
    alpha = float(params["alpha"])  # Advection strength alpha of the density equation.
    beta = float(params["beta"])  # Memory uptake rate beta of the map equation.
    memory_decay = float(params["memory_decay"])  # Memory decay rate mu of the map equation.
    radius = float(params["radius"])  # Perceptual range R of the detection kernel.
    periodic = boundary == "periodic"  # Whether the periodic variant of the scheme is used.
    if boundary not in ("periodic", "noflux"):  # Only two boundary conditions are implemented.
        raise ValueError(f"Unknown boundary condition {boundary!r}")  # Reject unknown options.
    density = np.array(u0, dtype=float)  # Working copy of the initial cell averages.
    n_cells = density.shape[0]  # Number of cells of the finite-volume grid.
    cognitive_map = np.zeros(n_cells) + np.asarray(k0, dtype=float)  # Initial cognitive map.
    cell = domain_length / n_cells  # Uniform cell width of the finite-volume grid.
    n_steps = int(round(t_final / dt))  # Number of time steps of the forward solve.
    # Times requested by the caller, defaulting to the end of the window alone.
    requested = np.atleast_1d(np.asarray(record_times if record_times is not None else [t_final]))
    record_steps = np.clip(np.round(requested / dt).astype(int), 0, n_steps)  # Nearest steps.
    recorded = np.empty((record_steps.size, n_cells), dtype=float)  # Storage of the records.
    initial_mass = float(np.sum(density) * cell)  # Conserved mass of the initial density.
    decay_factor = np.exp(-memory_decay * dt)  # Exact decay factor of the memory equation.
    uptake_factor = (beta / memory_decay) * (1.0 - decay_factor)  # Exact uptake contribution.
    flux_update = _select_flux_update(use_numba)  # Plain or compiled flux update routine.
    minimum_density = float(np.min(density))  # Smallest density observed during the solve.
    for index, step in enumerate(record_steps):  # Record the states requested at step zero.
        if step == 0:  # The record is requested at the initial time.
            recorded[index] = density  # Store the initial density for this record.
    for step in range(n_steps):  # Advance the solution one time step at a time.
        if periodic:  # Periodic convolution is evaluated exactly in Fourier space.
            # Perceived map obtained from the exact periodic convolution.
            perceived = convolve_periodic_grid(cognitive_map, radius, kernel, domain_length)
        else:  # The bounded variant extends the field by zero outside the interval.
            # Perceived map obtained from the zero extended discrete convolution.
            perceived = convolve_bounded_1d(cognitive_map, radius, kernel, cell)
        if periodic:  # Interface gradients wrap around the torus.
            gradient = (np.roll(perceived, -1) - perceived) / cell  # Gradient at the interfaces.
        else:  # Interior interfaces only, since no flux crosses the boundary.
            gradient = (perceived[1:] - perceived[:-1]) / cell  # Gradient at interior interfaces.
        velocity = alpha * gradient  # Advective velocity at the cell interfaces.
        flux = upwind_flux_1d(density, velocity, diffusion, cell, periodic)  # Numerical flux.
        density = flux_update(density, flux, dt, cell, periodic)  # Conservative cell update.
        cognitive_map = decay_factor * cognitive_map + uptake_factor * density  # Exact update.
        minimum_density = min(minimum_density, float(np.min(density)))  # Track the minimum.
        for index, record_step in enumerate(record_steps):  # Store the requested records.
            if record_step == step + 1:  # This step coincides with a requested record time.
                recorded[index] = density  # Store the density of the current state.
    final_mass = float(np.sum(density) * cell)  # Mass of the final density.
    mass_error = abs(final_mass - initial_mass) / abs(initial_mass)  # Relative mass error.
    # Assemble the recorded solution together with the solver diagnostics.
    return {
        "times": record_steps * dt,  # Times at which the density was actually recorded.
        "density": recorded,  # Recorded densities, one row per record time.
        "final_density": density,  # Density at the end of the observation window.
        "final_map": cognitive_map,  # Cognitive map at the end of the observation window.
        "min_density": minimum_density,  # Smallest density observed during the solve.
        "mass_error": mass_error,  # Relative error of the conserved mass.
        "wall_time": time.perf_counter() - started,  # Wall clock cost of the forward solve.
        "steps": n_steps,  # Number of time steps performed.
    }


# Conservative update of the cell averages from the numerical fluxes.
# Arguments:
#   density (numpy.ndarray): cell averages before the update.
#   flux (numpy.ndarray): numerical fluxes at the cell interfaces.
#   dt (float): time step of the explicit scheme.
#   cell (float): uniform cell width of the grid.
#   periodic (bool): True for the periodic variant, False for no-flux boundaries.
# Returns:
#   numpy.ndarray: cell averages after the update.
def _flux_update_numpy(density, flux, dt, cell, periodic):
    if periodic:  # Interfaces wrap around, so the left neighbour is cyclic.
        divergence = flux - np.roll(flux, 1)  # Difference of the fluxes bounding each cell.
    else:  # The flux array already contains the two boundary interfaces.
        divergence = flux[1:] - flux[:-1]  # Difference of the fluxes bounding each cell.
    return density - (dt / cell) * divergence  # Conservative finite-volume update.


# Select the flux update routine, optionally compiled with Numba.
# Arguments:
#   use_numba (bool): whether the compiled routine is requested.
# Returns:
#   callable: the routine that performs the conservative cell update.
def _select_flux_update(use_numba):
    if not use_numba:  # The plain NumPy implementation was requested.
        return _flux_update_numpy  # Return the vectorised reference implementation.
    try:  # Numba is an optional dependency of the package.
        from numba import njit  # Just in time compiler for numerical loops.
    except ImportError:  # Numba is not installed in the active environment.
        return _flux_update_numpy  # Fall back to the vectorised implementation.
    global _COMPILED_FLUX_UPDATE  # Cache of the compiled routine across calls.
    if _COMPILED_FLUX_UPDATE is None:  # The routine has not been compiled yet.
        _COMPILED_FLUX_UPDATE = njit(cache=False)(_flux_update_numpy)  # Compile it once.
    return _COMPILED_FLUX_UPDATE  # Return the compiled routine.


# Cache that holds the compiled flux update between calls.
_COMPILED_FLUX_UPDATE = None  # Populated on the first request for Numba acceleration.


# Positivity-preserving finite-volume solve of model (1) on a masked grid.
# The grid is a Cartesian rectangle in which cells outside the study area are
# masked, the density is set to zero there and no flux crosses the mask.
# Arguments:
#   params (dict): keys "diffusion", "alpha", "beta", "memory_decay", "radius".
#   kernel (str): "tophat" or "gaussian".
#   u0 (numpy.ndarray): initial cell averages of shape (n_rows, n_cols).
#   k0 (numpy.ndarray or float): initial cognitive map on the same grid.
#   mask (numpy.ndarray): boolean array that is True inside the study area.
#   t_final (float): end of the observation window.
#   dt (float): time step of the explicit scheme.
#   cell (float): uniform cell width of the Cartesian grid.
#   record_times (sequence): times at which the density is recorded.
# Returns:
#   dict: recorded times and densities, the final state and diagnostics.
def simulate_fv_masked_2d(
    params,  # Model parameters of equation (1).
    kernel,  # Detection kernel family.
    u0,  # Initial cell averages on the Cartesian grid.
    k0,  # Initial cognitive map on the same grid.
    mask,  # Boolean array that is True inside the study area.
    t_final,  # End of the observation window.
    dt,  # Time step of the explicit scheme.
    cell,  # Uniform cell width of the Cartesian grid.
    record_times=None,  # Times at which the density is recorded.
):  # End of the argument list.
    from nmi.wolf.masked_solver import convolve_masked  # Padded convolution on the masked grid.

    started = time.perf_counter()  # Wall clock reading at the start of the solve.
    diffusion = float(params["diffusion"])  # Diffusion rate d of the density equation.
    alpha = float(params["alpha"])  # Advection strength alpha of the density equation.
    beta = float(params["beta"])  # Memory uptake rate beta of the map equation.
    memory_decay = float(params["memory_decay"])  # Memory decay rate mu of the map equation.
    radius = float(params["radius"])  # Perceptual range R of the detection kernel.
    inside = np.asarray(mask, dtype=bool)  # Boolean mask of the study area.
    density = np.array(u0, dtype=float) * inside  # Initial density, zero outside the study area.
    cognitive_map = (np.zeros_like(density) + np.asarray(k0, dtype=float)) * inside  # Initial map.
    n_steps = int(round(t_final / dt))  # Number of time steps of the forward solve.
    # Times requested by the caller, defaulting to the end of the window alone.
    requested = np.atleast_1d(np.asarray(record_times if record_times is not None else [t_final]))
    record_steps = np.clip(np.round(requested / dt).astype(int), 0, n_steps)  # Nearest steps.
    recorded = np.empty((record_steps.size,) + density.shape, dtype=float)  # Record storage.
    area = cell * cell  # Area of one grid cell, used by the mass quadrature.
    initial_mass = float(np.sum(density) * area)  # Conserved mass of the initial density.
    decay_factor = np.exp(-memory_decay * dt)  # Exact decay factor of the memory equation.
    uptake_factor = (beta / memory_decay) * (1.0 - decay_factor)  # Exact uptake contribution.
    minimum_density = float(np.min(density[inside])) if inside.any() else 0.0  # Interior minimum.
    open_rows = inside[:-1, :] & inside[1:, :]  # Interfaces between two interior cells, first axis.
    open_cols = inside[:, :-1] & inside[:, 1:]  # Interfaces between two interior cells, second axis.
    for index, step in enumerate(record_steps):  # Record the states requested at step zero.
        if step == 0:  # The record is requested at the initial time.
            recorded[index] = density  # Store the initial density for this record.
    for step in range(n_steps):  # Advance the solution one time step at a time.
        perceived = convolve_masked(cognitive_map, radius, kernel, cell)  # Perceived map.
        gradient_rows = (perceived[1:, :] - perceived[:-1, :]) / cell  # Gradient across rows.
        gradient_cols = (perceived[:, 1:] - perceived[:, :-1]) / cell  # Gradient across columns.
        # Numerical flux across the interfaces that separate neighbouring rows.
        flux_rows = _masked_interface_flux(density[:-1, :], density[1:, :], alpha * gradient_rows, diffusion, cell)
        # Numerical flux across the interfaces that separate neighbouring columns.
        flux_cols = _masked_interface_flux(density[:, :-1], density[:, 1:], alpha * gradient_cols, diffusion, cell)
        flux_rows = flux_rows * open_rows  # No flux crosses the boundary of the study area.
        flux_cols = flux_cols * open_cols  # No flux crosses the boundary of the study area.
        divergence = np.zeros_like(density)  # Accumulator for the divergence of the flux.
        divergence[:-1, :] = divergence[:-1, :] + flux_rows  # Outflow through the upper interface.
        divergence[1:, :] = divergence[1:, :] - flux_rows  # Inflow through the lower interface.
        divergence[:, :-1] = divergence[:, :-1] + flux_cols  # Outflow through the right interface.
        divergence[:, 1:] = divergence[:, 1:] - flux_cols  # Inflow through the left interface.
        density = (density - (dt / cell) * divergence) * inside  # Conservative masked update.
        cognitive_map = (decay_factor * cognitive_map + uptake_factor * density) * inside  # Update.
        if inside.any():  # Track the minimum only over the cells inside the study area.
            minimum_density = min(minimum_density, float(np.min(density[inside])))  # Interior min.
        for index, record_step in enumerate(record_steps):  # Store the requested records.
            if record_step == step + 1:  # This step coincides with a requested record time.
                recorded[index] = density  # Store the density of the current state.
    final_mass = float(np.sum(density) * area)  # Mass of the final density.
    mass_error = abs(final_mass - initial_mass) / abs(initial_mass)  # Relative mass error.
    # Assemble the recorded solution together with the solver diagnostics.
    return {
        "times": record_steps * dt,  # Times at which the density was actually recorded.
        "density": recorded,  # Recorded densities, one slab per record time.
        "final_density": density,  # Density at the end of the observation window.
        "final_map": cognitive_map,  # Cognitive map at the end of the observation window.
        "min_density": minimum_density,  # Smallest interior density observed during the solve.
        "mass_error": mass_error,  # Relative error of the conserved mass.
        "wall_time": time.perf_counter() - started,  # Wall clock cost of the forward solve.
        "steps": n_steps,  # Number of time steps performed.
    }


# Upwind numerical flux across one family of interfaces of the masked grid.
# Arguments:
#   left (numpy.ndarray): cell averages on the left of each interface.
#   right (numpy.ndarray): cell averages on the right of each interface.
#   velocity (numpy.ndarray): advective velocity at the interfaces.
#   diffusion (float): diffusion rate d of the density equation.
#   cell (float): uniform cell width of the Cartesian grid.
# Returns:
#   numpy.ndarray: numerical fluxes at the interfaces.
def _masked_interface_flux(left, right, velocity, diffusion, cell):
    diffusive = -diffusion * (right - left) / cell  # Central difference of the diffusive flux.
    advective = np.maximum(velocity, 0.0) * left + np.minimum(velocity, 0.0) * right  # Upwind.
    return diffusive + advective  # Total numerical flux at the interfaces.
