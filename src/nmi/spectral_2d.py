# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Fourier pseudo-spectral solver for model (1) on the two-dimensional
# torus. The discretisation follows the one-dimensional solver: diffusion is
# implicit, the nonlocal advective flux is explicit, the convolution is
# evaluated exactly in Fourier space and the memory equation is advanced with
# an exact exponential update.
# Manuscript: Section 4.1 and Remark 2; the solver produces the data of
# Experiments 1.3, 3.1 and 3.2.
# Inputs: model parameters, initial data and a resolution. Outputs: recorded
# densities, the final state and solver diagnostics.

import time  # Wall clock measurement of the cost of a forward solve.

import numpy as np  # Numerical arrays and fast Fourier transforms.

from nmi.grids import dealias_mask_2d, wavenumbers_2d  # Wavenumbers and dealiasing mask.
from nmi.kernels import kernel_hat_2d  # Fourier transforms of the detection kernels.
from nmi.spectral_1d import PositivityError  # Shared error type of the spectral solvers.


# Right hand side of the advective term of model (1) in two space dimensions.
# Arguments:
#   density (numpy.ndarray): samples of the density on the periodic grid.
#   map_spectrum (numpy.ndarray): Fourier coefficients of the cognitive map.
#   alpha (float): advection strength alpha.
#   wavenumbers (tuple): the two wavenumber component arrays.
#   multiplier (numpy.ndarray): kernel multipliers G_hat(R |xi|).
#   mask (numpy.ndarray): boolean two-thirds dealiasing mask.
#   n_points (int): number of grid points N in each coordinate direction.
# Returns:
#   numpy.ndarray: Fourier coefficients of the advective contribution.
def advective_term_2d(density, map_spectrum, alpha, wavenumbers, multiplier, mask, n_points):
    wavenumber_x, wavenumber_y = wavenumbers  # Components of the wavenumber vector.
    perceived_spectrum = multiplier * map_spectrum  # Fourier coefficients of G_R * k.
    # First component of the gradient of the perceived map in physical space.
    gradient_x = np.fft.irfft2(1j * wavenumber_x * perceived_spectrum, s=(n_points, n_points))
    # Second component of the gradient of the perceived map in physical space.
    gradient_y = np.fft.irfft2(1j * wavenumber_y * perceived_spectrum, s=(n_points, n_points))
    flux_x = np.fft.rfft2(density * gradient_x) * mask  # Dealiased first flux component.
    flux_y = np.fft.rfft2(density * gradient_y) * mask  # Dealiased second flux component.
    divergence = 1j * (wavenumber_x * flux_x + wavenumber_y * flux_y)  # Divergence of the flux.
    return -alpha * divergence  # Advective contribution to the density equation.


# Solve model (1) on the two-dimensional torus with the pseudo-spectral method.
# Arguments:
#   params (dict): keys "diffusion", "alpha", "beta", "memory_decay", "radius".
#   kernel (str): "tophat" or "gaussian".
#   u0 (numpy.ndarray): initial density of shape (N, N).
#   k0 (numpy.ndarray or float): initial cognitive map of the same shape.
#   t_final (float): end of the observation window.
#   dt (float): time step of the implicit-explicit scheme.
#   n_points (int): number of grid points N in each coordinate direction.
#   domain_length (float): side length L of the periodic square.
#   record_times (sequence): times at which the density is recorded.
#   scheme (str): "imex_euler" or "sbdf2".
#   positivity_tolerance (float): relative tolerance of the positivity monitor.
# Returns:
#   dict: recorded times and densities, the final density and map, and the
#   diagnostics minimum density, relative mass error and wall time.
def simulate_2d(
    params,  # Model parameters of equation (1).
    kernel,  # Detection kernel family.
    u0,  # Initial density on the periodic grid.
    k0,  # Initial cognitive map on the same grid.
    t_final,  # End of the observation window.
    dt,  # Time step of the implicit-explicit scheme.
    n_points,  # Number of grid points N in each coordinate direction.
    domain_length=1.0,  # Side length L of the periodic square.
    record_times=None,  # Times at which the density is recorded.
    scheme="imex_euler",  # Time stepping scheme of Section 4.1.
    positivity_tolerance=1.0e-8,  # Relative tolerance of the positivity monitor.
):  # End of the argument list.
    started = time.perf_counter()  # Wall clock reading at the start of the solve.
    diffusion = float(params["diffusion"])  # Diffusion rate d of the density equation.
    alpha = float(params["alpha"])  # Advection strength alpha of the density equation.
    beta = float(params["beta"])  # Memory uptake rate beta of the map equation.
    memory_decay = float(params["memory_decay"])  # Memory decay rate mu of the map equation.
    radius = float(params["radius"])  # Perceptual range R of the detection kernel.
    wavenumber_x, wavenumber_y = wavenumbers_2d(n_points, domain_length)  # Wavenumber components.
    norm = np.sqrt(wavenumber_x**2 + wavenumber_y**2)  # Euclidean norm of the wavenumber vector.
    multiplier = kernel_hat_2d(radius * norm, kernel)  # Multipliers of Remark 2.
    mask = dealias_mask_2d(n_points)  # Two-thirds dealiasing mask of the nonlinear product.
    squared = wavenumber_x**2 + wavenumber_y**2  # Squared wavenumber used by the diffusion solve.
    density = np.array(u0, dtype=float)  # Working copy of the initial density.
    cognitive_map = np.zeros((n_points, n_points)) + np.asarray(k0, dtype=float)  # Initial map.
    n_steps = int(round(t_final / dt))  # Number of time steps of the forward solve.
    # Times requested by the caller, defaulting to the end of the window alone.
    requested = np.atleast_1d(np.asarray(record_times if record_times is not None else [t_final]))
    record_steps = np.clip(np.round(requested / dt).astype(int), 0, n_steps)  # Nearest steps.
    recorded = np.empty((record_steps.size, n_points, n_points), dtype=float)  # Record storage.
    cell = (domain_length / n_points) ** 2  # Area of one grid cell, used by the mass quadrature.
    initial_mass = float(np.sum(density) * cell)  # Conserved mass of the initial density.
    decay_factor = np.exp(-memory_decay * dt)  # Exact decay factor of the memory equation.
    uptake_factor = (beta / memory_decay) * (1.0 - decay_factor)  # Exact uptake contribution.
    implicit_euler = 1.0 / (1.0 + dt * diffusion * squared)  # Implicit diffusion factor.
    implicit_sbdf2 = 1.0 / (1.0 + (2.0 / 3.0) * dt * diffusion * squared)  # SBDF2 factor.
    density_spectrum = np.fft.rfft2(density)  # Fourier coefficients of the initial density.
    previous_spectrum = None  # Density spectrum of the previous step, used by SBDF2.
    previous_term = None  # Advective term of the previous step, used by SBDF2.
    minimum_density = float(np.min(density))  # Smallest density observed during the solve.
    for index, step in enumerate(record_steps):  # Record the states requested at step zero.
        if step == 0:  # The record is requested at the initial time.
            recorded[index] = density  # Store the initial density for this record.
    for step in range(n_steps):  # Advance the solution one time step at a time.
        map_spectrum = np.fft.rfft2(cognitive_map)  # Fourier coefficients of the cognitive map.
        # Advective contribution evaluated explicitly at the current state.
        term = advective_term_2d(
            density,  # Current density in physical space.
            map_spectrum,  # Fourier coefficients of the current cognitive map.
            alpha,  # Advection strength alpha.
            (wavenumber_x, wavenumber_y),  # Components of the wavenumber vector.
            multiplier,  # Kernel multipliers of Remark 2.
            mask,  # Two-thirds dealiasing mask.
            n_points,  # Number of grid points N in each direction.
        )  # Fourier coefficients of the advective contribution at the current state.
        use_sbdf2 = scheme == "sbdf2" and previous_spectrum is not None  # Second order available.
        if use_sbdf2:  # Second order backward differentiation with explicit advection.
            explicit = (2.0 / 3.0) * dt * (2.0 * term - previous_term)  # Extrapolated advection.
            history = (4.0 / 3.0) * density_spectrum - (1.0 / 3.0) * previous_spectrum  # History.
            updated_spectrum = implicit_sbdf2 * (history + explicit)  # Implicit diffusion solve.
        else:  # First order implicit-explicit Euler step, also used to start SBDF2.
            updated_spectrum = implicit_euler * (density_spectrum + dt * term)  # Euler update.
        previous_spectrum = density_spectrum  # Remember the spectrum of the completed step.
        previous_term = term  # Remember the advective term of the completed step.
        cognitive_map = decay_factor * cognitive_map + uptake_factor * density  # Exact update.
        density_spectrum = updated_spectrum  # Accept the updated density spectrum.
        density = np.fft.irfft2(density_spectrum, s=(n_points, n_points))  # Physical space.
        smallest = float(np.min(density))  # Smallest density of the current state.
        minimum_density = min(minimum_density, smallest)  # Track the smallest density observed.
        largest = float(np.max(np.abs(density)))  # Largest magnitude of the current state.
        if smallest < -positivity_tolerance * largest:  # The solution has lost positivity.
            # Compose an informative message before rejecting the run.
            message = f"Density became negative at step {step + 1}: min u = {smallest:.3e}"
            raise PositivityError(message)  # Reject the run as specified in Section 4.1.
        for index, record_step in enumerate(record_steps):  # Store the requested records.
            if record_step == step + 1:  # This step coincides with a requested record time.
                recorded[index] = density  # Store the density of the current state.
    final_mass = float(np.sum(density) * cell)  # Mass of the final density.
    mass_error = abs(final_mass - initial_mass) / abs(initial_mass)  # Relative mass error.
    # Assemble the recorded solution together with the solver diagnostics.
    return {
        "times": record_steps * dt,  # Times at which the density was actually recorded.
        "density": recorded,  # Recorded densities, one slab per record time.
        "final_density": density,  # Density at the end of the observation window.
        "final_map": cognitive_map,  # Cognitive map at the end of the observation window.
        "min_density": minimum_density,  # Smallest density observed during the solve.
        "mass_error": mass_error,  # Relative error of the conserved mass.
        "wall_time": time.perf_counter() - started,  # Wall clock cost of the forward solve.
        "steps": n_steps,  # Number of time steps performed.
    }
