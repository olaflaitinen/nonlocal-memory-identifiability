# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Fourier pseudo-spectral solver for model (1) on the one-dimensional
# torus. Diffusion is treated implicitly and the nonlocal advective flux
# explicitly, the convolution is evaluated exactly in Fourier space and the
# memory equation is advanced with an exact exponential update.
# Manuscript: Section 4.1, numerical solution; the solver produces the data of
# Experiments 1.1, 1.2, 2.1 to 2.6 and 4.2.
# Inputs: model parameters, initial data and a resolution. Outputs: recorded
# densities, the final state and solver diagnostics.

import time  # Wall clock measurement of the cost of a forward solve.

import numpy as np  # Numerical arrays and fast Fourier transforms.

from nmi.grids import dealias_mask_1d, wavenumbers_1d  # Wavenumbers and dealiasing mask.
from nmi.kernels import kernel_hat_1d  # Fourier transforms of the detection kernels.


# Error raised when a pseudo-spectral solution loses positivity.
# The class carries no further state and exists so that experiment scripts can
# distinguish a rejected run from a genuine programming error.
class PositivityError(RuntimeError):
    # The base class provides all required behaviour, so no body is needed.
    pass  # No additional attributes or methods are required.


# Right hand side of the advective term of model (1) in Fourier space.
# Arguments:
#   density (numpy.ndarray): samples of the density on the periodic grid.
#   map_spectrum (numpy.ndarray): Fourier coefficients of the cognitive map.
#   alpha (float): advection strength alpha.
#   wavenumbers (numpy.ndarray): wavenumbers of the real transform.
#   multiplier (numpy.ndarray): kernel multipliers G_hat(R xi_m).
#   mask (numpy.ndarray): boolean two-thirds dealiasing mask.
#   n_points (int): number of grid points of the discretisation.
# Returns:
#   numpy.ndarray: Fourier coefficients of the advective contribution.
def advective_term(density, map_spectrum, alpha, wavenumbers, multiplier, mask, n_points):
    perceived_gradient_spectrum = 1j * wavenumbers * multiplier * map_spectrum  # Gradient of G_R * k.
    perceived_gradient = np.fft.irfft(perceived_gradient_spectrum, n=n_points)  # Physical space.
    flux = density * perceived_gradient  # Advective flux u times the perceived gradient.
    flux_spectrum = np.fft.rfft(flux) * mask  # Dealiased Fourier coefficients of the flux.
    return -alpha * 1j * wavenumbers * flux_spectrum  # Divergence of the advective flux.


# Solve model (1) on the one-dimensional torus with the pseudo-spectral method.
# Arguments:
#   params (dict): keys "diffusion", "alpha", "beta", "memory_decay", "radius".
#   kernel (str): "tophat" or "gaussian".
#   u0 (numpy.ndarray): initial density on the periodic grid of N points.
#   k0 (numpy.ndarray or float): initial cognitive map on the same grid.
#   t_final (float): end of the observation window.
#   dt (float): time step of the implicit-explicit scheme.
#   n_points (int): number of grid points N.
#   domain_length (float): length L of the periodic domain.
#   record_times (sequence): times at which the density is recorded.
#   scheme (str): "imex_euler" or "sbdf2".
#   positivity_tolerance (float): relative tolerance of the positivity monitor.
# Returns:
#   dict: recorded times and densities, the final density and map, and the
#   diagnostics minimum density, relative mass error and wall time.
def simulate_1d(
    params,  # Model parameters of equation (1).
    kernel,  # Detection kernel family.
    u0,  # Initial density on the periodic grid.
    k0,  # Initial cognitive map on the same grid.
    t_final,  # End of the observation window.
    dt,  # Time step of the implicit-explicit scheme.
    n_points,  # Number of grid points N.
    domain_length=1.0,  # Length L of the periodic domain.
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
    wavenumbers = wavenumbers_1d(n_points, domain_length)  # Wavenumbers of the real transform.
    multiplier = kernel_hat_1d(radius * wavenumbers, kernel)  # Multipliers of equation (3).
    mask = dealias_mask_1d(n_points)  # Two-thirds dealiasing mask of the nonlinear product.
    density = np.array(u0, dtype=float)  # Working copy of the initial density.
    cognitive_map = np.zeros(n_points) + np.asarray(k0, dtype=float)  # Initial cognitive map.
    n_steps = int(round(t_final / dt))  # Number of time steps of the forward solve.
    # Times requested by the caller, defaulting to the end of the window alone.
    requested = np.atleast_1d(np.asarray(record_times if record_times is not None else [t_final]))
    record_steps = np.clip(np.round(requested / dt).astype(int), 0, n_steps)  # Nearest steps.
    recorded = np.empty((record_steps.size, n_points), dtype=float)  # Storage of the records.
    cell = domain_length / n_points  # Width of one grid cell, used by the mass quadrature.
    initial_mass = float(np.sum(density) * cell)  # Conserved mass of the initial density.
    decay_factor = np.exp(-memory_decay * dt)  # Exact decay factor of the memory equation.
    uptake_factor = (beta / memory_decay) * (1.0 - decay_factor)  # Exact uptake contribution.
    implicit_euler = 1.0 / (1.0 + dt * diffusion * wavenumbers**2)  # Implicit diffusion factor.
    implicit_sbdf2 = 1.0 / (1.0 + (2.0 / 3.0) * dt * diffusion * wavenumbers**2)  # SBDF2 factor.
    density_spectrum = np.fft.rfft(density)  # Fourier coefficients of the initial density.
    previous_spectrum = None  # Density spectrum of the previous step, used by SBDF2.
    previous_term = None  # Advective term of the previous step, used by SBDF2.
    minimum_density = float(np.min(density))  # Smallest density observed during the solve.
    for index, step in enumerate(record_steps):  # Record the states requested at step zero.
        if step == 0:  # The record is requested at the initial time.
            recorded[index] = density  # Store the initial density for this record.
    for step in range(n_steps):  # Advance the solution one time step at a time.
        map_spectrum = np.fft.rfft(cognitive_map)  # Fourier coefficients of the cognitive map.
        # Advective contribution evaluated explicitly at the current state.
        term = advective_term(
            density,  # Current density in physical space.
            map_spectrum,  # Fourier coefficients of the current cognitive map.
            alpha,  # Advection strength alpha.
            wavenumbers,  # Wavenumbers of the real transform.
            multiplier,  # Kernel multipliers of equation (3).
            mask,  # Two-thirds dealiasing mask.
            n_points,  # Number of grid points N.
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
        density = np.fft.irfft(density_spectrum, n=n_points)  # Density in physical space.
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
        "density": recorded,  # Recorded densities, one row per record time.
        "final_density": density,  # Density at the end of the observation window.
        "final_map": cognitive_map,  # Cognitive map at the end of the observation window.
        "min_density": minimum_density,  # Smallest density observed during the solve.
        "mass_error": mass_error,  # Relative error of the conserved mass.
        "wall_time": time.perf_counter() - started,  # Wall clock cost of the forward solve.
        "steps": n_steps,  # Number of time steps performed.
    }
