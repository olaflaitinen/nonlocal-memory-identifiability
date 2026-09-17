# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Steady states of model (1) on the periodic domain. The module
# provides the onset value of the aggregation ratio given by Proposition 3, the
# fixed-point characterisation of the positive steady state implied by
# equation (7), a continuation in the aggregation ratio and the residual of the
# stationary identity.
# Manuscript: Section 3.3, equation (7) and Theorem 1; Section 4.1.
# Inputs: model parameters and grid sizes. Outputs: steady-state densities,
# continuation branches and identity residuals as NumPy arrays.

import numpy as np  # Numerical arrays and fast Fourier transforms.

from nmi.grids import periodic_grid_1d, wavenumbers_1d  # Periodic grid and wavenumbers.
from nmi.kernels import kernel_hat_1d  # Fourier transforms of the detection kernels.


# Convolution of a periodic field with the scaled detection kernel.
# Arguments:
#   field (numpy.ndarray): samples of a real periodic function on the grid.
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
# Returns:
#   numpy.ndarray: samples of G_R * field, evaluated exactly in Fourier space.
def convolve_periodic_1d(field, radius, kernel, domain_length):
    n_points = field.shape[0]  # Number of grid points of the periodic discretisation.
    wavenumbers = wavenumbers_1d(n_points, domain_length)  # Wavenumbers of the real transform.
    multiplier = kernel_hat_1d(radius * wavenumbers, kernel)  # Multipliers of equation (3).
    spectrum = np.fft.rfft(field)  # Discrete Fourier coefficients of the field.
    return np.fft.irfft(spectrum * multiplier, n=n_points)  # Convolution evaluated exactly.


# Onset value of the aggregation ratio given by Proposition 3.
# Arguments:
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
#   mean_density (float): mean density u_bar of the uniform state.
#   m_max (int): largest mode index inspected by the maximisation.
# Returns:
#   float: the critical ratio kappa_c(R) of equation (8).
def critical_kappa(radius, kernel, domain_length, mean_density, m_max=512):
    modes = np.arange(1, m_max + 1)  # Mode indices inspected by the maximisation.
    wavenumbers = 2.0 * np.pi * modes / domain_length  # Wavenumbers xi_m of those modes.
    transforms = kernel_hat_1d(radius * wavenumbers, kernel)  # Kernel multipliers at the modes.
    largest = float(np.max(transforms))  # Largest multiplier over the inspected modes.
    if largest <= 0.0:  # No mode is destabilised by the detection kernel.
        raise ValueError("The detection kernel has no positive Fourier multiplier")  # Reject.
    return 1.0 / (mean_density * largest)  # Critical ratio of Proposition 3.


# Mode index at which the critical aggregation ratio is attained.
# Arguments:
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
#   m_max (int): largest mode index inspected by the maximisation.
# Returns:
#   int: the index of the first mode to become unstable at onset.
def critical_mode(radius, kernel, domain_length, m_max=512):
    modes = np.arange(1, m_max + 1)  # Mode indices inspected by the maximisation.
    wavenumbers = 2.0 * np.pi * modes / domain_length  # Wavenumbers xi_m of those modes.
    transforms = kernel_hat_1d(radius * wavenumbers, kernel)  # Kernel multipliers at the modes.
    return int(modes[int(np.argmax(transforms))])  # Mode index attaining the largest multiplier.


# Residual of the stationary identity of equation (7).
# Arguments:
#   density (numpy.ndarray): samples of a positive candidate steady state.
#   aggregation_ratio (float): the ratio kappa = gamma / (d mu).
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
# Returns:
#   float: the largest deviation of log u - kappa G_R * u from its own mean.
def identity_residual(density, aggregation_ratio, radius, kernel, domain_length):
    perceived = convolve_periodic_1d(density, radius, kernel, domain_length)  # Perceived map.
    potential = np.log(density) - aggregation_ratio * perceived  # Left side of equation (7).
    return float(np.max(np.abs(potential - np.mean(potential))))  # Deviation from a constant.


# Positive steady state of model (1) obtained by damped fixed-point iteration.
# Arguments:
#   aggregation_ratio (float): the ratio kappa = gamma / (d mu).
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   initial_density (numpy.ndarray): starting iterate, strictly positive.
#   domain_length (float): length L of the periodic domain.
#   damping (float): relaxation weight in (0, 1] applied to the fixed-point map.
#   tol (float): convergence tolerance on the largest change between iterates.
#   max_iter (int): maximum number of iterations before the solver gives up.
# Returns:
#   dict: the converged density, the number of iterations, the final increment,
#   the identity residual and the dominant Fourier mode of the state.
def fixed_point_steady_state(
    aggregation_ratio,  # Aggregation ratio kappa of the stationary identity.
    radius,  # Perceptual range R of the detection kernel.
    kernel,  # Detection kernel family.
    initial_density,  # Strictly positive starting iterate.
    domain_length=1.0,  # Length L of the periodic domain.
    damping=0.5,  # Relaxation weight applied to the fixed-point map.
    tol=1.0e-12,  # Convergence tolerance on the largest change between iterates.
    max_iter=20000,  # Maximum number of iterations before the solver gives up.
):  # End of the argument list.
    density = np.array(initial_density, dtype=float)  # Working copy of the starting iterate.
    if np.any(density <= 0.0):  # The fixed-point map requires a strictly positive iterate.
        # A nonpositive iterate would make the logarithm of equation (7) undefined.
        raise ValueError("The starting iterate of the fixed-point solver must be positive")
    n_points = density.shape[0]  # Number of grid points of the periodic discretisation.
    cell = domain_length / n_points  # Width of one grid cell, used by the quadrature.
    mass = float(np.sum(density) * cell)  # Conserved mass of the starting iterate.
    increment = np.inf  # Largest change between successive iterates.
    iteration = 0  # Number of iterations performed so far.
    while iteration < max_iter and increment > tol:  # Iterate until convergence or exhaustion.
        perceived = convolve_periodic_1d(density, radius, kernel, domain_length)  # Perceived map.
        exponent = aggregation_ratio * perceived  # Exponent of the fixed-point map.
        exponent = exponent - float(np.max(exponent))  # Shift for numerical stability.
        candidate = np.exp(exponent)  # Unnormalised image of the fixed-point map.
        candidate = mass * candidate / (float(np.sum(candidate)) * cell)  # Restore the mass.
        updated = (1.0 - damping) * density + damping * candidate  # Damped update of the iterate.
        increment = float(np.max(np.abs(updated - density)))  # Largest change of this iteration.
        density = updated  # Accept the updated iterate.
        iteration = iteration + 1  # Count the iteration that has just been completed.
    # Residual of the stationary identity, reported as a diagnostic of the solve.
    residual = identity_residual(density, aggregation_ratio, radius, kernel, domain_length)
    spectrum = np.abs(np.fft.rfft(density))  # Magnitudes of the Fourier coefficients.
    dominant = int(np.argmax(spectrum[1:]) + 1) if spectrum.size > 1 else 0  # Dominant mode index.
    # Assemble the converged state together with the diagnostics of the solve.
    return {
        "density": density,  # Converged steady-state density on the periodic grid.
        "iterations": iteration,  # Number of fixed-point iterations performed.
        "increment": increment,  # Largest change between the last two iterates.
        "residual": residual,  # Residual of the stationary identity of equation (7).
        "dominant_mode": dominant,  # Dominant Fourier mode labelling the branch.
        "converged": bool(increment <= tol),  # Whether the tolerance was reached.
    }


# Continuation of the positive steady state in the aggregation ratio.
# Arguments:
#   ratios (sequence): increasing aggregation ratios at which a state is sought.
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   n_points (int): number of grid points of the periodic discretisation.
#   domain_length (float): length L of the periodic domain.
#   mean_density (float): mean density u_bar of the conserved mass.
#   seed_amplitude (float): amplitude of the perturbation that starts the branch.
#   damping (float): relaxation weight applied to the fixed-point map.
#   tol (float): convergence tolerance on the largest change between iterates.
#   max_iter (int): maximum number of iterations per continuation step.
# Returns:
#   list: one result dictionary per requested aggregation ratio, in order.
def continuation_in_kappa(
    ratios,  # Increasing aggregation ratios at which a state is sought.
    radius,  # Perceptual range R of the detection kernel.
    kernel,  # Detection kernel family.
    n_points=256,  # Number of grid points of the periodic discretisation.
    domain_length=1.0,  # Length L of the periodic domain.
    mean_density=1.0,  # Mean density u_bar of the conserved mass.
    seed_amplitude=0.01,  # Amplitude of the perturbation that starts the branch.
    damping=0.5,  # Relaxation weight applied to the fixed-point map.
    tol=1.0e-12,  # Convergence tolerance on the largest change between iterates.
    max_iter=20000,  # Maximum number of iterations per continuation step.
):  # End of the argument list.
    grid = periodic_grid_1d(n_points, domain_length)  # Grid points of the periodic domain.
    mode = critical_mode(radius, kernel, domain_length)  # Mode that becomes unstable first.
    phase = np.cos(2.0 * np.pi * mode * grid / domain_length)  # Shape of the unstable mode.
    state = mean_density * (1.0 + seed_amplitude * phase)  # Perturbed uniform starting iterate.
    outcomes = []  # Accumulator for the result of each continuation step.
    for ratio in ratios:  # Solve the fixed-point problem at each requested ratio in turn.
        # Solve the fixed-point problem starting from the previous branch point.
        result = fixed_point_steady_state(
            ratio,  # Aggregation ratio of the current continuation step.
            radius,  # Perceptual range R of the detection kernel.
            kernel,  # Detection kernel family.
            state,  # Starting iterate taken from the previous continuation step.
            domain_length,  # Length L of the periodic domain.
            damping,  # Relaxation weight applied to the fixed-point map.
            tol,  # Convergence tolerance on the largest change between iterates.
            max_iter,  # Maximum number of iterations for this continuation step.
        )  # Converged state and diagnostics of the current continuation step.
        result["aggregation_ratio"] = float(ratio)  # Record the ratio that produced the state.
        outcomes.append(result)  # Store the outcome of this continuation step.
        state = result["density"]  # Continue the branch from the state just computed.
    return outcomes  # One result dictionary per requested aggregation ratio.
