# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Linearisation of model (1) about the uniform state, the instability
# criterion of Proposition 3, the index m_star of Corollary 1 and the
# reconstruction of the parameters from two Fourier modes given by Theorem 2.
# Manuscript: Section 3.2, equation (6); Section 3.4, Theorem 2; Section 3.5,
# Corollary 1.
# Inputs: model parameters, wavenumbers and observed mode trajectories.
# Outputs: mode matrices, stability verdicts and reconstructed parameters.

import math  # Scalar elementary functions used by the reconstruction.

import numpy as np  # Numerical arrays and linear algebra.
from scipy.linalg import expm  # Matrix exponential of the two by two mode matrix.

from nmi.kernels import kernel_hat_1d  # Fourier transforms of the detection kernels.


# Mode matrix A_m of the linearised system at a given wavenumber.
# Arguments:
#   wavenumber (float): the wavenumber xi_m = 2 pi m / L.
#   diffusion (float): diffusion rate d.
#   alpha (float): advection strength alpha.
#   beta (float): memory uptake rate beta.
#   memory_decay (float): memory decay rate mu.
#   radius (float): perceptual range R.
#   kernel (str): "tophat" or "gaussian".
#   mean_density (float): mean density u_bar of the uniform state.
# Returns:
#   numpy.ndarray: the two by two matrix A_m of equation (6).
def mode_matrix(wavenumber, diffusion, alpha, beta, memory_decay, radius, kernel, mean_density):
    transform = float(kernel_hat_1d(radius * wavenumber, kernel))  # Kernel multiplier G_hat(R xi).
    upper_right = alpha * mean_density * wavenumber**2 * transform  # Advective coupling term.
    first_row = [-diffusion * wavenumber**2, upper_right]  # First row of equation (6).
    second_row = [beta, -memory_decay]  # Second row of equation (6).
    return np.array([first_row, second_row])  # Mode matrix A_m of the linearised system.


# Test the instability criterion of Proposition 3 for the uniform state.
# Arguments:
#   aggregation_ratio (float): the ratio kappa = gamma / (d mu).
#   radius (float): perceptual range R.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
#   mean_density (float): mean density u_bar of the uniform state.
#   m_max (int): largest mode index inspected by the criterion.
# Returns:
#   bool: True when some mode m >= 1 satisfies kappa u_bar G_hat(R xi_m) > 1.
def is_unstable(aggregation_ratio, radius, kernel, domain_length, mean_density, m_max=512):
    modes = np.arange(1, m_max + 1)  # Mode indices inspected by the criterion.
    wavenumbers = 2.0 * np.pi * modes / domain_length  # Wavenumbers xi_m of those modes.
    transforms = kernel_hat_1d(radius * wavenumbers, kernel)  # Kernel multipliers at the modes.
    return bool(np.any(aggregation_ratio * mean_density * transforms > 1.0))  # Proposition 3.


# Largest growth rate of the linearised system over the inspected modes.
# Arguments:
#   diffusion (float): diffusion rate d.
#   alpha (float): advection strength alpha.
#   beta (float): memory uptake rate beta.
#   memory_decay (float): memory decay rate mu.
#   radius (float): perceptual range R.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
#   mean_density (float): mean density u_bar of the uniform state.
#   m_max (int): largest mode index inspected.
# Returns:
#   tuple: the largest real part of an eigenvalue and the mode index attaining it.
def dominant_growth_rate(
    diffusion,  # Diffusion rate d of the density equation.
    alpha,  # Advection strength alpha of the density equation.
    beta,  # Memory uptake rate beta of the map equation.
    memory_decay,  # Memory decay rate mu of the map equation.
    radius,  # Perceptual range R of the detection kernel.
    kernel,  # Detection kernel family.
    domain_length,  # Length L of the periodic domain.
    mean_density,  # Mean density u_bar of the uniform state.
    m_max=64,  # Largest mode index inspected by the search.
):  # End of the argument list.
    best_rate = -np.inf  # Largest real part observed so far.
    best_mode = 0  # Mode index that attains the largest real part.
    for mode in range(1, m_max + 1):  # Inspect every mode of the truncated spectrum.
        wavenumber = 2.0 * np.pi * mode / domain_length  # Wavenumber of the current mode.
        # Mode matrix of equation (6) evaluated at the wavenumber of this mode.
        matrix = mode_matrix(
            wavenumber,  # Wavenumber of the current mode.
            diffusion,  # Diffusion rate d.
            alpha,  # Advection strength alpha.
            beta,  # Memory uptake rate beta.
            memory_decay,  # Memory decay rate mu.
            radius,  # Perceptual range R.
            kernel,  # Detection kernel family.
            mean_density,  # Mean density u_bar.
        )  # Mode matrix of equation (6) at this wavenumber.
        rate = float(np.max(np.real(np.linalg.eigvals(matrix))))  # Largest real part of the spectrum.
        if rate > best_rate:  # The current mode grows faster than any earlier mode.
            best_rate = rate  # Record the new largest growth rate.
            best_mode = mode  # Record the mode index that attains it.
    return best_rate, best_mode  # Growth rate of the fastest mode and its index.


# Index of the first mode at which the top-hat transform is nonpositive.
# Arguments:
#   radius (float): perceptual range R, with 0 < R < L / 2.
#   domain_length (float): length L of the periodic domain.
# Returns:
#   int: the index m_star = ceil(L / (2 R)) of Corollary 1.
def m_star(radius, domain_length):
    if radius <= 0.0 or radius >= 0.5 * domain_length:  # Outside the admissible range.
        raise ValueError("The top-hat perceptual range must satisfy 0 < R < L / 2")  # Reject.
    return int(math.ceil(domain_length / (2.0 * radius)))  # Corollary 1 gives this index.


# Exact trajectory of one linearised Fourier mode and its first two derivatives.
# Arguments:
#   matrix (numpy.ndarray): the two by two mode matrix A_m of equation (6).
#   initial_state (numpy.ndarray): the initial state (v_m(0), w_m(0)).
#   times (numpy.ndarray): times at which the trajectory is evaluated.
# Returns:
#   tuple: arrays of v_m, its first derivative and its second derivative.
def mode_trajectory(matrix, initial_state, times):
    sample_times = np.asarray(times, dtype=float)  # Times at which the mode is evaluated.
    values = np.empty(sample_times.size, dtype=float)  # Values of the density mode.
    first = np.empty(sample_times.size, dtype=float)  # First derivative of the density mode.
    second = np.empty(sample_times.size, dtype=float)  # Second derivative of the density mode.
    squared = matrix @ matrix  # Square of the mode matrix, used for the second derivative.
    for index, time in enumerate(sample_times):  # Evaluate the exact solution at each time.
        state = expm(matrix * time) @ np.asarray(initial_state, dtype=float)  # Exact state.
        values[index] = state[0]  # First component is the density mode v_m.
        first[index] = (matrix @ state)[0]  # Derivative from the linear system itself.
        second[index] = (squared @ state)[0]  # Second derivative from the same system.
    return values, first, second  # Trajectory and its first two derivatives.


# Estimate the characteristic coefficients (p, q) of a linearised mode.
# Arguments:
#   values (numpy.ndarray): samples of the mode v_m.
#   first (numpy.ndarray): samples of the first derivative of v_m.
#   second (numpy.ndarray): samples of the second derivative of v_m.
# Returns:
#   tuple: the coefficients p and q of Step 3 of the proof of Theorem 2.
def characteristic_coefficients(values, first, second):
    design = np.column_stack([first, values])  # Columns multiplying p and q respectively.
    solution, _, _, _ = np.linalg.lstsq(design, -second, rcond=None)  # Least squares fit.
    return float(solution[0]), float(solution[1])  # Coefficients p and q of the mode.


# Decide whether a linearised mode is a pure exponential, as in Step 2 of Theorem 2.
# Arguments:
#   values (numpy.ndarray): samples of the mode v_m.
#   first (numpy.ndarray): samples of the first derivative of v_m.
#   diffusion (float): diffusion rate d recovered in Step 1.
#   wavenumber (float): the wavenumber xi_m of the mode.
#   tolerance (float): relative tolerance of the exponential test.
# Returns:
#   bool: True when the mode is indistinguishable from a pure exponential.
def is_exponential_mode(values, first, diffusion, wavenumber, tolerance=1.0e-10):
    predicted = -diffusion * wavenumber**2 * values  # Derivative of a pure exponential mode.
    residual = np.max(np.abs(first - predicted))  # Largest deviation from that prediction.
    scale = np.max(np.abs(predicted)) + np.finfo(float).tiny  # Scale used for the relative test.
    return bool(residual / scale < tolerance)  # Compare the relative residual with the tolerance.


# Reconstruct (d, mu, gamma, R) from the first two linearised Fourier modes.
# Arguments:
#   first_mode (tuple): samples of v_1 and of its first two derivatives.
#   second_mode (tuple): samples of v_2 and of its first two derivatives.
#   domain_length (float): length L of the periodic domain.
#   mean_density (float): mean density u_bar of the uniform state.
#   kernel (str): "tophat" or "gaussian".
#   exponential_tolerance (float): relative tolerance of the exponential test.
# Returns:
#   dict: the recovered diffusion rate, memory decay rate, combined advection
#   strength and perceptual range. Implements Theorem 2.
def reconstruct_parameters_theorem2(
    first_mode,  # Samples of the mode m = 1 and of its first two derivatives.
    second_mode,  # Samples of the mode m = 2 and of its first two derivatives.
    domain_length,  # Length L of the periodic domain.
    mean_density,  # Mean density u_bar of the uniform state.
    kernel,  # Detection kernel family, assumed known.
    exponential_tolerance=1.0e-10,  # Relative tolerance of the exponential test.
):  # End of the argument list.
    values_1, first_1, second_1 = first_mode  # Unpack the trajectory of the mode m = 1.
    values_2, first_2, second_2 = second_mode  # Unpack the trajectory of the mode m = 2.
    wavenumber_1 = 2.0 * np.pi / domain_length  # Wavenumber xi_1 of the first mode.
    wavenumber_2 = 4.0 * np.pi / domain_length  # Wavenumber xi_2 of the second mode.
    diffusion = -first_1[0] / (wavenumber_1**2 * values_1[0])  # Step 1 of the proof of Theorem 2.
    coupling = {}  # Products gamma G_hat(R xi_m) recovered for the two observed modes.
    memory_decay = None  # Memory decay rate mu, recovered from the first mode.
    # Both observed modes are processed in turn by Steps 2 to 4 of the proof.
    for mode, wavenumber, trajectory in (
        (1, wavenumber_1, (values_1, first_1, second_1)),  # Data of the first mode.
        (2, wavenumber_2, (values_2, first_2, second_2)),  # Data of the second mode.
    ):  # End of the mode list.
        sampled, derivative, curvature = trajectory  # Unpack the trajectory of this mode.
        # Step 2 decides whether the kernel multiplier vanishes at this mode.
        exponential = is_exponential_mode(
            sampled,  # Samples of the density mode.
            derivative,  # Samples of its first derivative.
            diffusion,  # Diffusion rate recovered in Step 1.
            wavenumber,  # Wavenumber of this mode.
            exponential_tolerance,  # Relative tolerance of the exponential test.
        )  # Step 2 decides whether the kernel multiplier vanishes at this mode.
        if exponential:  # The multiplier G_hat(R xi_m) vanishes for this mode.
            coupling[mode] = 0.0  # The product gamma G_hat(R xi_m) is zero.
            continue  # Nothing further can be learned from a pure exponential mode.
        # Step 3 determines the characteristic coefficients of the second order equation.
        p_coefficient, q_coefficient = characteristic_coefficients(sampled, derivative, curvature)
        if mode == 1:  # The first mode determines the memory decay rate.
            memory_decay = p_coefficient - diffusion * wavenumber**2  # Step 4 of the proof.
        if memory_decay is None:  # The first mode was a pure exponential, which cannot happen.
            # Both kernel families have a strictly positive transform at the first mode.
            raise ValueError("The first mode must not be a pure exponential for both kernels")
        # Step 4 expresses the product gamma G_hat(R xi_m) through the fitted coefficient q.
        product = (diffusion * memory_decay - q_coefficient / wavenumber**2) / mean_density
        coupling[mode] = product  # Record gamma G_hat(R xi_m) for this mode.
    if memory_decay is None:  # No memory decay rate could be recovered from the data.
        # Without the memory decay rate the remaining parameters cannot be identified.
        raise ValueError("The memory decay rate could not be recovered from the first mode")
    # Invert the two kernel identities for the perceptual range and for gamma.
    radius, advection = _invert_kernel_ratio(
        coupling[1],  # Product gamma G_hat(R xi_1) of the first mode.
        coupling[2],  # Product gamma G_hat(R xi_2) of the second mode.
        domain_length,  # Length L of the periodic domain.
        kernel,  # Detection kernel family, assumed known.
    )  # Invert the two kernel identities for the perceptual range and gamma.
    # Assemble the four recovered parameters in the order used by the tests.
    return {
        "diffusion": float(diffusion),  # Recovered diffusion rate d.
        "memory_decay": float(memory_decay),  # Recovered memory decay rate mu.
        "advection": float(advection),  # Recovered combined advection strength gamma.
        "radius": float(radius),  # Recovered perceptual range R.
    }


# Invert the two kernel identities of Theorem 2 for the range and for gamma.
# Arguments:
#   product_1 (float): the value gamma G_hat(R xi_1).
#   product_2 (float): the value gamma G_hat(R xi_2).
#   domain_length (float): length L of the periodic domain.
#   kernel (str): "tophat" or "gaussian".
# Returns:
#   tuple: the perceptual range R and the combined advection strength gamma.
def _invert_kernel_ratio(product_1, product_2, domain_length, kernel):
    if product_1 <= 0.0:  # Both kernels have a strictly positive transform at the first mode.
        # A nonpositive first mode coupling contradicts the hypotheses of Theorem 2.
        raise ValueError("The first mode coupling must be positive for both kernel families")
    if kernel == "gaussian":  # Gaussian transform, inverted through its logarithm.
        ratio = product_2 / product_1  # Ratio exp(-R^2 (xi_2^2 - xi_1^2) / 2) of the transforms.
        if ratio <= 0.0:  # A nonpositive ratio is incompatible with a Gaussian kernel.
            # The Gaussian transform is strictly positive, so the ratio must be positive.
            raise ValueError("The observed mode ratio is incompatible with a Gaussian kernel")
        wavenumber_1 = 2.0 * np.pi / domain_length  # Wavenumber xi_1 of the first mode.
        wavenumber_2 = 4.0 * np.pi / domain_length  # Wavenumber xi_2 of the second mode.
        difference = wavenumber_2**2 - wavenumber_1**2  # Difference of the squared wavenumbers.
        radius = math.sqrt(-2.0 * math.log(ratio) / difference)  # Perceptual range of the kernel.
        advection = product_1 / math.exp(-0.5 * radius**2 * wavenumber_1**2)  # Combined strength.
        return radius, advection  # Recovered perceptual range and combined advection strength.
    if kernel == "tophat":  # Top-hat transform, inverted through Lemma 1.
        cosine = product_2 / product_1  # Lemma 1 gives cos(theta) as the ratio of the products.
        if not -1.0 < cosine < 1.0:  # Outside this range no admissible angle exists.
            # Lemma 1 requires the angle theta to lie strictly inside the interval (0, pi).
            raise ValueError("The observed mode ratio is incompatible with a top-hat kernel")
        angle = math.acos(cosine)  # The angle theta = 2 pi R / L, which lies in (0, pi).
        radius = angle * domain_length / (2.0 * np.pi)  # Perceptual range implied by that angle.
        advection = product_1 * angle / math.sin(angle)  # Combined advection strength gamma.
        return radius, advection  # Recovered perceptual range and combined advection strength.
    raise ValueError(f"Unknown kernel name {kernel!r}")  # Reject unsupported kernel names.
