# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Steady state of model (1) on the masked Cartesian grid of a study
# area. The convolution with the detection kernel is evaluated by fast Fourier
# transform on a padded rectangle with the density set to zero outside the
# study area, and the steady state is obtained from the fixed-point form of
# Proposition 4.
# Manuscript: Section 4.5; Proposition 4 for the bounded domain identity.
# Inputs: a mask, model parameters and a cell width. Outputs: the stationary
# density on the masked grid together with the diagnostics of the solve.

import numpy as np  # Numerical arrays and fast Fourier transforms.

from nmi.kernels import kernel_values_2d  # Profiles of the two-dimensional detection kernels.


# Samples of the detection kernel on the padded rectangle, centred at the origin.
# Arguments:
#   shape (tuple): the shape of the padded rectangle.
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   cell (float): uniform cell width of the Cartesian grid.
# Returns:
#   numpy.ndarray: kernel samples arranged for a circular convolution.
def padded_kernel(shape, radius, kernel, cell):
    n_rows, n_cols = shape  # Dimensions of the padded rectangle.
    rows = ((np.arange(n_rows) + n_rows // 2) % n_rows - n_rows // 2) * cell  # Wrapped offsets.
    cols = ((np.arange(n_cols) + n_cols // 2) % n_cols - n_cols // 2) * cell  # Wrapped offsets.
    distance = np.hypot(rows[:, None], cols[None, :])  # Euclidean distance between cell centres.
    return kernel_values_2d(distance, radius, kernel)  # Kernel profile on the padded rectangle.


# Convolution of a field on a Cartesian grid with the detection kernel.
# The field is extended by zero outside the grid, which realises the bounded
# domain convolution of Proposition 4.
# Arguments:
#   field (numpy.ndarray): cell averages on the Cartesian grid.
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   cell (float): uniform cell width of the Cartesian grid.
# Returns:
#   numpy.ndarray: cell averages of G_R * field on the same grid.
def convolve_masked(field, radius, kernel, cell):
    values = np.asarray(field, dtype=float)  # Cell averages of the field on the grid.
    n_rows, n_cols = values.shape  # Dimensions of the Cartesian grid.
    shape = (2 * n_rows, 2 * n_cols)  # Padded rectangle that avoids wrap-around contributions.
    padded = np.zeros(shape)  # Zero extension of the field outside the study area.
    padded[:n_rows, :n_cols] = values  # Place the field in the upper left block of the rectangle.
    weights = padded_kernel(shape, radius, kernel, cell) * cell * cell  # Quadrature weights.
    spectrum = np.fft.rfft2(padded) * np.fft.rfft2(weights)  # Product of the two transforms.
    convolved = np.fft.irfft2(spectrum, s=shape)  # Circular convolution on the padded rectangle.
    return convolved[:n_rows, :n_cols]  # Restriction to the original Cartesian grid.


# Stationary density of model (1) on a masked study area.
# The aggregation ratio is used in the dimensionless form kappa u_bar of
# Proposition 3, where u_bar is the mean density of the uniform state on the
# study area. The exponent of the fixed-point map is therefore the ratio times
# the area of the study region times the perceived probability density, which
# makes the reported value comparable across individuals of different range
# size and with the synthetic experiments of Section 5.
# Arguments:
#   aggregation_ratio (float): the dimensionless ratio kappa u_bar.
#   radius (float): perceptual range R of the detection kernel.
#   kernel (str): "tophat" or "gaussian".
#   mask (numpy.ndarray): boolean array that is True inside the study area.
#   cell (float): uniform cell width of the Cartesian grid.
#   initial (numpy.ndarray): starting iterate, strictly positive inside the mask.
#   damping (float): relaxation weight in (0, 1] applied to the fixed-point map.
#   tol (float): convergence tolerance on the largest change between iterates.
#   max_iter (int): maximum number of iterations before the solver gives up.
# Returns:
#   dict: the stationary probability density on the masked grid, the number of
#   iterations, the final increment and a convergence flag.
def masked_steady_state(
    aggregation_ratio,  # Aggregation ratio kappa of the stationary identity.
    radius,  # Perceptual range R of the detection kernel.
    kernel,  # Detection kernel family.
    mask,  # Boolean array that is True inside the study area.
    cell,  # Uniform cell width of the Cartesian grid.
    initial=None,  # Starting iterate of the fixed-point solver.
    damping=0.5,  # Relaxation weight applied to the fixed-point map.
    tol=1.0e-10,  # Convergence tolerance on the largest change between iterates.
    max_iter=5000,  # Maximum number of iterations before the solver gives up.
):  # End of the argument list.
    inside = np.asarray(mask, dtype=bool)  # Boolean mask of the study area.
    area = float(np.sum(inside)) * cell * cell  # Area of the study area in the units of the grid.
    if area <= 0.0:  # An empty study area carries no probability density.
        raise ValueError("The study area mask contains no interior cell")  # Reject the input.
    if initial is None:  # No starting iterate was supplied by the caller.
        density = inside.astype(float) / area  # Uniform density on the study area.
    else:  # A starting iterate was supplied, usually the estimated utilisation distribution.
        density = np.asarray(initial, dtype=float) * inside  # Restrict the iterate to the mask.
        total = float(np.sum(density)) * cell * cell  # Mass of the supplied starting iterate.
        density = density / total if total > 0.0 else inside.astype(float) / area  # Normalise.
    increment = np.inf  # Largest change between successive iterates.
    iteration = 0  # Number of iterations performed so far.
    while iteration < int(max_iter) and increment > float(tol):  # Iterate until convergence.
        perceived = convolve_masked(density, radius, kernel, cell)  # Perceived map of the density.
        exponent = float(aggregation_ratio) * area * perceived  # Dimensionless exponent.
        exponent = np.where(inside, exponent - float(np.max(exponent[inside])), 0.0)  # Stabilise.
        candidate = np.exp(exponent) * inside  # Unnormalised image of the fixed-point map.
        candidate = candidate / (float(np.sum(candidate)) * cell * cell)  # Normalise to unit mass.
        updated = (1.0 - damping) * density + damping * candidate  # Damped update of the iterate.
        increment = float(np.max(np.abs(updated - density)))  # Largest change of this iteration.
        density = updated  # Accept the updated iterate.
        iteration = iteration + 1  # Count the iteration that has just been completed.
    # Assemble the stationary density together with the diagnostics of the solve.
    return {
        "density": density,  # Stationary probability density on the masked grid.
        "iterations": iteration,  # Number of fixed-point iterations performed.
        "increment": increment,  # Largest change between the last two iterates.
        "converged": bool(increment <= float(tol)),  # Whether the tolerance was reached.
    }
