# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Detection kernels of model (1) and their Fourier transforms on the
# line and on the plane, together with grid samples of the periodised kernel
# used for quadrature cross-checks.
# Manuscript: Section 2.2, equations (2) and (3); Remark 2 for two dimensions.
# Inputs: kernel names and transform arguments. Outputs: kernel values and
# Fourier transform values as NumPy arrays.

import numpy as np  # Numerical arrays and vectorised elementary functions.
from scipy.special import j1  # Bessel function of the first kind, order one.

# Names of the detection kernel families supported by the package.
KERNEL_NAMES = ("tophat", "gaussian")  # Top-hat and Gaussian detection kernels.


# Raise an informative error when an unsupported kernel name is supplied.
# Arguments:
#   kernel (str): the name that failed validation.
# Returns:
#   None: the function always raises ValueError.
def _reject_kernel(kernel):
    allowed = ", ".join(KERNEL_NAMES)  # Human readable list of the valid names.
    raise ValueError(f"Unknown kernel name {kernel!r}, expected one of: {allowed}")  # Reject.


# Fourier transform of the one-dimensional detection kernel on the line.
# Arguments:
#   s (numpy.ndarray or float): argument R * xi, dimensionless.
#   kernel (str): "tophat" or "gaussian".
# Returns:
#   numpy.ndarray: values of G_hat(s) as defined in equation (3).
def kernel_hat_1d(s, kernel):
    values = np.asarray(s, dtype=float)  # Accept scalars and arrays with a common type.
    if kernel == "tophat":  # Top-hat kernel G(y) = (1/2) 1_[-1,1](y).
        return np.sinc(values / np.pi)  # NumPy sinc(x) is sin(pi x)/(pi x), hence sin(s)/s.
    if kernel == "gaussian":  # Standard Gaussian density on the real line.
        return np.exp(-0.5 * values**2)  # Transform exp(-s^2/2) of the unit-variance density.
    _reject_kernel(kernel)  # Reject unsupported kernel names explicitly.


# Fourier transform of the two-dimensional detection kernel on the plane.
# Arguments:
#   s_norm (numpy.ndarray or float): Euclidean norm of the argument R * xi.
#   kernel (str): "tophat" or "gaussian".
# Returns:
#   numpy.ndarray: values of G_hat(s) as defined in Remark 2.
def kernel_hat_2d(s_norm, kernel):
    values = np.asarray(s_norm, dtype=float)  # Accept scalars and arrays with a common type.
    if kernel == "tophat":  # Normalised indicator of the unit disc in the plane.
        small = values < 1.0e-8  # Arguments at which the ratio 2 J_1(s)/s is evaluated by its limit.
        safe = np.where(small, 1.0, values)  # Avoid a division by zero in the vectorised branch.
        transform = 2.0 * j1(safe) / safe  # Transform 2 J_1(|s|)/|s| of the disc kernel.
        return np.where(small, 1.0, transform)  # The limit at the origin equals one.
    if kernel == "gaussian":  # Standard Gaussian density on the plane.
        return np.exp(-0.5 * values**2)  # Transform exp(-|s|^2/2) of the isotropic density.
    _reject_kernel(kernel)  # Reject unsupported kernel names explicitly.


# Values of the scaled one-dimensional kernel G_R on the real line.
# Arguments:
#   y (numpy.ndarray): displacement in the same units as the perceptual range.
#   radius (float): perceptual range R, strictly positive.
#   kernel (str): "tophat" or "gaussian".
# Returns:
#   numpy.ndarray: values of G_R(y) = R^(-1) G(y / R) from equation (2).
def kernel_values_1d(y, radius, kernel):
    displacement = np.asarray(y, dtype=float)  # Accept scalars and arrays with a common type.
    scaled = displacement / radius  # Dimensionless displacement used inside the profile.
    if kernel == "tophat":  # Top-hat kernel with unit mass on the interval of half width R.
        inside = np.abs(scaled) <= 1.0  # Indicator of the perceptual interval.
        return np.where(inside, 0.5 / radius, 0.0)  # Constant density inside, zero outside.
    if kernel == "gaussian":  # Gaussian kernel with standard deviation equal to R.
        return np.exp(-0.5 * scaled**2) / (radius * np.sqrt(2.0 * np.pi))  # Normalised density.
    _reject_kernel(kernel)  # Reject unsupported kernel names explicitly.


# Samples of the periodised one-dimensional kernel on a periodic grid.
# Arguments:
#   x (numpy.ndarray): grid points on the torus of length domain_length.
#   radius (float): perceptual range R, strictly positive.
#   kernel (str): "tophat" or "gaussian".
#   domain_length (float): length L of the periodic domain.
#   images (int): number of periodic images summed on each side of the origin.
# Returns:
#   numpy.ndarray: samples of the periodised kernel used for quadrature checks.
def kernel_on_grid_1d(x, radius, kernel, domain_length, images=8):
    points = np.asarray(x, dtype=float)  # Grid points at which the kernel is evaluated.
    total = np.zeros_like(points)  # Accumulator for the sum over the periodic images.
    shifts = np.arange(-images, images + 1)  # Indices of the periodic images to be summed.
    for shift in shifts:  # Accumulate the contribution of each periodic image.
        displaced = points + shift * domain_length  # Displacement of the current image.
        total = total + kernel_values_1d(displaced, radius, kernel)  # Add the image contribution.
    return total  # Periodised kernel that represents G_R on the torus.


# Values of the scaled two-dimensional kernel G_R on the plane.
# Arguments:
#   distance (numpy.ndarray): Euclidean distance from the origin.
#   radius (float): perceptual range R, strictly positive.
#   kernel (str): "tophat" or "gaussian".
# Returns:
#   numpy.ndarray: values of G_R(y) = R^(-2) G(y / R) from Remark 2.
def kernel_values_2d(distance, radius, kernel):
    separation = np.asarray(distance, dtype=float)  # Accept scalars and arrays with a common type.
    if kernel == "tophat":  # Normalised indicator of the disc of radius R.
        inside = separation <= radius  # Indicator of the perceptual disc.
        return np.where(inside, 1.0 / (np.pi * radius**2), 0.0)  # Constant density on the disc.
    if kernel == "gaussian":  # Isotropic Gaussian kernel with standard deviation R.
        factor = 1.0 / (2.0 * np.pi * radius**2)  # Normalising constant of the density.
        return factor * np.exp(-0.5 * (separation / radius) ** 2)  # Isotropic Gaussian density.
    _reject_kernel(kernel)  # Reject unsupported kernel names explicitly.
