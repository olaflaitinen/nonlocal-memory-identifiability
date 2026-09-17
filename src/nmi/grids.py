# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Periodic grids, wavenumbers and dealiasing masks for the Fourier
# pseudo-spectral discretisation of model (1) in one and two space dimensions.
# Manuscript: Section 4.1, numerical solution.
# Inputs: grid sizes and domain lengths. Outputs: grids, wavenumber arrays and
# boolean dealiasing masks as NumPy arrays.

import numpy as np  # Numerical arrays and fast Fourier transform helpers.


# Equispaced grid on the one-dimensional torus of length L.
# Arguments:
#   n_points (int): number of grid points N.
#   domain_length (float): length L of the periodic domain.
# Returns:
#   numpy.ndarray: the N grid points x_i = i L / N.
def periodic_grid_1d(n_points, domain_length):
    return np.arange(n_points) * (domain_length / n_points)  # Left endpoints of the N cells.


# Equispaced tensor product grid on the two-dimensional torus of side L.
# Arguments:
#   n_points (int): number of grid points N in each coordinate direction.
#   domain_length (float): side length L of the periodic square.
# Returns:
#   tuple: two arrays of shape (N, N) holding the first and second coordinates.
def periodic_grid_2d(n_points, domain_length):
    axis = periodic_grid_1d(n_points, domain_length)  # Shared one-dimensional coordinate axis.
    return np.meshgrid(axis, axis, indexing="ij")  # Coordinates in matrix index order.


# Wavenumbers of the real fast Fourier transform on the one-dimensional torus.
# Arguments:
#   n_points (int): number of grid points N.
#   domain_length (float): length L of the periodic domain.
# Returns:
#   numpy.ndarray: the wavenumbers xi_m = 2 pi m / L for m = 0, ..., N // 2.
def wavenumbers_1d(n_points, domain_length):
    modes = np.fft.rfftfreq(n_points, d=domain_length / n_points)  # Cycles per unit length.
    return 2.0 * np.pi * modes  # Convert cycles per unit length into angular wavenumbers.


# Wavenumbers of the real fast Fourier transform on the two-dimensional torus.
# Arguments:
#   n_points (int): number of grid points N in each coordinate direction.
#   domain_length (float): side length L of the periodic square.
# Returns:
#   tuple: arrays xi_x and xi_y of shape (N, N // 2 + 1) with the two components.
def wavenumbers_2d(n_points, domain_length):
    full = np.fft.fftfreq(n_points, d=domain_length / n_points) * 2.0 * np.pi  # Full axis.
    half = wavenumbers_1d(n_points, domain_length)  # Half axis of the real transform.
    return np.meshgrid(full, half, indexing="ij")  # Components in the layout of rfft2.


# Boolean two-thirds dealiasing mask for the one-dimensional real transform.
# Arguments:
#   n_points (int): number of grid points N.
# Returns:
#   numpy.ndarray: boolean array of shape (N // 2 + 1) that retains the low modes.
def dealias_mask_1d(n_points):
    modes = np.arange(n_points // 2 + 1)  # Mode indices retained by the real transform.
    cutoff = n_points // 3  # Largest retained mode index under the two-thirds rule.
    return modes <= cutoff  # Retain the modes below the cutoff and discard the rest.


# Boolean two-thirds dealiasing mask for the two-dimensional real transform.
# Arguments:
#   n_points (int): number of grid points N in each coordinate direction.
# Returns:
#   numpy.ndarray: boolean array of shape (N, N // 2 + 1) that retains low modes.
def dealias_mask_2d(n_points):
    signed = np.fft.fftfreq(n_points, d=1.0 / n_points)  # Signed mode indices of the full axis.
    half = np.arange(n_points // 2 + 1)  # Mode indices of the half axis of the real transform.
    cutoff = n_points // 3  # Largest retained mode index under the two-thirds rule.
    keep_first = np.abs(signed) <= cutoff  # Retained modes along the first coordinate.
    keep_second = half <= cutoff  # Retained modes along the second coordinate.
    return keep_first[:, None] & keep_second[None, :]  # Tensor product of the two criteria.


# Uniform cell width of a periodic grid.
# Arguments:
#   n_points (int): number of grid points N.
#   domain_length (float): length L of the periodic domain.
# Returns:
#   float: the cell width L / N used by the finite volume schemes.
def cell_width(n_points, domain_length):
    return domain_length / n_points  # Uniform spacing between neighbouring grid points.
