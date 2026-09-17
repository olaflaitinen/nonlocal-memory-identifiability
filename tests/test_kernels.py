# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the Fourier transforms of the detection kernels against
# numerical quadrature of the kernels themselves, their value at the origin and
# the closed form of the two-dimensional top-hat transform.
# Manuscript: Section 2.2, equations (2) and (3); Remark 2.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and elementary functions.
import pytest  # Parametrised test declaration.
from scipy.special import j1  # Bessel function of the first kind, order one.

from nmi.kernels import (  # Kernel profiles and their Fourier transforms.
    kernel_hat_1d,  # Transform of the one-dimensional detection kernel.
    kernel_hat_2d,  # Transform of the two-dimensional detection kernel.
    kernel_values_1d,  # Profile of the one-dimensional detection kernel.
    kernel_values_2d,  # Profile of the two-dimensional detection kernel.
)


# The transform of each kernel equals one at the origin.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected value.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_transform_at_origin(kernel):  # Value of the transform at the origin.
    assert kernel_hat_1d(0.0, kernel) == pytest.approx(1.0)  # Unit mass of the kernel.
    assert kernel_hat_2d(0.0, kernel) == pytest.approx(1.0)  # Unit mass in two dimensions.


# The one-dimensional transform agrees with numerical quadrature of the kernel.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected agreement.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_transform_matches_quadrature(kernel):  # Transform against quadrature.
    radius = 0.15  # Perceptual range used by the quadrature comparison.
    extent = radius if kernel == "tophat" else 20.0 * radius  # Support of the top-hat kernel.
    displacement = np.linspace(-extent, extent, 200001)  # Fine quadrature grid of the support.
    values = kernel_values_1d(displacement, radius, kernel)  # Samples of the scaled kernel.
    for wavenumber in (2.0 * np.pi, 4.0 * np.pi, 10.0 * np.pi):  # Several torus wavenumbers.
        integrand = values * np.cos(wavenumber * displacement)  # Real part of the transform.
        quadrature = float(np.trapezoid(integrand, displacement))  # Numerical transform value.
        expected = float(kernel_hat_1d(radius * wavenumber, kernel))  # Closed form transform.
        assert quadrature == pytest.approx(expected, abs=1.0e-6)  # Agreement of the two values.


# The two-dimensional top-hat transform agrees with the closed form.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected agreement.
def test_two_dimensional_tophat_transform():
    arguments = np.linspace(0.1, 20.0, 200)  # Arguments at which the transform is compared.
    expected = 2.0 * j1(arguments) / arguments  # Closed form of Remark 2.
    assert np.allclose(kernel_hat_2d(arguments, "tophat"), expected)  # Agreement of the values.


# The two-dimensional kernels integrate to one over the plane.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected unit mass.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_two_dimensional_kernel_mass(kernel):  # Unit mass of the plane kernels.
    radius = 0.2  # Perceptual range used by the quadrature comparison.
    extent = 12.0 * radius  # Half width of the quadrature square around the origin.
    axis = np.linspace(-extent, extent, 2001)  # Quadrature grid along each coordinate axis.
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")  # Coordinates of the quadrature nodes.
    values = kernel_values_2d(np.hypot(grid_x, grid_y), radius, kernel)  # Samples of the kernel.
    cell = float(axis[1] - axis[0]) ** 2  # Area of one quadrature cell.
    assert float(np.sum(values) * cell) == pytest.approx(1.0, abs=2.0e-3)  # Unit mass.


# An unsupported kernel name is rejected with an informative error.
# Arguments:
#   none.
# Returns:
#   None: the test asserts that the error is raised.
def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="Unknown kernel"):  # The error names the offending value.
        kernel_hat_1d(1.0, "triangular")  # An unsupported kernel family.
