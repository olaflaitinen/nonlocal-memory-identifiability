# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the scaling symmetry of Proposition 1, both in the nonlinear
# pseudo-spectral solver and in the linearised Fourier modes.
# Manuscript: Section 3.1, Proposition 1.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and elementary functions.
import pytest  # Parametrised test declaration and approximate comparison.

from nmi.grids import periodic_grid_1d  # Equispaced grid of the periodic domain.
from nmi.linear_theory import mode_matrix, mode_trajectory  # Linearised mode of equation (6).
from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.


# The nonlinear density is unchanged by the transformation of Proposition 1.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected invariance.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_nonlinear_density_is_invariant(kernel):  # Invariance of the observed density.
    n_points = 128  # Grid size of the two compared forward solves.
    grid = periodic_grid_1d(n_points, 1.0)  # Grid points of the periodic domain.
    initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid)  # Initial density of both solves.
    base = {  # Model parameters of the reference solve.
        "diffusion": 0.01,  # Diffusion rate d of the reference solve.
        "alpha": 0.02,  # Advection strength alpha of the reference solve.
        "beta": 1.0,  # Memory uptake rate beta of the reference solve.
        "memory_decay": 1.0,  # Memory decay rate mu of the reference solve.
        "radius": 0.15,  # Perceptual range R of the reference solve.
    }
    factor = 4.3  # Factor c of the scaling symmetry of Proposition 1.
    scaled = dict(base, alpha=base["alpha"] / factor, beta=base["beta"] * factor)  # Transformed.
    first = simulate_1d(base, kernel, initial, 0.0, 5.0, 0.005, n_points, 1.0)  # Reference solve.
    second = simulate_1d(scaled, kernel, initial, 0.0, 5.0, 0.005, n_points, 1.0)  # Transformed.
    assert np.allclose(first["final_density"], second["final_density"], atol=1.0e-13)  # Invariance.


# The linearised density mode is unchanged by the transformation of Proposition 1.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected invariance.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_linear_mode_is_invariant(kernel):  # Invariance of the linearised density mode.
    wavenumber = 2.0 * np.pi  # Wavenumber xi_1 of the first mode on the unit torus.
    factor = 2.5  # Factor c of the scaling symmetry of Proposition 1.
    times = np.linspace(0.0, 3.0, 25)  # Times at which the two modes are compared.
    base = mode_matrix(wavenumber, 0.01, 0.02, 1.0, 1.0, 0.15, kernel, 1.0)  # Reference matrix.
    # Mode matrix of the transformed parameters of Proposition 1.
    scaled = mode_matrix(wavenumber, 0.01, 0.02 / factor, factor, 1.0, 0.15, kernel, 1.0)
    first, _, _ = mode_trajectory(base, np.array([1.0, 0.0]), times)  # Reference mode.
    second, _, _ = mode_trajectory(scaled, np.array([1.0, 0.0]), times)  # Transformed mode.
    assert np.allclose(first, second, atol=1.0e-12)  # The observed mode is unchanged.


# The combined advection strength is the only identifiable product.
# Arguments:
#   none.
# Returns:
#   None: the test asserts that the product determines the mode.
def test_only_the_product_matters():
    wavenumber = 2.0 * np.pi  # Wavenumber xi_1 of the first mode on the unit torus.
    times = np.linspace(0.0, 3.0, 25)  # Times at which the two modes are compared.
    first_matrix = mode_matrix(wavenumber, 0.01, 0.04, 0.5, 1.0, 0.15, "tophat", 1.0)  # Product.
    second_matrix = mode_matrix(wavenumber, 0.01, 0.005, 4.0, 1.0, 0.15, "tophat", 1.0)  # Same.
    first, _, _ = mode_trajectory(first_matrix, np.array([1.0, 0.0]), times)  # First mode.
    second, _, _ = mode_trajectory(second_matrix, np.array([1.0, 0.0]), times)  # Second mode.
    assert first[-1] == pytest.approx(second[-1], abs=1.0e-12)  # Equal observed modes.
