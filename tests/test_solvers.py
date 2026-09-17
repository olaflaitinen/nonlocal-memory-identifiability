# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the forward solvers: mass conservation, spectral convergence
# on smooth data and agreement between the pseudo-spectral scheme and the
# positivity-preserving finite-volume scheme.
# Manuscript: Section 4.1, numerical solution; Section 5.1.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and elementary functions.
import pytest  # Parametrised test declaration and markers.

from nmi.finite_volume import simulate_fv_1d  # Positivity-preserving finite-volume scheme.
from nmi.grids import periodic_grid_1d, periodic_grid_2d  # Equispaced periodic grids.
from nmi.spectral_1d import PositivityError, simulate_1d  # One-dimensional spectral solver.
from nmi.spectral_2d import simulate_2d  # Two-dimensional spectral solver.

# Model parameters shared by the tests of this module.
PARAMS = {  # Model parameters of equation (1) used by the solver tests.
    "diffusion": 0.01,  # Diffusion rate d of the tested solves.
    "alpha": 0.01398,  # Advection strength alpha, taken from Table 3 at R = 0.15.
    "beta": 1.0,  # Memory uptake rate beta of the tested solves.
    "memory_decay": 1.0,  # Memory decay rate mu of the tested solves.
    "radius": 0.15,  # Perceptual range R of the tested solves.
}


# Mass is conserved to machine precision by the pseudo-spectral solver.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected mass error.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_mass_conservation_1d(kernel):  # Mass conservation of the spectral solver.
    n_points = 128  # Grid size of the tested solve.
    grid = periodic_grid_1d(n_points, 1.0)  # Grid points of the periodic domain.
    initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid)  # Initial density of the tested solve.
    result = simulate_1d(PARAMS, kernel, initial, 0.0, 10.0, 0.005, n_points, 1.0)  # Solve.
    assert result["mass_error"] < 1.0e-13  # The spectral scheme conserves the mass exactly.


# Mass is conserved to machine precision by the two-dimensional solver.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected mass error.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_mass_conservation_2d(kernel):  # Mass conservation in two space dimensions.
    n_points = 32  # Grid size of the tested solve in each coordinate direction.
    grid_x, grid_y = periodic_grid_2d(n_points, 1.0)  # Coordinates of the grid cells.
    initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid_x) * np.cos(2.0 * np.pi * grid_y)  # Data.
    result = simulate_2d(PARAMS, kernel, initial, 0.0, 2.0, 0.005, n_points, 1.0)  # Solve.
    assert result["mass_error"] < 1.0e-13  # The spectral scheme conserves the mass exactly.


# The spatial error decays faster than any power of the grid size on smooth data.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected spectral convergence.
def test_spectral_convergence_on_smooth_data():
    reference_points = 512  # Grid size of the reference solution of the comparison.
    reference_grid = periodic_grid_1d(reference_points, 1.0)  # Grid of the reference solution.
    reference_initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * reference_grid)  # Smooth initial data.
    reference = simulate_1d(  # Reference solve on the fine grid.
        PARAMS,  # Model parameters of the tested solves.
        "gaussian",  # Smooth detection kernel, which gives a smooth solution.
        reference_initial,  # Smooth initial density of the reference solve.
        0.0,  # Initial cognitive map of the reference solve.
        5.0,  # End of the reference solve.
        0.001,  # Time step of the reference solve.
        reference_points,  # Grid size of the reference solve.
        1.0,  # Length L of the periodic domain.
    )  # Reference solution against which the errors are measured.
    errors = []  # Relative errors of the coarse solves, one per inspected grid size.
    for n_points in (32, 64):  # Two coarse grids, both divisors of the reference grid.
        grid = periodic_grid_1d(n_points, 1.0)  # Grid points of the coarse solve.
        initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid)  # Smooth initial density.
        result = simulate_1d(PARAMS, "gaussian", initial, 0.0, 5.0, 0.001, n_points, 1.0)  # Solve.
        restricted = reference["final_density"][:: reference_points // n_points]  # Reference.
        # Relative error of the coarse solve against the restricted reference solution.
        errors.append(float(np.linalg.norm(result["final_density"] - restricted) / np.linalg.norm(restricted)))
    assert errors[1] <= errors[0]  # Refining the grid does not increase the error.
    assert errors[1] < 1.0e-10  # The smooth problem is resolved to near machine precision.


# The two independent schemes agree over a short time window.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected agreement.
@pytest.mark.slow
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])  # Both kernel families.
def test_spectral_agrees_with_finite_volume(kernel):  # Agreement of the two schemes.
    n_points = 128  # Grid size shared by the two compared solves.
    grid = periodic_grid_1d(n_points, 1.0)  # Grid points of the periodic domain.
    initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid)  # Initial density of both solves.
    cell = 1.0 / n_points  # Uniform cell width of the finite-volume grid.
    time_step = 0.2 * cell**2 / PARAMS["diffusion"]  # Explicit step inside the stability limit.
    volume = simulate_fv_1d(PARAMS, kernel, initial, 0.0, 10.0, time_step, 1.0, "periodic")  # FV.
    spectral = simulate_1d(PARAMS, kernel, initial, 0.0, 10.0, 0.001, n_points, 1.0)  # Spectral.
    difference = float(np.max(np.abs(volume["final_density"] - spectral["final_density"])))  # Gap.
    assert difference / float(np.max(spectral["final_density"])) < 5.0e-3  # Agreement.
    assert float(volume["min_density"]) > 0.0  # The finite-volume scheme preserves positivity.


# A run that loses positivity is rejected with an informative error.
# Arguments:
#   none.
# Returns:
#   None: the test asserts that the error is raised.
def test_positivity_monitor_rejects_negative_density():
    n_points = 64  # Grid size of the rejected solve.
    grid = periodic_grid_1d(n_points, 1.0)  # Grid points of the periodic domain.
    initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid)  # Initial density of the rejected solve.
    params = dict(PARAMS, alpha=800.0)  # An advection strength far beyond the admissible range.
    with pytest.raises(PositivityError, match="Density became negative"):  # Documented error.
        simulate_1d(params, "tophat", initial, 0.0, 5.0, 0.01, n_points, 1.0)  # Rejected solve.
