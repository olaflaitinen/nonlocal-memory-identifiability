# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify Proposition 4 on a bounded interval with no-flux boundaries.
# The finite-volume scheme is run to a stationary state and the quantity
# log u - kappa G_R * u is checked to be constant across the interval.
# Manuscript: Section 3.3, Proposition 4.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and elementary functions.
import pytest  # Parametrised test declaration and markers.

from nmi.finite_volume import convolve_bounded_1d, simulate_fv_1d  # Bounded domain solver.
from nmi.kernels import kernel_hat_1d  # Fourier transform of the detection kernel.


# The stationary state of the bounded problem satisfies the identity of Proposition 4.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts that the potential is constant.
@pytest.mark.slow
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])  # Both kernel families.
def test_bounded_identity(kernel):  # Stationary identity on a bounded interval.
    n_cells = 200  # Number of cells of the finite-volume grid.
    length = 1.0  # Length of the bounded interval of the test.
    cell = length / n_cells  # Uniform cell width of the finite-volume grid.
    centres = (np.arange(n_cells) + 0.5) * cell  # Centres of the finite-volume cells.
    initial = 1.0 + 0.05 * np.cos(np.pi * centres / length)  # Initial density of the solve.
    radius = 0.15  # Perceptual range R of the bounded solve.
    diffusion = 0.02  # Diffusion rate d of the bounded solve.
    # On the bounded interval the admissible perturbations are the cosine modes
    # with wavenumber m pi, so the onset value is formed from those modes.
    onset = 1.0 / float(np.max(kernel_hat_1d(radius * np.pi * np.arange(1, 65), kernel)))
    ratio = 1.2 * onset  # Twenty per cent above the onset, as in the design of Section 4.2.
    params = {  # Model parameters of equation (1) used by the bounded solve.
        "diffusion": diffusion,  # Diffusion rate d of the bounded solve.
        "alpha": ratio * diffusion,  # Advection strength implied by the aggregation ratio.
        "beta": 1.0,  # Memory uptake rate beta of the bounded solve.
        "memory_decay": 1.0,  # Memory decay rate mu of the bounded solve.
        "radius": radius,  # Perceptual range R of the bounded solve.
    }
    time_step = 0.2 * cell**2 / params["diffusion"]  # Explicit step inside the stability limit.
    result = simulate_fv_1d(  # Bounded finite-volume solve with no-flux boundaries.
        params,  # Model parameters of the bounded solve.
        kernel,  # Detection kernel family of the bounded solve.
        initial,  # Initial density of the bounded solve.
        0.0,  # Initial cognitive map of the bounded solve.
        300.0,  # End of the bounded solve, well beyond the relaxation time.
        time_step,  # Time step of the explicit finite-volume scheme.
        length,  # Length of the bounded interval.
        "noflux",  # No-flux boundary condition of Proposition 4.
    )  # Stationary state reached by the bounded solve.
    density = result["final_density"]  # Stationary density of the bounded solve.
    assert float(np.min(density)) > 0.0  # The scheme preserves positivity, as Section 4.1 states.
    assert result["mass_error"] < 1.0e-12  # No flux crosses the boundary of the interval.
    # Aggregation ratio kappa implied by the parameters of the bounded solve.
    ratio = params["alpha"] * params["beta"] / (params["diffusion"] * params["memory_decay"])
    perceived = convolve_bounded_1d(density, params["radius"], kernel, cell)  # Perceived map.
    potential = np.log(density) - ratio * perceived  # Quantity of Proposition 4.
    spread = float(np.max(potential) - np.min(potential))  # Variation of that quantity.
    assert spread / float(np.max(np.abs(potential))) < 1.0e-2  # The potential is constant.


# The bounded convolution agrees with a direct sum over the cells.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected agreement.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_bounded_convolution_matches_direct_sum(kernel):  # Correctness of the convolution.
    from nmi.kernels import kernel_values_1d  # Profile of the one-dimensional detection kernel.

    n_cells = 40  # Number of cells of the small grid used by the comparison.
    cell = 0.025  # Uniform cell width of the small grid.
    field = np.random.default_rng(3).random(n_cells)  # Arbitrary field on the bounded interval.
    computed = convolve_bounded_1d(field, 0.15, kernel, cell)  # Convolution under test.
    direct = np.zeros(n_cells)  # Accumulator for the direct double sum.
    for index in range(n_cells):  # Evaluate the convolution at each cell of the interval.
        offsets = (np.arange(n_cells) - index) * cell  # Displacements between the cell centres.
        # Direct quadrature of the bounded convolution at the current cell.
        direct[index] = float(np.sum(kernel_values_1d(offsets, 0.15, kernel) * field) * cell)
    assert np.allclose(computed, direct, atol=1.0e-12)  # Agreement of the two evaluations.
