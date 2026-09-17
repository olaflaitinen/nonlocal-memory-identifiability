# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the stationary identity of equation (7) at the steady states
# of the design of Table 3, the invariance of the steady state under changes of
# the parameters that keep the aggregation ratio fixed, and the agreement of
# the fixed-point solver with time integration.
# Manuscript: Section 3.3, equation (7) and Theorem 1; Section 5.1.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and elementary functions.
import pytest  # Parametrised test declaration and approximate comparison.

from nmi.grids import periodic_grid_1d  # Equispaced grid of the periodic domain.
from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.
from nmi.steady_state import (  # Steady state solver and the stationary identity.
    continuation_in_kappa,  # Continuation of the branch from the onset value.
    critical_kappa,  # Onset value of the aggregation ratio.
    fixed_point_steady_state,  # Fixed-point characterisation of the steady state.
    identity_residual,  # Residual of the stationary identity of equation (7).
)

# Conditions of Table 3 at which the identity is verified at the tested resolution.
CONDITIONS = [("tophat", 0.15), ("tophat", 0.30), ("gaussian", 0.15), ("gaussian", 0.30)]


# Steady state of one condition of Table 3, obtained by continuation.
# Arguments:
#   kernel (str): the detection kernel family of the condition.
#   radius (float): the perceptual range R of the condition.
#   n_points (int): the grid size of the fixed-point solver.
# Returns:
#   tuple: the converged steady state and the aggregation ratio of the condition.
def steady_state_of(kernel, radius, n_points=256):
    critical = critical_kappa(radius, kernel, 1.0, 1.0)  # Onset value of the aggregation ratio.
    ratio = 1.2 * critical  # Aggregation ratio of equation (8) with the factor of Section 4.2.
    branch = continuation_in_kappa(  # Continuation of the branch from the onset value.
        np.linspace(critical * 1.01, ratio, 12),  # Aggregation ratios of the continuation path.
        radius,  # Perceptual range R of the condition.
        kernel,  # Detection kernel family of the condition.
        int(n_points),  # Grid size of the fixed-point solver.
    )  # Steady states along the continuation path.
    return branch[-1], ratio  # Steady state at the requested ratio and that ratio itself.


# The stationary identity of equation (7) holds at the computed steady states.
# Arguments:
#   kernel (str): the detection kernel family of the condition.
#   radius (float): the perceptual range R of the condition.
# Returns:
#   None: the test asserts the expected residual.
@pytest.mark.parametrize(("kernel", "radius"), CONDITIONS)
def test_identity_residual_is_small(kernel, radius):  # Residual of the stationary identity.
    result, ratio = steady_state_of(kernel, radius)  # Steady state of this condition.
    residual = identity_residual(result["density"], ratio, radius, kernel, 1.0)  # Residual.
    assert residual < 1.0e-8  # The identity of equation (7) holds at the tested resolution.
    assert result["dominant_mode"] == 1  # Every condition of Table 3 forms a single aggregate.


# The steady state depends on the parameters only through the aggregation ratio.
# Arguments:
#   kernel (str): the detection kernel family of the condition.
#   radius (float): the perceptual range R of the condition.
# Returns:
#   None: the test asserts the expected invariance.
@pytest.mark.parametrize(("kernel", "radius"), CONDITIONS)
def test_steady_state_depends_only_on_kappa(kernel, radius):  # Invariance of Theorem 1(i).
    result, ratio = steady_state_of(kernel, radius)  # Steady state of this condition.
    diffusion, memory_decay = 0.01 * 6.1, 1.0 / 3.7  # Transformed diffusion and decay rates.
    advection = ratio * diffusion * memory_decay  # Combined advection strength of that triple.
    transformed = advection / (diffusion * memory_decay)  # Aggregation ratio of the triple.
    recomputed = fixed_point_steady_state(  # Steady state of the transformed parameter triple.
        transformed,  # Aggregation ratio implied by the transformed triple.
        radius,  # Perceptual range R of the condition.
        kernel,  # Detection kernel family of the condition.
        np.maximum(result["density"], 1.0e-12),  # Reference steady state as the first iterate.
    )  # Steady state of the transformed triple.
    assert np.allclose(recomputed["density"], result["density"], atol=1.0e-10)  # Invariance.


# The fixed-point steady state agrees with the state reached by time integration.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected agreement.
@pytest.mark.slow
def test_fixed_point_matches_time_integration():  # Agreement of the two solvers.
    kernel, radius, n_points = "tophat", 0.15, 256  # Condition and resolution of the comparison.
    result, ratio = steady_state_of(kernel, radius, n_points)  # Fixed-point steady state.
    grid = periodic_grid_1d(n_points, 1.0)  # Grid points of the periodic domain.
    initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid)  # Initial density of the transient solve.
    params = {  # Model parameters of the transient solve of this condition.
        "diffusion": 0.01,  # Diffusion rate d of the condition.
        "alpha": ratio * 0.01,  # Advection strength alpha, since the uptake rate is one.
        "beta": 1.0,  # Memory uptake rate beta of the condition.
        "memory_decay": 1.0,  # Memory decay rate mu of the condition.
        "radius": radius,  # Perceptual range R of the condition.
    }
    transient = simulate_1d(params, kernel, initial, 0.0, 400.0, 0.005, n_points, 1.0)  # Solve.
    polished = fixed_point_steady_state(  # Steady state started from the transient state.
        ratio,  # Aggregation ratio kappa of the condition.
        radius,  # Perceptual range R of the condition.
        kernel,  # Detection kernel family of the condition.
        np.maximum(transient["final_density"], 1.0e-12),  # Transient state as the first iterate.
    )  # Steady state on the branch selected by the transient solution.
    difference = float(np.max(np.abs(polished["density"] - transient["final_density"])))  # Gap.
    assert difference / float(np.max(polished["density"])) < 1.0e-4  # Agreement of the two states.
    assert polished["dominant_mode"] == result["dominant_mode"]  # Same branch of the steady state.
