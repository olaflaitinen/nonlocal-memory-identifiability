# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the instability criterion of Proposition 3 against direct
# eigenvalue computation for random parameter sets, and verify the critical
# aggregation ratio at which the criterion changes its verdict.
# Manuscript: Section 3.2, equation (6) and Proposition 3.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and random number generation.
import pytest  # Approximate comparison of floating point values.

from nmi.linear_theory import is_unstable, mode_matrix  # Linear theory of Section 3.2.
from nmi.steady_state import critical_kappa  # Onset value of the aggregation ratio.


# The criterion of Proposition 3 agrees with a direct eigenvalue computation.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected agreement.
def test_criterion_agrees_with_eigenvalues():
    rng = np.random.default_rng(12345)  # Deterministic generator of the parameter sets.
    for _ in range(300):  # Inspect one random parameter set at a time.
        diffusion = 10.0 ** rng.uniform(-3.0, -1.0)  # Diffusion rate d of this parameter set.
        memory_decay = 10.0 ** rng.uniform(-1.0, 1.0)  # Memory decay rate mu of this set.
        advection = 10.0 ** rng.uniform(-4.0, 0.0)  # Combined advection strength gamma.
        radius = rng.uniform(0.02, 0.48)  # Perceptual range R of this parameter set.
        kernel = "tophat" if rng.uniform() < 0.5 else "gaussian"  # Detection kernel family.
        ratio = advection / (diffusion * memory_decay)  # Aggregation ratio kappa of this set.
        predicted = is_unstable(ratio, radius, kernel, 1.0, 1.0, m_max=48)  # Proposition 3.
        observed = False  # Whether a direct eigenvalue computation finds an unstable mode.
        for mode in range(1, 49):  # Inspect the same modes as the criterion does.
            matrix = mode_matrix(  # Mode matrix of equation (6) at this wavenumber.
                2.0 * np.pi * mode,  # Wavenumber xi_m of the current mode.
                diffusion,  # Diffusion rate d of this parameter set.
                advection,  # Advection strength alpha, with the uptake rate set to one.
                1.0,  # Memory uptake rate beta, set to one without loss of generality.
                memory_decay,  # Memory decay rate mu of this parameter set.
                radius,  # Perceptual range R of this parameter set.
                kernel,  # Detection kernel family of this parameter set.
                1.0,  # Mean density u_bar of the uniform state.
            )  # Mode matrix whose spectrum is computed directly.
            if float(np.max(np.real(np.linalg.eigvals(matrix)))) > 0.0:  # An unstable eigenvalue.
                observed = True  # The direct computation finds an unstable mode.
                break  # No further mode needs to be inspected.
        assert predicted == observed  # The two verdicts must agree at every parameter set.


# The criterion changes its verdict exactly at the critical aggregation ratio.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected behaviour at the onset.
def test_verdict_changes_at_the_onset():
    for kernel in ("tophat", "gaussian"):  # Both detection kernel families of the manuscript.
        for radius in (0.075, 0.15, 0.30):  # Perceptual ranges of the design of Section 4.2.
            critical = critical_kappa(radius, kernel, 1.0, 1.0)  # Onset value of the ratio.
            assert not is_unstable(critical * 0.999, radius, kernel, 1.0, 1.0)  # Below the onset.
            assert is_unstable(critical * 1.001, radius, kernel, 1.0, 1.0)  # Above the onset.


# The determinant of the mode matrix vanishes exactly at the onset.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected determinant.
def test_determinant_vanishes_at_the_onset():
    radius = 0.15  # Perceptual range of the inspected condition.
    kernel = "tophat"  # Detection kernel family of the inspected condition.
    diffusion = 0.01  # Diffusion rate d of the inspected condition.
    memory_decay = 1.0  # Memory decay rate mu of the inspected condition.
    critical = critical_kappa(radius, kernel, 1.0, 1.0)  # Onset value of the ratio.
    advection = critical * diffusion * memory_decay  # Combined advection strength at the onset.
    matrix = mode_matrix(  # Mode matrix at the mode that becomes unstable first.
        2.0 * np.pi,  # Wavenumber xi_1 of the first mode on the unit torus.
        diffusion,  # Diffusion rate d of the inspected condition.
        advection,  # Advection strength alpha, with the uptake rate set to one.
        1.0,  # Memory uptake rate beta, set to one without loss of generality.
        memory_decay,  # Memory decay rate mu of the inspected condition.
        radius,  # Perceptual range R of the inspected condition.
        kernel,  # Detection kernel family of the inspected condition.
        1.0,  # Mean density u_bar of the uniform state.
    )  # Mode matrix at the onset of instability.
    assert float(np.linalg.det(matrix)) == pytest.approx(0.0, abs=1.0e-12)  # Vanishing determinant.
