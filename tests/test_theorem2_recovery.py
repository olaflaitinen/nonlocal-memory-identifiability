# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the reconstruction of the parameters from the first two
# linearised Fourier modes given by Theorem 2, including the degenerate top-hat
# case at which the second multiplier vanishes.
# Manuscript: Section 3.4, Theorem 2.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and random number generation.
import pytest  # Approximate comparison of floating point values.

from nmi.kernels import kernel_hat_1d  # Fourier transform of the detection kernel.
from nmi.linear_theory import (  # Linearised modes and the reconstruction of Theorem 2.
    mode_matrix,  # Mode matrix of equation (6).
    mode_trajectory,  # Exact trajectory of one linearised Fourier mode.
    reconstruct_parameters_theorem2,  # Parameter reconstruction of Theorem 2.
)


# Build the observed trajectories of the first two linearised modes.
# Arguments:
#   diffusion (float): the diffusion rate d.
#   advection (float): the combined advection strength gamma.
#   memory_decay (float): the memory decay rate mu.
#   radius (float): the perceptual range R.
#   kernel (str): the detection kernel family.
#   times (numpy.ndarray): the times at which the modes are observed.
# Returns:
#   list: the trajectories of the first and of the second mode.
def observed_modes(diffusion, advection, memory_decay, radius, kernel, times):
    trajectories = []  # Accumulator for the trajectories of the two observed modes.
    for mode in (1, 2):  # Observe the first two modes, as Theorem 2 requires.
        matrix = mode_matrix(  # Mode matrix of equation (6) at this wavenumber.
            2.0 * np.pi * mode,  # Wavenumber xi_m of the current mode.
            diffusion,  # Diffusion rate d of the parameter set.
            advection,  # Advection strength alpha, with the uptake rate set to one.
            1.0,  # Memory uptake rate beta, set to one without loss of generality.
            memory_decay,  # Memory decay rate mu of the parameter set.
            radius,  # Perceptual range R of the parameter set.
            kernel,  # Detection kernel family of the parameter set.
            1.0,  # Mean density u_bar of the uniform state.
        )  # Mode matrix whose exact trajectory is evaluated.
        trajectories.append(mode_trajectory(matrix, np.array([1.0, 0.0]), times))  # Trajectory.
    return trajectories  # Trajectories of the first and of the second observed mode.


# The reconstruction recovers the parameters for random admissible parameter sets.
# Arguments:
#   kernel (str): the detection kernel family under test.
# Returns:
#   None: the test asserts the expected accuracy.
@pytest.mark.parametrize("kernel", ["tophat", "gaussian"])
def test_recovery_for_random_parameters(kernel):  # Accuracy of the reconstruction.
    rng = np.random.default_rng(2024)  # Deterministic generator of the parameter sets.
    times = np.linspace(0.0, 2.0, 40)  # Times at which the two modes are observed.
    inspected = 0  # Number of admissible parameter sets that were inspected.
    while inspected < 60:  # Inspect a fixed number of admissible parameter sets.
        diffusion = 10.0 ** rng.uniform(-3.0, -1.0)  # Diffusion rate d of this parameter set.
        memory_decay = 10.0 ** rng.uniform(-1.0, 1.0)  # Memory decay rate mu of this set.
        advection = 10.0 ** rng.uniform(-3.0, -1.0)  # Combined advection strength gamma.
        radius = rng.uniform(0.02, 0.48)  # Perceptual range R of this parameter set.
        if abs(float(kernel_hat_1d(radius * 4.0 * np.pi, kernel))) < 1.0e-2:  # Degenerate mode.
            continue  # The second mode carries no information at this parameter set.
        inspected += 1  # Count the admissible parameter set that is now inspected.
        # Exact trajectories of the first two modes at this parameter set.
        first, second = observed_modes(diffusion, advection, memory_decay, radius, kernel, times)
        recovered = reconstruct_parameters_theorem2(first, second, 1.0, 1.0, kernel)  # Theorem 2.
        assert recovered["diffusion"] == pytest.approx(diffusion, rel=1.0e-3)  # Diffusion rate.
        assert recovered["memory_decay"] == pytest.approx(memory_decay, rel=1.0e-3)  # Decay rate.
        assert recovered["advection"] == pytest.approx(advection, rel=1.0e-3)  # Advection.
        assert recovered["radius"] == pytest.approx(radius, rel=1.0e-3)  # Perceptual range.


# The degenerate top-hat case at R = L / 4 is recovered exactly.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected accuracy at the degenerate case.
def test_degenerate_tophat_case():
    times = np.linspace(0.0, 2.0, 40)  # Times at which the two modes are observed.
    diffusion, advection, memory_decay, radius = 0.01, 0.02, 1.0, 0.25  # Degenerate parameters.
    # The second multiplier vanishes exactly at this degenerate perceptual range.
    assert float(kernel_hat_1d(radius * 4.0 * np.pi, "tophat")) == pytest.approx(0.0, abs=1.0e-15)
    # Exact trajectories of the first two modes at the degenerate parameter set.
    first, second = observed_modes(diffusion, advection, memory_decay, radius, "tophat", times)
    recovered = reconstruct_parameters_theorem2(first, second, 1.0, 1.0, "tophat")  # Theorem 2.
    assert recovered["diffusion"] == pytest.approx(diffusion, rel=1.0e-6)  # Diffusion rate.
    assert recovered["memory_decay"] == pytest.approx(memory_decay, rel=1.0e-6)  # Decay rate.
    assert recovered["advection"] == pytest.approx(advection, rel=1.0e-6)  # Advection strength.
    assert recovered["radius"] == pytest.approx(radius, rel=1.0e-6)  # Perceptual range R = L / 4.
