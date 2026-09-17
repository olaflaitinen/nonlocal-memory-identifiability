# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify that an interrupted sampler resumes exactly, and that the
# atomic checkpoint writer stores and restores the state of the bit generator.
# Manuscript: Section 4.6, computational resources; Experiment 0.3.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and random number generation.
import pytest  # Approximate comparison and temporary directory fixture.

from nmi.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint  # Checkpoints.
from nmi.mcmc_adaptive import AdaptiveMetropolis  # Adaptive random-walk Metropolis sampler.
from nmi.random_state import deserialise_generator, serialise_generator  # Generator state.


# Unnormalised log posterior used by the tests of this module.
# Arguments:
#   theta (numpy.ndarray): the state of the chain.
# Returns:
#   float: the log density of a standard normal distribution.
def standard_normal_log_posterior(theta):
    return float(-0.5 * np.sum(np.asarray(theta, dtype=float) ** 2))  # Standard normal density.


# Build a sampler with fixed settings, so that two runs are comparable.
# Arguments:
#   none.
# Returns:
#   AdaptiveMetropolis: a sampler in its initial state.
def build_sampler():
    return AdaptiveMetropolis(  # Sampler configured identically in every run of the test.
        standard_normal_log_posterior,  # Standard normal log posterior of the test.
        np.zeros(3),  # Starting point of the chain, at the origin.
        0.2,  # Standard deviation of the initial diagonal proposal.
        60,  # Iteration at which the covariance adaptation begins.
        rng=np.random.default_rng(4242),  # Deterministic generator of the proposals.
    )  # Sampler in its initial state.


# A chain that is interrupted and resumed equals an uninterrupted chain.
# Arguments:
#   tmp_path (pathlib.Path): the temporary directory fixture of pytest.
# Returns:
#   None: the test asserts the expected equality.
def test_resumed_chain_matches_uninterrupted_chain(tmp_path):
    reference = build_sampler().run(300)  # Uninterrupted reference chain of the test.
    path = tmp_path / "chain.npz"  # Checkpoint file used by the interrupted chain.
    build_sampler().run(120, path, 40)  # First block of the interrupted chain.
    assert checkpoint_exists(path)  # The first block wrote a readable checkpoint.
    resumed = build_sampler().run(300, path, 40, resume=True)  # Second block of that chain.
    assert np.array_equal(reference["chain"], resumed["chain"])  # The two chains are identical.
    assert reference["acceptance_rate"] == pytest.approx(resumed["acceptance_rate"])  # Same rate.


# The wall-time guard stops the sampler before the requested length is reached.
# Arguments:
#   tmp_path (pathlib.Path): the temporary directory fixture of pytest.
# Returns:
#   None: the test asserts the expected early stop.
def test_wall_time_guard_stops_the_sampler(tmp_path):
    sampler = build_sampler()  # Sampler whose run is stopped by the wall-time guard.
    result = sampler.run(10**7, tmp_path / "guard.npz", 100, wall_time_limit=0.2)  # Guarded run.
    assert not result["complete"]  # The guard stopped the run before the requested length.
    assert result["iterations"] < 10**7  # Fewer iterations were performed than were requested.
    assert checkpoint_exists(tmp_path / "guard.npz")  # The guard wrote a readable checkpoint.


# The checkpoint writer stores and restores the state of the bit generator.
# Arguments:
#   tmp_path (pathlib.Path): the temporary directory fixture of pytest.
# Returns:
#   None: the test asserts the expected restoration.
def test_generator_state_round_trip(tmp_path):
    generator = np.random.default_rng(17)  # Generator whose state is stored and restored.
    generator.standard_normal(5)  # Advance the generator before its state is recorded.
    path = tmp_path / "state.npz"  # Checkpoint file used by this test.
    # Write an array, the generator state and the metadata into one checkpoint.
    write_checkpoint(path, {"values": np.arange(4.0)}, serialise_generator(generator), {"step": 2})
    arrays, state, metadata = read_checkpoint(path)  # Contents of the written checkpoint.
    assert np.array_equal(arrays["values"], np.arange(4.0))  # The stored array is restored.
    assert metadata["step"] == 2  # The stored metadata are restored.
    restored = deserialise_generator(state)  # Generator restored from the stored state.
    assert np.array_equal(generator.standard_normal(3), restored.standard_normal(3))  # Same stream.


# A missing or unreadable checkpoint is reported as absent.
# Arguments:
#   tmp_path (pathlib.Path): the temporary directory fixture of pytest.
# Returns:
#   None: the test asserts the expected verdict.
def test_missing_checkpoint_is_reported_as_absent(tmp_path):
    assert not checkpoint_exists(tmp_path / "absent.npz")  # No checkpoint has been written.
    truncated = tmp_path / "truncated.npz"  # File that is not a valid checkpoint archive.
    truncated.write_bytes(b"not an archive")  # Write content that cannot be interpreted.
    assert not checkpoint_exists(truncated)  # An unreadable checkpoint counts as absent.
