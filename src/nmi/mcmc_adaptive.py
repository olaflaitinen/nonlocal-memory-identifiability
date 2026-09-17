# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Adaptive random-walk Metropolis sampler in log space, following the
# covariance adaptation of Haario, Saksman and Tamminen (2001). The sampler
# checkpoints its state atomically, resumes exactly and stops cleanly before a
# configured wall-time limit so that long chains can run in blocks on Slurm.
# Manuscript: Section 4.4, Bayesian inference; Section 4.6.
# Inputs: a log posterior callable and sampler settings. Outputs: the chain,
# the log posterior trace and the acceptance rate.

import time  # Wall clock measurement used by the wall-time guard.

import numpy as np  # Numerical arrays and linear algebra.

from nmi.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint  # Checkpoints.
from nmi.random_state import deserialise_generator, serialise_generator  # Generator state.


# Adaptive random-walk Metropolis sampler with checkpointing and resume.
# The proposal covariance is the scaled empirical covariance of the chain after
# an initial period in which a fixed diagonal proposal is used.
class AdaptiveMetropolis:
    # Construct the sampler from a log posterior and its settings.
    # Arguments:
    #   log_posterior (callable): unnormalised log posterior in log space.
    #   initial (numpy.ndarray): starting point of the chain, in log space.
    #   initial_scale (numpy.ndarray or float): diagonal of the first proposal.
    #   adaptation_start (int): iteration at which the adaptation begins.
    #   epsilon (float): regularisation added to the adapted covariance.
    #   rng (numpy.random.Generator): generator of the proposals.
    # Returns:
    #   None: the constructor stores the settings on the instance.
    def __init__(
        self,  # The sampler instance under construction.
        log_posterior,  # Unnormalised log posterior density in log space.
        initial,  # Starting point of the chain, in log space.
        initial_scale=0.1,  # Standard deviation of the initial diagonal proposal.
        adaptation_start=500,  # Iteration at which the covariance adaptation begins.
        epsilon=1.0e-10,  # Regularisation added to the adapted covariance.
        rng=None,  # Generator of the proposals, created when omitted.
    ):  # End of the argument list.
        self.log_posterior = log_posterior  # Unnormalised log posterior density.
        self.position = np.array(initial, dtype=float)  # Current state of the chain.
        self.dimension = self.position.size  # Number of parameters sampled by the chain.
        # Standard deviation of the fixed diagonal proposal used before adaptation.
        self.scale = np.broadcast_to(np.asarray(initial_scale, dtype=float), (self.dimension,))
        self.adaptation_start = int(adaptation_start)  # First iteration that uses the adaptation.
        self.epsilon = float(epsilon)  # Regularisation of the adapted covariance.
        self.rng = rng if rng is not None else np.random.default_rng()  # Proposal generator.
        self.factor = 2.38**2 / self.dimension  # Optimal scaling of Haario and co-authors.
        self.current_log_prob = float(self.log_posterior(self.position))  # Log density at the start.
        self.mean = self.position.copy()  # Running mean of the chain, used by the adaptation.
        self.covariance = np.diag(self.scale**2)  # Running covariance, used by the adaptation.
        self.iteration = 0  # Number of iterations completed so far.
        self.accepted = 0  # Number of accepted proposals so far.
        self.samples = []  # Accumulated states of the chain.
        self.log_probs = []  # Accumulated log posterior values of the chain.

    # Draw one proposal and accept or reject it.
    # Arguments:
    #   none.
    # Returns:
    #   None: the state of the sampler is updated in place.
    def step(self):
        if self.iteration < self.adaptation_start:  # The adaptation has not begun yet.
            proposal = self.position + self.rng.normal(0.0, self.scale)  # Diagonal proposal.
        else:  # The adapted covariance is used once enough states are available.
            # Empirical covariance scaled by the factor of Haario and co-authors.
            adapted = self.factor * (self.covariance + self.epsilon * np.eye(self.dimension))
            proposal = self.rng.multivariate_normal(self.position, adapted)  # Adapted proposal.
        proposed_log_prob = float(self.log_posterior(proposal))  # Log density at the proposal.
        ratio = proposed_log_prob - self.current_log_prob  # Logarithm of the acceptance ratio.
        if np.log(self.rng.uniform()) < ratio:  # Metropolis acceptance rule in log space.
            self.position = proposal  # Move the chain to the accepted proposal.
            self.current_log_prob = proposed_log_prob  # Record the log density of the new state.
            self.accepted = self.accepted + 1  # Count the accepted proposal.
        self.iteration = self.iteration + 1  # Count the completed iteration.
        self._update_moments()  # Update the running mean and covariance of the chain.
        self.samples.append(self.position.copy())  # Store the state of the chain.
        self.log_probs.append(self.current_log_prob)  # Store the log density of the state.

    # Update the running mean and covariance used by the adaptation.
    # Arguments:
    #   none.
    # Returns:
    #   None: the running moments are updated in place.
    def _update_moments(self):
        count = self.iteration  # Number of states that have entered the running moments.
        previous_mean = self.mean.copy()  # Running mean before the current state is added.
        self.mean = previous_mean + (self.position - previous_mean) / count  # Updated mean.
        if count < 2:  # The covariance is undefined for a single state.
            return  # Keep the initial covariance until a second state is available.
        deviation = (self.position - previous_mean).reshape(-1, 1)  # Deviation from the old mean.
        outer = deviation @ deviation.T  # Outer product entering the recursive update.
        # Recursive update of the empirical covariance of the chain.
        self.covariance = ((count - 2) * self.covariance + outer * (count - 1) / count) / (count - 1)

    # Run the sampler, with checkpointing, resume and a wall-time guard.
    # Arguments:
    #   n_iterations (int): total number of iterations of the chain.
    #   checkpoint_path (str or pathlib.Path): file used for the checkpoints.
    #   checkpoint_every (int): number of iterations between two checkpoints.
    #   wall_time_limit (float): wall time in seconds after which the sampler stops.
    #   resume (bool): whether an existing checkpoint is resumed.
    # Returns:
    #   dict: the chain, the log posterior trace, the acceptance rate and a flag
    #   that states whether the requested number of iterations was reached.
    def run(
        self,  # The sampler instance.
        n_iterations,  # Total number of iterations of the chain.
        checkpoint_path=None,  # File used for the checkpoints, when requested.
        checkpoint_every=1000,  # Number of iterations between two checkpoints.
        wall_time_limit=None,  # Wall time in seconds after which the sampler stops.
        resume=False,  # Whether an existing checkpoint is resumed.
    ):  # End of the argument list.
        started = time.perf_counter()  # Wall clock reading at the start of the run.
        # A previous block of this chain may already have written a checkpoint.
        if resume and checkpoint_path is not None and checkpoint_exists(checkpoint_path):
            self._restore(checkpoint_path)  # Continue exactly where the previous block stopped.
        while self.iteration < int(n_iterations):  # Iterate until the requested length is reached.
            self.step()  # Draw one proposal and accept or reject it.
            # The wall-time guard stops the sampler before the job limit is reached.
            reached_limit = wall_time_limit is not None and time.perf_counter() - started > wall_time_limit
            # A checkpoint is written at regular intervals of the chain.
            due = checkpoint_path is not None and self.iteration % int(checkpoint_every) == 0
            if due or reached_limit:  # A checkpoint is due or the wall-time guard has fired.
                if checkpoint_path is not None:  # Checkpointing was requested by the caller.
                    self._store(checkpoint_path)  # Write the state of the sampler atomically.
            if reached_limit:  # The wall-time guard stops the sampler cleanly.
                break  # Leave the loop so that the job can be continued by a dependent job.
        if checkpoint_path is not None:  # Record the final state for a later continuation.
            self._store(checkpoint_path)  # Write the state of the sampler atomically.
        # Assemble the chain together with the diagnostics of the run.
        return {
            "chain": np.array(self.samples),  # States of the chain, one row per iteration.
            "log_prob": np.array(self.log_probs),  # Log posterior value of each state.
            "acceptance_rate": self.accepted / max(1, self.iteration),  # Empirical acceptance rate.
            "iterations": self.iteration,  # Number of iterations completed by the sampler.
            "complete": self.iteration >= int(n_iterations),  # Whether the chain reached its length.
        }

    # Write the state of the sampler to a checkpoint.
    # Arguments:
    #   path (str or pathlib.Path): the checkpoint file to write.
    # Returns:
    #   None: the checkpoint is written as a side effect.
    def _store(self, path):
        # State of the sampler that is required to resume the chain exactly.
        arrays = {
            "position": self.position,  # Current state of the chain.
            "mean": self.mean,  # Running mean used by the covariance adaptation.
            "covariance": self.covariance,  # Running covariance used by the adaptation.
            "samples": np.array(self.samples),  # States accumulated so far.
            "log_probs": np.array(self.log_probs),  # Log densities accumulated so far.
        }
        # Scalar counters of the sampler, stored alongside the arrays.
        metadata = {
            "iteration": int(self.iteration),  # Number of completed iterations.
            "accepted": int(self.accepted),  # Number of accepted proposals.
            "current_log_prob": float(self.current_log_prob),  # Log density of the current state.
        }
        write_checkpoint(path, arrays, serialise_generator(self.rng), metadata)  # Atomic write.

    # Restore the state of the sampler from a checkpoint.
    # Arguments:
    #   path (str or pathlib.Path): the checkpoint file to read.
    # Returns:
    #   None: the state of the sampler is restored in place.
    def _restore(self, path):
        arrays, state, metadata = read_checkpoint(path)  # Read the stored state of the sampler.
        self.position = np.array(arrays["position"], dtype=float)  # Current state of the chain.
        self.mean = np.array(arrays["mean"], dtype=float)  # Running mean of the adaptation.
        self.covariance = np.array(arrays["covariance"], dtype=float)  # Running covariance.
        self.samples = [row.copy() for row in np.array(arrays["samples"], dtype=float)]  # States.
        self.log_probs = list(np.array(arrays["log_probs"], dtype=float))  # Log densities.
        self.iteration = int(metadata["iteration"])  # Number of completed iterations.
        self.accepted = int(metadata["accepted"])  # Number of accepted proposals.
        self.current_log_prob = float(metadata["current_log_prob"])  # Log density of the state.
        self.rng = deserialise_generator(state)  # Continue the same stream of random numbers.
