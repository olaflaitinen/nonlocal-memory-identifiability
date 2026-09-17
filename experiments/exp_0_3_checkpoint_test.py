# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify that an interrupted sampler resumes exactly. A reference
# chain is run without interruption and a second chain is stopped part way,
# checkpointed and resumed; the two chains must agree to machine precision.
# Manuscript: Section 4.6, computational resources; Experiment 0.3.
# Inputs: an experiment configuration. Outputs: a log line and a tracked
# summary row under results/summary/.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
import tempfile  # Temporary directory that holds the checkpoint of the test.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.io import write_csv  # Writer of the tracked summary table.
from nmi.mcmc_adaptive import AdaptiveMetropolis  # Sampler under test.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Checkpoint and resume test")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Unnormalised log posterior used by the test, a standard normal density.
# Arguments:
#   theta (numpy.ndarray): the state of the chain.
# Returns:
#   float: the log density of a standard normal distribution.
def test_log_posterior(theta):
    return float(-0.5 * np.sum(np.asarray(theta, dtype=float) ** 2))  # Standard normal density.


# Build a sampler with the settings of the test.
# Arguments:
#   settings (dict): the checkpoint_test block of the configuration.
# Returns:
#   AdaptiveMetropolis: a sampler in its initial state.
def build_sampler(settings):
    generator = np.random.default_rng(int(settings["seed"]))  # Generator of the proposals.
    return AdaptiveMetropolis(  # Sampler configured exactly as in the reference run.
        test_log_posterior,  # Standard normal log posterior used by the test.
        np.zeros(3),  # Starting point of the chain, at the origin.
        float(settings["initial_scale"]),  # Standard deviation of the initial proposal.
        int(settings["adaptation_start"]),  # Iteration at which the adaptation begins.
        rng=generator,  # Generator of the proposals of this sampler.
    )  # Sampler in its initial state.


# Entry point of the checkpoint and resume test.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero when the two chains agree and one otherwise.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["checkpoint_test"]  # Settings of the checkpoint and resume test.
    total = int(settings["n_iterations"])  # Number of iterations of the reference chain.
    interrupt = int(settings["interrupt_at"])  # Iteration at which the second chain stops.
    every = int(settings["checkpoint_every"])  # Number of iterations between two checkpoints.
    reference = build_sampler(settings).run(total)  # Uninterrupted reference chain.
    with tempfile.TemporaryDirectory() as folder:  # Temporary directory for the checkpoint.
        path = Path(folder) / "checkpoint_test.npz"  # Checkpoint file used by the test.
        build_sampler(settings).run(interrupt, path, every)  # Interrupted first block.
        resumed = build_sampler(settings).run(total, path, every, resume=True)  # Second block.
    difference = float(np.max(np.abs(reference["chain"] - resumed["chain"])))  # Largest difference.
    identical = difference == 0.0  # The resumed chain must agree to machine precision.
    record = {  # Single summary row describing the outcome of the test.
        "n_iterations": total,  # Number of iterations of the two chains.
        "interrupt_at": interrupt,  # Iteration at which the second chain was interrupted.
        "max_absolute_difference": difference,  # Largest difference between the two chains.
        "identical": identical,  # Whether the two chains agree exactly.
        "acceptance_rate": reference["acceptance_rate"],  # Acceptance rate of the reference chain.
    }
    summary = Path("results/summary") / f"{config['experiment']}_checkpoint_test.csv"  # Summary.
    write_csv(summary, [record], list(record.keys()))  # Write the tracked summary table.
    print(f"checkpoint test: largest difference {difference:.3e}, identical {identical}")  # Log.
    return 0 if identical else 1  # Signal failure when the resumed chain differs.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
