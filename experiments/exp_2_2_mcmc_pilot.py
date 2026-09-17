# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Sampler pilot that compares several proposal scales on a small
# number of conditions and reports the acceptance rate and the effective sample
# size per iteration, so that the settings of the full one-dimensional study
# can be fixed.
# Manuscript: Section 5.3, Experiment 2.2.
# Inputs: an experiment configuration and the generated synthetic data.
# Outputs: a tracked summary under results/summary/.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import design_1d  # Construction of the factorial design.
from nmi.diagnostics import convergence_statistics  # Convergence diagnostics of the chains.
from nmi.io import read_condition_data, write_csv  # Data reader and summary writer.
from nmi.likelihood import forward_transient_loglik, make_log_posterior, transient_context  # Maps.
from nmi.mcmc_adaptive import AdaptiveMetropolis  # Adaptive random-walk Metropolis sampler.
from nmi.priors import transient_prior  # Log-uniform prior on the four free parameters.

# Names of the sampled parameters, in the order used by every sampler.
PARAMETER_NAMES = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameter vector.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Sampler pilot")  # Argument parser of the script.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Run one pilot chain at a given proposal scale.
# Arguments:
#   condition (dict): one condition of the factorial design.
#   config (dict): the configuration mapping of the run.
#   scale (float): standard deviation of the initial diagonal proposal.
#   settings (dict): the pilot block of the configuration.
# Returns:
#   dict: the chain, its acceptance rate and the diagnostics of the run.
def run_pilot_chain(condition, config, scale, settings):
    data = read_condition_data(config["experiment"], condition["index"])  # Generated data.
    context = transient_context(condition, config)  # Context of the transient forward map.
    prior = transient_prior(condition["advection"], config["design"]["priors"])  # Prior of Table 3.
    observations = {  # Data mapping used by the transient log-likelihood.
        "observations": data["observations"],  # Noisy transient observations.
        "sigma": data["sigma"],  # Standard deviation of the observation noise.
    }
    # Unnormalised log posterior of the condition, in log space.
    log_posterior = make_log_posterior(
        lambda point: forward_transient_loglik(point, observations, context),  # Log-likelihood.
        prior,  # Log-uniform prior on the four free parameters.
    )  # Callable passed to the sampler.
    start = np.log(  # Start the chain at the true parameters of the condition.
        [  # True parameters in the order used by the samplers.
            float(condition["diffusion"]),  # True diffusion rate d.
            float(condition["advection"]),  # True combined advection strength gamma.
            float(condition["memory_decay"]),  # True memory decay rate mu.
            float(condition["radius"]),  # True perceptual range R.
        ]
    )  # Starting point of the pilot chain.
    sampler = AdaptiveMetropolis(  # Sampler configured with the current proposal scale.
        log_posterior,  # Unnormalised log posterior of the condition.
        start,  # Starting point of the pilot chain.
        float(scale),  # Standard deviation of the initial diagonal proposal.
        int(settings["adaptation_start"]),  # Iteration at which the adaptation begins.
        rng=np.random.default_rng(int(condition["seed"]) % (2**32)),  # Deterministic generator.
    )  # Sampler in its initial state.
    return sampler.run(int(settings["n_iterations"]))  # Chain and diagnostics of the pilot run.


# Entry point of the sampler pilot.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["pilot"]  # Settings of the sampler pilot.
    design = design_1d(config)  # Factorial design of the one-dimensional experiments.
    burn_in = float(settings["burn_in_fraction"])  # Fraction discarded as burn-in.
    records = []  # Accumulator for the rows of the pilot table.
    for raw_index in settings["condition_indices"]:  # Repeat the pilot for each chosen condition.
        index = int(raw_index) % len(design["transient"])  # Index inside the available design.
        condition = design["transient"][index]  # Condition used by this part of the pilot.
        for scale in settings["initial_scales"]:  # Compare the configured proposal scales.
            result = run_pilot_chain(condition, config, float(scale), settings)  # Pilot chain.
            chain = result["chain"]  # States of the pilot chain, one row per iteration.
            kept = chain[int(burn_in * chain.shape[0]) :]  # States retained after the burn-in.
            statistics = convergence_statistics(kept[None, :, :], PARAMETER_NAMES)  # Diagnostics.
            smallest = min(entry["ess_bulk"] for entry in statistics.values())  # Worst parameter.
            records.append(  # One row of the pilot table.
                {  # Diagnostics of the pilot chain at this proposal scale.
                    "condition_index": condition["index"],  # Index of the condition.
                    "identifier": condition["identifier"],  # Label of the condition.
                    "initial_scale": float(scale),  # Proposal scale of this pilot chain.
                    "n_iterations": int(settings["n_iterations"]),  # Length of the pilot chain.
                    "acceptance_rate": result["acceptance_rate"],  # Empirical acceptance rate.
                    "min_ess_bulk": float(smallest),  # Smallest effective sample size.
                    "ess_per_iteration": float(smallest) / max(1, kept.shape[0]),  # Efficiency.
                }
            )  # Row appended to the pilot table.
            print(f"pilot {condition['identifier']} scale {float(scale):.3f}: acceptance {result['acceptance_rate']:.3f}")
    summary = Path("results/summary") / f"{config['experiment']}_mcmc_pilot.csv"  # Summary path.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    best = max(records, key=lambda record: record["ess_per_iteration"])  # Most efficient setting.
    print(f"most efficient proposal scale: {best['initial_scale']}")  # Report the chosen setting.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
