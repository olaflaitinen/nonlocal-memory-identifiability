# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: One-dimensional Bayesian inference for every condition of the
# factorial design. Four independent adaptive Metropolis chains are run per
# condition in log space, with checkpointing, resume and a wall-time guard, so
# that long runs proceed in blocks on the cluster.
# Manuscript: Section 5.3, Experiment 2.4; Fig. 5 and Online Resource 1.
# Inputs: an experiment configuration and the generated synthetic data.
# Outputs: one HDF5 archive of chains per condition and a tracked summary.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import design_1d  # Construction of the factorial design.
from nmi.diagnostics import convergence_statistics, has_converged, posterior_summary  # Diagnostics.
from nmi.io import ensure_dir, read_condition_data, run_metadata, write_csv, write_hdf5  # Output.
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
    parser = argparse.ArgumentParser(description="One-dimensional inference")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--condition-index", type=int, default=None, help="one condition")  # Subset.
    parser.add_argument("--chain-index", type=int, default=None, help="one chain")  # Subset.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints")  # Resume.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# True parameters of one condition, in log space.
# Arguments:
#   condition (dict): one condition of the factorial design.
# Returns:
#   numpy.ndarray: the logarithms of (d, gamma, mu, R), in that order.
def true_parameters(condition):
    return np.log(  # True parameters of the condition in log space.
        [  # Reduced parameter vector in the order used by the samplers.
            float(condition["diffusion"]),  # True diffusion rate d.
            float(condition["advection"]),  # True combined advection strength gamma.
            float(condition["memory_decay"]),  # True memory decay rate mu.
            float(condition["radius"]),  # True perceptual range R.
        ]
    )  # Parameter vector used to start the chains and to form the relative widths.


# Run the chains of one condition and write them to an archive.
# Arguments:
#   condition (dict): one condition of the factorial design.
#   config (dict): the configuration mapping of the run.
#   directory (pathlib.Path): the directory that receives the archive.
#   chain_index (int): a single chain to run, or None for every chain.
#   resume (bool): whether existing checkpoints are continued.
# Returns:
#   dict: a summary record of the condition, or None when the run is partial.
def run_condition(condition, config, directory, chain_index, resume):
    settings = config["mcmc"]  # Settings of the one-dimensional sampler.
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
    )  # Callable passed to every chain of this condition.
    truth = true_parameters(condition)  # True parameters of the condition, in log space.
    n_chains = int(settings["n_chains"])  # Number of independent chains per condition.
    selected = range(n_chains) if chain_index is None else [int(chain_index)]  # Chains to run.
    chains = []  # Accumulator for the chains of this condition.
    complete = True  # Whether every requested chain reached its full length.
    acceptance = []  # Acceptance rate of each chain of this condition.
    for index in selected:  # Run each selected chain of the condition.
        generator = np.random.default_rng([int(condition["seed"]) % (2**32), index])  # Chain seed.
        start = truth + 0.05 * generator.standard_normal(truth.size)  # Dispersed starting point.
        sampler = AdaptiveMetropolis(  # Sampler of this chain.
            log_posterior,  # Unnormalised log posterior of the condition.
            start,  # Dispersed starting point of this chain.
            float(settings["initial_scale"]),  # Standard deviation of the initial proposal.
            int(settings["adaptation_start"]),  # Iteration at which the adaptation begins.
            rng=generator,  # Generator of the proposals of this chain.
        )  # Sampler in its initial state.
        checkpoint = Path("checkpoints") / str(config["experiment"])  # Directory of checkpoints.
        checkpoint.mkdir(parents=True, exist_ok=True)  # Create the directory when it is absent.
        result = sampler.run(  # Run the chain with checkpointing and the wall-time guard.
            int(settings["n_iterations"]),  # Number of iterations of the chain.
            checkpoint / f"chain_{condition['index']:03d}_{index}.npz",  # Checkpoint of the chain.
            int(settings["checkpoint_every"]),  # Number of iterations between two checkpoints.
            settings["wall_time_limit"],  # Wall time after which the chain stops cleanly.
            resume,  # Whether an existing checkpoint is continued.
        )  # Chain and diagnostics of this run.
        chains.append(result["chain"])  # Store the chain of this run.
        acceptance.append(result["acceptance_rate"])  # Store the acceptance rate of this run.
        complete = complete and bool(result["complete"])  # Track whether the block finished.
        print(f"condition {condition['index']:03d} chain {index}: {result['iterations']} iterations")
    if chain_index is not None or not complete:  # A partial run produces no summary row.
        return None  # The summary is written once every chain of the condition is complete.
    stacked = np.stack(chains)  # Chains of shape (chains, iterations, parameters).
    burn_in = int(float(settings["burn_in_fraction"]) * stacked.shape[1])  # Discarded iterations.
    kept = np.exp(stacked[:, burn_in:, :])  # Retained draws, converted to natural units.
    statistics = convergence_statistics(kept, PARAMETER_NAMES)  # Convergence diagnostics.
    truth_map = {name: float(np.exp(truth[position])) for position, name in enumerate(PARAMETER_NAMES)}
    summary = posterior_summary(kept, PARAMETER_NAMES, truth_map)  # Posterior summaries.
    write_hdf5(  # Archive of the chains of this condition.
        directory / f"chains_{condition['index']:03d}.h5",  # Archive of this condition.
        {"chains": stacked, "truth": truth},  # Chains in log space and the true parameters.
        run_metadata(config, {"condition": condition["identifier"]}),  # Metadata of the archive.
    )  # Archive written for the diagnostics and the figure scripts.
    record = {  # Summary row of this condition, written to the tracked summary table.
        "condition_index": condition["index"],  # Stable integer index of the condition.
        "identifier": condition["identifier"],  # Stable textual label of the condition.
        "kernel": condition["kernel"],  # Detection kernel family of the condition.
        "radius": condition["radius"],  # Perceptual range R of the condition.
        "noise_level": condition["noise_level"],  # Relative noise level eta of the condition.
        "sampling": condition["sampling"],  # Name of the sampling design of the condition.
        "acceptance_rate": float(np.mean(acceptance)),  # Mean acceptance rate over the chains.
        "converged": has_converged(statistics),  # Whether the criteria of Section 4.4 hold.
    }
    for name in PARAMETER_NAMES:  # Record the diagnostics and summaries of every parameter.
        record[f"rhat_{name}"] = statistics[name]["rhat"]  # Rank-normalised split R-hat.
        record[f"ess_{name}"] = statistics[name]["ess_bulk"]  # Bulk effective sample size.
        record[f"median_{name}"] = summary[name]["median"]  # Posterior median of the parameter.
        record[f"lower_{name}"] = summary[name]["lower"]  # Lower end of the credible interval.
        record[f"upper_{name}"] = summary[name]["upper"]  # Upper end of the credible interval.
        record[f"width_{name}"] = summary[name]["relative_width"]  # Relative interval width.
        record[f"truth_{name}"] = summary[name]["truth"]  # True value of the parameter.
    return record  # Summary row of this condition.


# Entry point of the one-dimensional inference study.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    design = design_1d(config)  # Factorial design of the one-dimensional experiments.
    conditions = design["transient"]  # Transient conditions of the factorial design.
    if arguments.condition_index is not None:  # A single condition was requested.
        conditions = [conditions[int(arguments.condition_index) % len(conditions)]]  # That one.
    directory = ensure_dir(arguments.output_dir or Path("results/raw") / str(config["experiment"]))
    records = []  # Accumulator for the summary rows of the completed conditions.
    for condition in conditions:  # Run the chains of each selected condition in turn.
        record = run_condition(condition, config, directory, arguments.chain_index, arguments.resume)
        if record is not None:  # The condition completed every one of its chains.
            records.append(record)  # Store the summary row of the condition.
    summary = Path("results/summary") / f"{config['experiment']}_mcmc_1d.csv"  # Summary path.
    if records:  # Nothing is written when every requested run was partial.
        write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
        print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
