# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Profile likelihood of every free parameter for the sixteen
# conditions of Table 6, under transient and under stationary data, together
# with the classification of each confidence interval.
# Manuscript: Section 5.3, Experiment 2.3; Table 6 and Fig. 4.
# Inputs: an experiment configuration and the generated synthetic data.
# Outputs: an HDF5 archive of the profiles and a tracked summary.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import design_1d  # Construction of the factorial design.
from nmi.io import ensure_dir, read_condition_data, run_metadata, write_csv, write_hdf5  # Output.
from nmi.likelihood import (  # Forward maps of the transient and stationary observations.
    forward_stationary_loglik,  # Log-likelihood of the stationary observations.
    forward_transient_loglik,  # Log-likelihood of the transient observations.
    stationary_context,  # Context mapping of the stationary forward map.
    transient_context,  # Context mapping of the transient forward map.
)
from nmi.priors import stationary_prior, transient_prior  # Log-uniform priors of Table 3.
from nmi.profile_likelihood import classify_interval, profile_parameter  # Profile likelihood.

# Names of the parameters profiled under transient data, in sampler order.
TRANSIENT_NAMES = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameter vector.
# Names of the parameters profiled under stationary data, in sampler order.
STATIONARY_NAMES = ("aggregation_ratio", "radius")  # Parameters of Theorem 1.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Profile likelihood")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--condition-index", type=int, default=None, help="one condition")  # Subset.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    parser.add_argument("--resume", action="store_true", help="skip completed conditions")  # Resume.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Build the log-likelihood, the prior and the true values of one condition.
# Arguments:
#   condition (dict): one profile likelihood condition of Table 6.
#   config (dict): the configuration mapping of the run.
# Returns:
#   tuple: the log-likelihood callable, the prior, the true parameters in log
#   space and the names of the profiled parameters.
def build_problem(condition, config):
    data = read_condition_data(config["experiment"], condition["index"])  # Generated data.
    if condition["data_type"] == "stationary":  # Stationary data of setting (S).
        context = stationary_context(condition, config)  # Context of the stationary forward map.
        # Prior on the two parameters that stationary data can determine.
        prior = stationary_prior(condition["aggregation_ratio"], config["design"]["priors"])
        observations = {  # Data mapping used by the stationary log-likelihood.
            "observations": data["stationary_observations"],  # Noisy stationary observations.
            "sigma": data["sigma"],  # Standard deviation of the observation noise.
        }
        # True stationary parameters of the condition, in log space.
        truth = np.log([float(condition["aggregation_ratio"]), float(condition["radius"])])
        # Log-likelihood of the stationary observations at parameters in log space.
        def evaluate(point):
            return forward_stationary_loglik(point, observations, context)  # Equation (9).

        return evaluate, prior, truth, STATIONARY_NAMES  # Problem of the stationary condition.
    context = transient_context(condition, config)  # Context of the transient forward map.
    prior = transient_prior(condition["advection"], config["design"]["priors"])  # Prior of Table 3.
    observations = {  # Data mapping used by the transient log-likelihood.
        "observations": data["observations"],  # Noisy transient observations.
        "sigma": data["sigma"],  # Standard deviation of the observation noise.
    }
    truth = np.log(  # True parameters of the condition in log space.
        [  # Reduced parameter vector in the order used by the samplers.
            float(condition["diffusion"]),  # True diffusion rate d.
            float(condition["advection"]),  # True combined advection strength gamma.
            float(condition["memory_decay"]),  # True memory decay rate mu.
            float(condition["radius"]),  # True perceptual range R.
        ]
    )  # True parameters of the transient condition.

    # Log-likelihood of the transient observations at parameters in log space.
    def evaluate_transient(point):
        return forward_transient_loglik(point, observations, context)  # Equation (9).

    return evaluate_transient, prior, truth, TRANSIENT_NAMES  # Problem of the transient condition.


# Entry point of the profile likelihood study.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["profile"]  # Settings of the profile likelihood study.
    design = design_1d(config)  # Factorial design of the one-dimensional experiments.
    conditions = design["stationary"]  # The profile likelihood conditions of Table 6.
    if arguments.condition_index is not None:  # A single condition was requested.
        conditions = [conditions[int(arguments.condition_index) % len(conditions)]]  # That one.
    # Directory that receives the archives of the computed profiles.
    directory = ensure_dir(arguments.output_dir or Path("results/raw") / str(config["experiment"]))
    records = []  # Accumulator for the rows of the classification table.
    for condition in conditions:  # Profile every parameter of each condition in turn.
        label = f"{condition['profile_index']:02d}_{condition['data_type']}"  # Label of the run.
        archive = directory / f"profile_{label}.h5"  # Archive of the profiles of this condition.
        if arguments.resume and archive.is_file():  # The condition has already been profiled.
            print(f"skipping completed condition {label}")  # Report the skipped condition.
            continue  # Continue with the next condition of the study.
        evaluate, prior, truth, names = build_problem(condition, config)  # Problem of the condition.
        # Deterministic generator of the random starting points of this condition.
        generator = np.random.default_rng(int(settings["seed"]) + condition["profile_index"])
        arrays = {}  # Arrays stored in the archive of this condition.
        for position, name in enumerate(names):  # Profile each parameter of the condition.
            outcome = profile_parameter(  # Profile likelihood of the current parameter.
                evaluate,  # Log-likelihood of the condition, in log space.
                prior.bounds(),  # Prior bounds of every parameter, in log space.
                position,  # Index of the parameter that is profiled.
                int(settings["n_grid"]),  # Number of grid values of the profiled parameter.
                int(settings["n_starts"]),  # Number of starting points of the optimisation.
                generator,  # Generator of the random starting points.
                truth,  # Warm start of the first grid value, at the true parameters.
            )  # Profile of the current parameter.
            # Classification of the confidence interval of the profiled parameter.
            verdict = classify_interval(outcome["grid"], outcome["profile"], float(settings["threshold"]))
            arrays[f"grid_{name}"] = outcome["grid"]  # Grid of the profiled parameter.
            arrays[f"profile_{name}"] = outcome["profile"]  # Profile log-likelihood values.
            arrays[f"optima_{name}"] = outcome["optima"]  # Optimal parameters at each grid value.
            records.append(  # One row of the classification table of Table 6.
                {  # Classification of the profiled parameter of this condition.
                    "profile_index": condition["profile_index"],  # Index within Table 6.
                    "condition_index": condition["index"],  # Index within the factorial design.
                    "identifier": condition["identifier"],  # Label of the condition.
                    "data_type": condition["data_type"],  # Transient or stationary data.
                    "kernel": condition["kernel"],  # Detection kernel family of the condition.
                    "radius": condition["radius"],  # Perceptual range R of the condition.
                    "noise_level": condition["noise_level"],  # Relative noise level eta.
                    "parameter": name,  # Name of the profiled parameter.
                    "classification": verdict["classification"],  # Practically identifiable or not.
                    # Lower end of the interval in natural units, when it exists.
                    "lower": "" if verdict["lower"] is None else float(np.exp(verdict["lower"])),
                    # Upper end of the interval in natural units, when it exists.
                    "upper": "" if verdict["upper"] is None else float(np.exp(verdict["upper"])),
                    # Largest profile value over the grid, when it exists.
                    "maximum_loglik": "" if verdict["maximum"] is None else verdict["maximum"],
                    "truth": float(np.exp(truth[position])),  # True value of the parameter.
                }
            )  # Row appended to the classification table.
            print(f"profile {label} {name}: {verdict['classification']}")  # Report the outcome.
        arrays["truth"] = truth  # True parameters of the condition, in log space.
        write_hdf5(archive, arrays, run_metadata(config, {"condition": label}))  # Archive.
    summary = Path("results/summary") / f"{config['experiment']}_profile_likelihood.csv"  # Path.
    if records:  # Nothing is written when every condition was skipped on resume.
        write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
        print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
