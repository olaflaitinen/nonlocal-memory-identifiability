# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Two-dimensional Bayesian inference for the conditions selected in
# Experiment 3.1, using the affine-invariant ensemble sampler with an HDF5
# backend, checkpointing, resume and a wall-time guard.
# Manuscript: Section 5.4, Experiment 3.2; Fig. 6.
# Inputs: an experiment configuration and the data of Experiment 3.1.
# Outputs: one HDF5 backend per condition and a tracked summary.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.diagnostics import convergence_statistics, has_converged, posterior_summary  # Diagnostics.
from nmi.io import ensure_dir, read_hdf5, write_csv  # Archive reader and summary writer.
from nmi.likelihood import gaussian_loglik  # Gaussian log-likelihood of equation (9).
from nmi.mcmc_ensemble import initial_positions, run_ensemble  # Ensemble sampler wrapper.
from nmi.priors import transient_prior  # Log-uniform prior on the four free parameters.
from nmi.spectral_2d import simulate_2d  # Two-dimensional pseudo-spectral solver.

from experiments.exp_3_1_pilot_2d import conditions_2d, initial_density_2d  # Shared design helpers.

# Names of the sampled parameters, in the order used by every sampler.
PARAMETER_NAMES = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameter vector.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Two-dimensional inference")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--condition-index", type=int, default=None, help="one condition")  # Subset.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    parser.add_argument("--resume", action="store_true", help="continue from the backend")  # Resume.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Log-likelihood of the two-dimensional observations at parameters in log space.
# Arguments:
#   theta_log (numpy.ndarray): logarithms of (d, gamma, mu, R), in that order.
#   observations (numpy.ndarray): the noisy observations of the condition.
#   sigma (float): the standard deviation of the observation noise.
#   context (dict): the fixed quantities of the two-dimensional forward map.
# Returns:
#   float: the log-likelihood, or negative infinity for a rejected proposal.
def loglik_2d(theta_log, observations, sigma, context):
    values = np.exp(np.asarray(theta_log, dtype=float))  # Parameters in natural units.
    diffusion, advection, memory_decay, radius = values  # Unpack the four free parameters.
    params = {  # Model parameters of equation (1) implied by the proposal.
        "diffusion": float(diffusion),  # Diffusion rate d of the density equation.
        "alpha": float(advection / context["beta"]),  # Advection strength implied by gamma.
        "beta": float(context["beta"]),  # Memory uptake rate beta of the map equation.
        "memory_decay": float(memory_decay),  # Memory decay rate mu of the map equation.
        "radius": float(radius),  # Perceptual range R of the detection kernel.
    }
    try:  # A proposal may drive the pseudo-spectral solution negative.
        result = simulate_2d(  # Two-dimensional forward solve at the inference resolution.
            params,  # Model parameters implied by the proposal.
            context["kernel"],  # Detection kernel family of the condition.
            context["initial_density"],  # Initial density of the two-dimensional experiments.
            context["initial_map"],  # Initial cognitive map of the two-dimensional experiments.
            context["t_final"],  # Upper end T of the observation window.
            context["time_step"],  # Time step of the inference resolution.
            context["n_points"],  # Grid size of the inference resolution.
            context["domain_length"],  # Side length L of the periodic square.
            context["record_times"],  # Times at which the density is recorded.
            context["scheme"],  # Time stepping scheme of Section 4.1.
            context["positivity_tolerance"],  # Relative tolerance of the positivity monitor.
        )  # Forward solution of model (1) at the proposed parameters.
    except Exception:  # The proposal was rejected by the positivity monitor.
        return -np.inf  # Assign zero likelihood to an inadmissible proposal.
    stride = context["n_points"] // context["n_space"]  # Stride of the equally spaced samples.
    prediction = result["density"][:, ::stride, ::stride]  # Prediction at the sampling points.
    return gaussian_loglik(observations, prediction, sigma)  # Log-likelihood of equation (9).


# Entry point of the two-dimensional inference study.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["mcmc_2d"]  # Settings of the two-dimensional ensemble sampler.
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    source = Path("results/raw") / str(config["experiment"])  # Directory of the generated data.
    directory = ensure_dir(arguments.output_dir or source)  # Directory of the sampler backends.
    conditions = conditions_2d(config)  # Two-dimensional conditions of Section 5.4.
    if arguments.condition_index is not None:  # A single condition was requested.
        conditions = [conditions[int(arguments.condition_index) % len(conditions)]]  # That one.
    n_points = int(settings["n_points"])  # Grid size of the inference resolution.
    records = []  # Accumulator for the summary rows of the completed conditions.
    for condition in conditions:  # Sample the posterior of each selected condition in turn.
        archive = source / f"data2d_{condition['index']:03d}.h5"  # Archive of the generated data.
        if not archive.is_file():  # The data of this condition have not been generated yet.
            print(f"missing data for condition {condition['index']:03d}")  # Report the gap.
            continue  # Continue with the next condition of the study.
        arrays, metadata = read_hdf5(archive)  # Generated data of this condition.
        context = {  # Fixed quantities of the two-dimensional forward map.
            "kernel": condition["kernel"],  # Detection kernel family of the condition.
            "beta": float(condition["beta"]),  # Memory uptake rate beta of the condition.
            "domain_length": float(design["domain_length"]),  # Side length L of the square.
            "n_points": n_points,  # Grid size of the inference resolution.
            "n_space": int(condition["n_space"]),  # Spatial sample points along each axis.
            "time_step": float(settings["time_step"]),  # Time step of the inference resolution.
            "t_final": float(design["observation_time"]),  # Upper end T of the window.
            "record_times": np.asarray(arrays["times"]),  # Observation times of the design.
            "initial_density": initial_density_2d(  # Initial density of the forward map.
                n_points,  # Grid size of the inference resolution.
                float(design["domain_length"]),  # Side length L of the periodic square.
                float(design["mean_density"]),  # Mean density u_bar of the uniform state.
                float(design["perturbation_amplitude"]),  # Amplitude of the perturbation.
            ),  # Initial density used at every likelihood evaluation.
            "initial_map": float(design["initial_map"]),  # Initial cognitive map.
            "scheme": solver["scheme"],  # Time stepping scheme of Section 4.1.
            "positivity_tolerance": float(solver["positivity_tolerance"]),  # Positivity monitor.
        }
        prior = transient_prior(condition["advection"], design["priors"])  # Prior of Table 3.
        sigma = float(metadata["sigma"])  # Standard deviation of the observation noise.
        observations = np.asarray(arrays["observations"], dtype=float)  # Noisy observations.

        # Unnormalised log posterior of the two-dimensional condition, in log space.
        def log_posterior(point, observations=observations, sigma=sigma, context=context, prior=prior):
            log_prior = prior.logpdf(point)  # Log density of the log-uniform prior.
            if not np.isfinite(log_prior):  # The proposal lies outside the prior support.
                return -np.inf  # Reject the proposal without a forward solve.
            return log_prior + loglik_2d(point, observations, sigma, context)  # Log posterior.

        generator = np.random.default_rng(int(condition["seed"]) % (2**32))  # Deterministic seed.
        start = initial_positions(prior, int(settings["n_walkers"]), generator)  # Walker starts.
        result = run_ensemble(  # Run the ensemble sampler with the HDF5 backend.
            log_posterior,  # Unnormalised log posterior of the condition.
            start,  # Starting positions of the walkers.
            int(settings["n_steps"]),  # Number of ensemble steps requested in total.
            directory / f"ensemble_{condition['index']:03d}.h5",  # Backend of this condition.
            arguments.resume,  # Whether an existing backend is continued.
            settings["wall_time_limit"],  # Wall time after which the run stops cleanly.
            int(condition["seed"]),  # Seed recorded in the output metadata.
        )  # Chain and diagnostics of this run.
        # Report the number of ensemble steps completed for this condition.
        print(f"condition {condition['index']:03d}: {result['iterations']} ensemble steps")
        if not result["complete"]:  # The wall-time guard stopped the run before completion.
            continue  # The summary is written once the run has finished in a later block.
        chain = np.asarray(result["chain"])  # Chain of shape (steps, walkers, parameters).
        burn_in = int(float(settings["burn_in_fraction"]) * chain.shape[0])  # Discarded steps.
        kept = np.exp(np.transpose(chain[burn_in:], (1, 0, 2)))  # Draws in natural units.
        statistics = convergence_statistics(kept, PARAMETER_NAMES)  # Convergence diagnostics.
        truth = {  # True parameters of the condition, in natural units.
            "diffusion": float(condition["diffusion"]),  # True diffusion rate d.
            "advection": float(condition["advection"]),  # True combined advection strength gamma.
            "memory_decay": float(condition["memory_decay"]),  # True memory decay rate mu.
            "radius": float(condition["radius"]),  # True perceptual range R.
        }
        summary_values = posterior_summary(kept, PARAMETER_NAMES, truth)  # Posterior summaries.
        record = {  # Summary row of this two-dimensional condition.
            "condition_index": condition["index"],  # Stable integer index of the condition.
            "identifier": condition["identifier"],  # Stable textual label of the condition.
            "kernel": condition["kernel"],  # Detection kernel family of the condition.
            "radius": condition["radius"],  # Perceptual range R of the condition.
            "noise_level": condition["noise_level"],  # Relative noise level eta.
            "acceptance_rate": result["acceptance_rate"],  # Mean acceptance fraction.
            "converged": has_converged(statistics),  # Whether the criteria of Section 4.4 hold.
        }
        for name in PARAMETER_NAMES:  # Record the diagnostics and summaries of every parameter.
            record[f"rhat_{name}"] = statistics[name]["rhat"]  # Rank-normalised split R-hat.
            record[f"ess_{name}"] = statistics[name]["ess_bulk"]  # Bulk effective sample size.
            record[f"median_{name}"] = summary_values[name]["median"]  # Posterior median.
            record[f"width_{name}"] = summary_values[name]["relative_width"]  # Relative width.
            record[f"truth_{name}"] = summary_values[name]["truth"]  # True value.
        records.append(record)  # Row appended to the two-dimensional summary table.
    summary = Path("results/summary") / f"{config['experiment']}_mcmc_2d.csv"  # Summary path.
    if records:  # Nothing is written when every requested run was partial.
        write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
        print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
