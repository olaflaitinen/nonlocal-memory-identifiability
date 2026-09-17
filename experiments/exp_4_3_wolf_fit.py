# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Fit the three candidate models of Section 6 to the space use of each
# included wolf, sample the posterior of the aggregation ratio and the
# perceptual range with the ensemble sampler, and compare the models by Pareto
# smoothed importance sampling leave one out cross-validation together with the
# leave one block out robustness check.
# Manuscript: Section 6, Experiment 4.3; Table 8 and Fig. 7.
# Data: Latham ADM, Boutin S (2019), Movebank Data Repository,
# https://doi.org/10.5441/001/1.7vr1k987, licensed CC0 1.0.
# Inputs: an experiment configuration and the output of Experiment 4.1.
# Outputs: one HDF5 backend per fit and tracked summaries.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.
import pandas  # Tabular handling of the cleaned table of location fixes.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.diagnostics import convergence_statistics, has_converged, posterior_summary  # Diagnostics.
from nmi.io import ensure_dir, write_csv  # Output writers of the experiment.
from nmi.mcmc_ensemble import initial_positions, run_ensemble  # Ensemble sampler wrapper.
from nmi.priors import LogUniformPrior  # Log-uniform prior on the fitted parameters.
from nmi.wolf.masked_solver import masked_steady_state  # Stationary density on the masked grid.
from nmi.wolf.model_comparison import (  # Model comparison of Section 4.5.
    block_indices,  # Contiguous blocks of fixes used by the robustness check.
    compare_models,  # Leave one out comparison of the candidate models.
    leave_one_block_out,  # Leave one block out robustness check.
    ranking,  # Ranking implied by a comparison table.
)
from nmi.wolf.point_likelihood import (  # Weighted point likelihood of equation (10).
    pointwise_log_density,  # Log density of the fitted model at every fix.
    uniform_loglik,  # Weighted log-likelihood of the model without nonlocal advection.
    weighted_pointwise,  # Weighted contribution of every fix.
)

from experiments.exp_4_2_wolf_pilot import load_study_area, read_wolf_summary  # Shared helpers.

# Names of the sampled parameters of the wolf fits, in sampler order.
PARAMETER_NAMES = ("aggregation_ratio", "radius")  # Parameters identified by Proposition 4.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Wolf fits")  # Argument parser of the script.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--condition-index", type=int, default=None, help="one individual")  # Subset.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    parser.add_argument("--resume", action="store_true", help="continue from the backend")  # Resume.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Assemble the data of one individual, as the likelihood expects them.
# Arguments:
#   identifier (str): the animal identifier of the individual.
#   fixes (pandas.DataFrame): the cleaned table of location fixes.
#   area (dict): the study area of the individual.
#   n_eff (float): the effective sample size for area of the individual.
#   kernel (str): the detection kernel family of the fitted model.
# Returns:
#   dict: the mapping consumed by the weighted point likelihood.
def build_individual(identifier, fixes, area, n_eff, kernel):
    group = fixes[fixes["individual_id"].astype(str) == str(identifier)]  # Fixes of this animal.
    # Data of the individual in the layout used by the weighted point likelihood.
    return {
        "identifier": str(identifier),  # Identifier of the animal.
        "easting": group["easting_m"].to_numpy(dtype=float),  # Projected first coordinate.
        "northing": group["northing_m"].to_numpy(dtype=float),  # Projected second coordinate.
        "timestamps": pandas.to_datetime(group["timestamp_utc"], utc=True),  # Times of the fixes.
        "n_eff": float(n_eff),  # Effective sample size for area of the individual.
        "kernel": kernel,  # Detection kernel family of the fitted model.
        "area": area,  # Study area of the individual, as a masked Cartesian grid.
    }


# Weighted log-likelihood of one individual at parameters given in log space.
# Arguments:
#   theta_log (numpy.ndarray): logarithms of (kappa, R), in that order.
#   individual (dict): the mapping produced by build_individual.
# Returns:
#   float: the weighted log-likelihood of equation (10).
def individual_log_likelihood(theta_log, individual):
    values = np.exp(np.asarray(theta_log, dtype=float))  # Parameters in natural units.
    result = masked_steady_state(  # Stationary density implied by the proposal.
        float(values[0]),  # Aggregation ratio kappa of the proposal.
        float(values[1]),  # Perceptual range R of the proposal, in metres.
        individual["kernel"],  # Detection kernel family of the fitted model.
        individual["area"]["mask"],  # Boolean mask of the study area.
        individual["area"]["cell"],  # Cell width of the Cartesian grid, in metres.
    )  # Converged stationary density of Proposition 4.
    if not result["converged"]:  # The fixed-point solver failed to reach the tolerance.
        return -np.inf  # Assign zero likelihood to an inadmissible proposal.
    log_density = pointwise_log_density(  # Log density of the fitted model at every fix.
        individual["easting"],  # Projected first coordinate of the retained fixes.
        individual["northing"],  # Projected second coordinate of the retained fixes.
        result["density"],  # Stationary density on the masked grid.
        individual["area"],  # Study area mapping that defines the grid.
    )  # Pointwise log densities of the retained fixes.
    return float(np.sum(weighted_pointwise(log_density, individual["n_eff"])))  # Equation (10).


# Pointwise contributions of a set of posterior draws for one individual.
# Arguments:
#   draws (numpy.ndarray): retained draws in log space.
#   individual (dict): the mapping produced by build_individual.
#   n_draws (int): number of draws retained for the cross-validation.
# Returns:
#   numpy.ndarray: contributions of shape (draws, fixes).
def draw_contributions(draws, individual, n_draws):
    stride = max(1, draws.shape[0] // int(n_draws))  # Stride that thins the posterior draws.
    thinned = draws[::stride][: int(n_draws)]  # Draws retained for the cross-validation.
    count = individual["easting"].size  # Number of retained fixes of the individual.
    matrix = np.full((thinned.shape[0], count), -np.inf)  # Storage of the contributions.
    for position, point in enumerate(thinned):  # Evaluate the density at each retained draw.
        values = np.exp(np.asarray(point, dtype=float))  # Parameters in natural units.
        result = masked_steady_state(  # Stationary density implied by this draw.
            float(values[0]),  # Aggregation ratio kappa of the draw.
            float(values[1]),  # Perceptual range R of the draw, in metres.
            individual["kernel"],  # Detection kernel family of the fitted model.
            individual["area"]["mask"],  # Boolean mask of the study area.
            individual["area"]["cell"],  # Cell width of the Cartesian grid, in metres.
        )  # Converged stationary density of this draw.
        if not result["converged"]:  # The fixed-point solver failed at this draw.
            continue  # Leave the row at minus infinity, which excludes the draw.
        log_density = pointwise_log_density(  # Log density of the fitted model at every fix.
            individual["easting"],  # Projected first coordinate of the retained fixes.
            individual["northing"],  # Projected second coordinate of the retained fixes.
            result["density"],  # Stationary density on the masked grid.
            individual["area"],  # Study area mapping that defines the grid.
        )  # Pointwise log densities of the retained fixes.
        matrix[position] = weighted_pointwise(log_density, individual["n_eff"])  # Contributions.
    finite = np.all(np.isfinite(matrix), axis=1)  # Draws at which every contribution is finite.
    return matrix[finite]  # Contributions of the admissible draws only.


# Entry point of the wolf fits.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["wolf"]  # Settings of the application to the wolf data.
    source = Path("results/raw/exp_4_1")  # Directory that holds the output of Experiment 4.1.
    directory = ensure_dir(arguments.output_dir or Path("results/raw") / str(config["experiment"]))
    rows = read_wolf_summary(str(config["experiment"]))  # Summary table of Experiment 4.1.
    fixes_path = source / "cleaned_fixes.csv"  # Internal cleaned table of location fixes.
    if not rows or not fixes_path.is_file():  # The preprocessing has not been run yet.
        print("no preprocessed wolf data were found, run exp_4_1_wolf_preprocess.py first")
        return 0  # Signal success, since the absence of the input is not an error here.
    fixes = pandas.read_csv(fixes_path)  # Cleaned and projected table of location fixes.
    included = [row for row in rows if str(row["included"]).lower() == "true"]  # Included animals.
    if arguments.condition_index is not None:  # A single individual was requested.
        included = [included[int(arguments.condition_index) % max(1, len(included))]]  # That one.
    prior = LogUniformPrior(  # Log-uniform prior on the two fitted parameters.
        PARAMETER_NAMES,  # Names of the fitted parameters, in sampler order.
        [float(settings["kappa_bounds"][0]), float(settings["radius_bounds"][0])],  # Lower bounds.
        [float(settings["kappa_bounds"][1]), float(settings["radius_bounds"][1])],  # Upper bounds.
    )  # Prior shared by both detection kernel families.
    posterior_records = []  # Accumulator for the posterior summary rows.
    comparison_records = []  # Accumulator for the model comparison rows.
    for position, row in enumerate(included):  # Fit every included individual in turn.
        identifier = str(row["individual_id"])  # Identifier of the animal.
        area = load_study_area(identifier, source)  # Study area of this individual.
        if area is None:  # The study area of this individual has not been built.
            print(f"missing study area for individual {identifier}")  # Report the gap.
            continue  # Continue with the next included individual.
        contributions = {}  # Pointwise contributions of each candidate model.
        for kernel in settings["kernels"]:  # Fit both detection kernel families.
            individual = build_individual(identifier, fixes, area, float(row["n_eff"]), kernel)

            # Unnormalised log posterior of this individual and kernel family.
            def log_posterior(point, individual=individual, prior=prior):
                log_prior = prior.logpdf(point)  # Log density of the log-uniform prior.
                if not np.isfinite(log_prior):  # The proposal lies outside the prior support.
                    return -np.inf  # Reject the proposal without a forward solve.
                return log_prior + individual_log_likelihood(point, individual)  # Log posterior.

            generator = np.random.default_rng(int(settings["seed"]) + position)  # Deterministic.
            start = initial_positions(prior, int(settings["n_walkers"]), generator)  # Walkers.
            result = run_ensemble(  # Run the ensemble sampler with the HDF5 backend.
                log_posterior,  # Unnormalised log posterior of this fit.
                start,  # Starting positions of the walkers.
                int(settings["n_steps"]),  # Number of ensemble steps requested in total.
                directory / f"wolf_{identifier}_{kernel}.h5",  # Backend of this fit.
                arguments.resume,  # Whether an existing backend is continued.
                settings["wall_time_limit"],  # Wall time after which the run stops cleanly.
                int(settings["seed"]) + position,  # Seed recorded in the output metadata.
            )  # Chain and diagnostics of this fit.
            print(f"individual {identifier} kernel {kernel}: {result['iterations']} steps")
            if not result["complete"]:  # The wall-time guard stopped the run before completion.
                continue  # The summary is written once the run has finished in a later block.
            chain = np.asarray(result["chain"])  # Chain of shape (steps, walkers, parameters).
            burn_in = int(float(settings["burn_in_fraction"]) * chain.shape[0])  # Discarded steps.
            kept = np.exp(np.transpose(chain[burn_in:], (1, 0, 2)))  # Draws in natural units.
            statistics = convergence_statistics(kept, PARAMETER_NAMES)  # Convergence diagnostics.
            flat = np.log(kept.reshape(-1, kept.shape[-1]))  # Pooled draws, in log space.
            contributions[kernel] = draw_contributions(flat, individual, int(settings["n_draws"]))
            truth = {name: 1.0 for name in PARAMETER_NAMES}  # No true value exists for real data.
            summary_values = posterior_summary(kept, PARAMETER_NAMES, truth)  # Posterior summaries.
            record = {  # Posterior summary row of this fit.
                "individual_id": identifier,  # Identifier of the animal.
                "model": kernel,  # Detection kernel family of the fitted model.
                "acceptance_rate": result["acceptance_rate"],  # Mean acceptance fraction.
                "converged": has_converged(statistics),  # Whether the criteria of Section 4.4 hold.
                "max_rhat": max(statistics[name]["rhat"] for name in PARAMETER_NAMES),  # Worst.
                "min_ess": min(statistics[name]["ess_bulk"] for name in PARAMETER_NAMES),  # Worst.
                "data_citation": "Latham and Boutin (2019) https://doi.org/10.5441/001/1.7vr1k987",
            }
            for name in PARAMETER_NAMES:  # Record the posterior summary of every parameter.
                record[f"median_{name}"] = summary_values[name]["median"]  # Posterior median.
                record[f"lower_{name}"] = summary_values[name]["lower"]  # Lower credible bound.
                record[f"upper_{name}"] = summary_values[name]["upper"]  # Upper credible bound.
            posterior_records.append(record)  # Row appended to the posterior summary table.
        if len(contributions) < len(settings["kernels"]):  # Not every fit of this animal finished.
            continue  # The comparison is written once every fit of the animal is complete.
        uniform_individual = build_individual(identifier, fixes, area, float(row["n_eff"]), "tophat")
        _, uniform_pointwise = uniform_loglik(uniform_individual)  # Contributions of the uniform model.
        reference_draws = next(iter(contributions.values())).shape[0]  # Number of retained draws.
        contributions["no_advection"] = np.tile(uniform_pointwise, (reference_draws, 1))  # Fixed.
        comparison = compare_models(contributions)  # Leave one out comparison of the models.
        times = uniform_individual["timestamps"]  # Times of the retained fixes of this animal.
        days = (times - times.min()).dt.total_seconds().to_numpy() / 86400.0  # Days since the first.
        blocks = block_indices(days, float(settings["block_days"]))  # Contiguous blocks of fixes.
        block_comparison = leave_one_block_out(contributions, blocks)  # Robustness check.
        block_rank = ranking(block_comparison, "elpd_block")  # Ranking of the robustness check.
        loo_rank = ranking(comparison, "elpd")  # Ranking of the leave one out comparison.
        for entry in comparison:  # Record one comparison row per candidate model.
            comparison_records.append(  # One row of the model comparison table of Table 8.
                {  # Outcome of the comparison for this candidate model.
                    "individual_id": identifier,  # Identifier of the animal.
                    "model": entry["model"],  # Name of the candidate model.
                    "elpd": entry["elpd"],  # Expected log pointwise predictive density.
                    "elpd_difference": entry["elpd_difference"],  # Difference from the best model.
                    "difference_se": entry["difference_se"],  # Standard error of the difference.
                    "max_pareto_k": entry["max_pareto_k"],  # Largest Pareto shape value.
                    "loo_rank": loo_rank.index(entry["model"]) + 1,  # Rank under the comparison.
                    "block_rank": block_rank.index(entry["model"]) + 1,  # Rank under the check.
                    "rankings_agree": loo_rank == block_rank,  # Whether the two rankings agree.
                    "n_blocks": int(np.unique(blocks).size),  # Number of contiguous blocks.
                    "data_citation": "Latham and Boutin (2019) https://doi.org/10.5441/001/1.7vr1k987",
                }
            )  # Row appended to the comparison table.
    folder = Path("results/summary")  # Directory that holds the tracked summary tables.
    if posterior_records:  # Nothing is written when every fit was partial.
        path = folder / f"{config['experiment']}_wolf_posteriors.csv"  # Posterior summary path.
        write_csv(path, posterior_records, list(posterior_records[0].keys()))  # Write the table.
        print(f"wrote {path} with {len(posterior_records)} row(s)")  # Report the location.
    if comparison_records:  # Nothing is written when no animal completed every fit.
        path = folder / f"{config['experiment']}_wolf_comparison.csv"  # Comparison table path.
        write_csv(path, comparison_records, list(comparison_records[0].keys()))  # Write the table.
        print(f"wrote {path} with {len(comparison_records)} row(s)")  # Report the location.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
