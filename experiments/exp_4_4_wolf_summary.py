# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Aggregate the wolf fits of Experiment 4.3 across individuals, report
# the pooled perceptual range and aggregation ratio and state how often each
# candidate model is preferred, which supplies the values quoted in Section 6.
# Manuscript: Section 6, Experiment 4.4.
# Data: Latham ADM, Boutin S (2019), Movebank Data Repository,
# https://doi.org/10.5441/001/1.7vr1k987, licensed CC0 1.0.
# Inputs: the summaries of Experiment 4.3.
# Outputs: a tracked summary under results/summary/.

import argparse  # Command line interface of the experiment script.
import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and summary statistics.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.io import write_csv  # Writer of the tracked summary table.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Wolf summary")  # Argument parser of the script.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Read a tracked summary table into a list of dictionaries.
# Arguments:
#   path (str or pathlib.Path): the comma separated file to read.
# Returns:
#   list: one dictionary per row, or an empty list when the file is absent.
def read_summary(path):
    location = Path(path)  # Normalise the argument into a path object.
    if not location.is_file():  # The summary has not been produced yet.
        return []  # Report the absence as an empty table.
    with location.open(encoding="utf-8", newline="") as handle:  # Open the summary for reading.
        return list(csv.DictReader(handle))  # One dictionary per row of the table.


# Entry point of the wolf summary.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    experiment = str(config["experiment"])  # Identifier of the experiment, used in the paths.
    folder = Path("results/summary")  # Directory that holds the tracked summary tables.
    posteriors = read_summary(folder / f"{experiment}_wolf_posteriors.csv")  # Posterior summaries.
    comparison = read_summary(folder / f"{experiment}_wolf_comparison.csv")  # Model comparison.
    if not posteriors:  # The fits have not produced a posterior summary yet.
        print("no wolf posteriors were found, run exp_4_3_wolf_fit.py first")  # Report the gap.
        return 0  # Signal success, since the absence of the input is not an error here.
    records = []  # Accumulator for the rows of the pooled summary table.
    for model in sorted({row["model"] for row in posteriors}):  # Summarise each fitted model.
        rows = [row for row in posteriors if row["model"] == model]  # Fits of the current model.
        radii = np.array([float(row["median_radius"]) for row in rows])  # Median ranges, in metres.
        ratios = np.array([float(row["median_aggregation_ratio"]) for row in rows])  # Ratios.
        widths = np.array(  # Relative width of the credible interval of the perceptual range.
            [  # One entry per fitted individual of the current model.
                # Relative width of the credible interval of this individual.
                (float(row["upper_radius"]) - float(row["lower_radius"])) / float(row["median_radius"])
                for row in rows  # Fits of the current model.
            ]
        )  # Relative widths used to describe how well the range is constrained.
        # Individuals for which the current model ranks first in the comparison.
        preferred = [row for row in comparison if row["model"] == model and row["loo_rank"] == "1"]
        records.append(  # One row of the pooled summary table of Section 6.
            {  # Pooled summary of the current candidate model across individuals.
                "model": model,  # Name of the candidate model.
                "n_individuals": len(rows),  # Number of individuals fitted with this model.
                "median_radius_m": float(np.median(radii)),  # Median perceptual range, in metres.
                "min_radius_m": float(np.min(radii)),  # Smallest median perceptual range.
                "max_radius_m": float(np.max(radii)),  # Largest median perceptual range.
                "median_aggregation_ratio": float(np.median(ratios)),  # Median aggregation ratio.
                "median_relative_width_radius": float(np.median(widths)),  # Median relative width.
                # Number of fits of this model that met the convergence criteria.
                "n_converged": sum(1 for row in rows if row["converged"].lower() == "true"),
                "n_preferred": len(preferred),  # Individuals for which this model ranks first.
                # Citation of the data package, carried through every output.
                "data_citation": "Latham and Boutin (2019) https://doi.org/10.5441/001/1.7vr1k987",
            }
        )  # Row appended to the pooled summary table.
        print(f"model {model}: median perceptual range {float(np.median(radii)):.0f} m")  # Report.
    agree = [row for row in comparison if str(row["rankings_agree"]).lower() == "true"]  # Checks.
    if comparison:  # The robustness check is reported only when a comparison exists.
        share = len(agree) / len(comparison)  # Share of rows whose two rankings agree.
        # Report how often the two cross-validation rankings agree.
        print(f"leave one block out and leave one out rankings agree in {share:.2f} of the rows")
    summary = folder / f"{experiment}_wolf_summary.csv"  # Path of the pooled summary table.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
