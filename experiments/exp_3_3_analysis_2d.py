# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Compare the relative credible interval widths obtained in two space
# dimensions with the widths obtained in one dimension for the corresponding
# conditions, which is the content of Fig. 6.
# Manuscript: Section 5.4, Experiment 3.3; Fig. 6 and Remark 2.
# Inputs: the summaries of Experiments 2.4 and 3.2.
# Outputs: a tracked summary under results/summary/.

import argparse  # Command line interface of the experiment script.
import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.io import write_csv  # Writer of the tracked summary table.

# Names of the sampled parameters, in the order used by every sampler.
PARAMETER_NAMES = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameter vector.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Two-dimensional analysis")  # Argument parser.
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


# Entry point of the two-dimensional analysis.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    experiment = str(config["experiment"])  # Identifier of the experiment, used in the paths.
    folder = Path("results/summary")  # Directory that holds the tracked summary tables.
    rows_1d = read_summary(folder / f"{experiment}_mcmc_1d.csv")  # Widths in one dimension.
    rows_2d = read_summary(folder / f"{experiment}_mcmc_2d.csv")  # Widths in two dimensions.
    if not rows_2d:  # The two-dimensional runs have not produced a summary yet.
        print("no two-dimensional summary was found, run exp_3_2_mcmc_2d.py first")  # Report.
        return 0  # Signal success, since the absence of the input is not an error here.
    index_1d = {}  # Mapping from the factor combination to the one-dimensional widths.
    for row in rows_1d:  # Index the one-dimensional rows by their factor combination.
        # Factor combination that identifies a row of either summary table.
        key = (row["kernel"], f"{float(row['radius']):.3f}", f"{float(row['noise_level']):.2f}")
        index_1d[key] = row  # Store the one-dimensional row under its factor combination.
    records = []  # Accumulator for the rows of the comparison table.
    for row in rows_2d:  # Pair each two-dimensional row with its one-dimensional counterpart.
        # Factor combination that identifies a row of either summary table.
        key = (row["kernel"], f"{float(row['radius']):.3f}", f"{float(row['noise_level']):.2f}")
        partner = index_1d.get(key)  # One-dimensional row with the same factor combination.
        for name in PARAMETER_NAMES:  # Compare the relative width of every parameter.
            width_2d = float(row[f"width_{name}"])  # Relative width in two dimensions.
            width_1d = float(partner[f"width_{name}"]) if partner else ""  # One-dimensional width.
            ratio = (width_2d / width_1d) if partner and width_1d > 0.0 else ""  # Ratio of widths.
            records.append(  # One row of the comparison table of Fig. 6.
                {  # Comparison of the two widths for this parameter and condition.
                    "identifier": row["identifier"],  # Label of the two-dimensional condition.
                    "kernel": row["kernel"],  # Detection kernel family of the condition.
                    "radius": float(row["radius"]),  # Perceptual range R of the condition.
                    "noise_level": float(row["noise_level"]),  # Relative noise level eta.
                    "parameter": name,  # Name of the compared parameter.
                    "width_1d": width_1d,  # Relative credible interval width in one dimension.
                    "width_2d": width_2d,  # Relative credible interval width in two dimensions.
                    "ratio_2d_over_1d": ratio,  # Ratio of the two relative widths.
                    "converged_2d": row["converged"],  # Convergence flag of the two-dimensional run.
                }
            )  # Row appended to the comparison table.
    summary = folder / f"{experiment}_analysis_2d.csv"  # Path of the comparison table.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
