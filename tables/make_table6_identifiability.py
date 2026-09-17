# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Table 6 of the manuscript, classifying each parameter as practically
# identifiable or not under transient and under stationary data, for the
# intermediate sampling design and the perceptual range 0.15.
# Manuscript: Section 5.3, Table 6; Experiment 2.3.
# Inputs: the tracked summaries under results/summary.
# Outputs: results/tables/Table6.csv and results/tables/Table6.tex.

import argparse  # Command line interface of the table script.
import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the table script.
from pathlib import Path  # Portable filesystem paths.

from nmi.io import ensure_dir, write_csv, write_latex  # Output writers of the table script.

# Caption of the table, reproduced in the LaTeX output.
CAPTION = "Practical identifiability by profile likelihood"  # Caption of Table 6.
# LaTeX label of the table, used for cross references in the manuscript.
LABEL = "tab:identifiability"  # Label of Table 6.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Table 6")  # Argument parser of the script.
    parser.add_argument("--summary-dir", default="results/summary", help="summaries")  # Input.
    parser.add_argument("--output-dir", default="results/tables", help="output directory")  # Output.
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


# Locate the first non-empty summary that matches a pattern.
# Arguments:
#   folder (pathlib.Path): the directory that holds the tracked summaries.
#   pattern (str): the glob pattern of the summary file name.
# Returns:
#   list: the rows of the first summary that exists, possibly empty.
def find_summary(folder, pattern):
    for path in sorted(Path(folder).glob(pattern)):  # Inspect every matching summary in turn.
        rows = read_summary(path)  # Rows of the current candidate summary.
        if rows:  # The candidate summary carries at least one row.
            return rows  # Use the first non-empty summary that was found.
    return []  # No matching summary exists yet.


# Entry point of the table script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    rows = find_summary(arguments.summary_dir, "*_profile_likelihood.csv")  # Profile study.
    if not rows:  # The profile likelihood study has not produced a summary yet.
        print("no profile summary was found, run exp_2_3_profile_likelihood.py first")  # Report.
        return 0  # Signal success, since the absence of the input is not an error here.
    noise_levels = sorted({f"{float(row['noise_level']):.2f}" for row in rows})  # Noise levels.
    grouped = {}  # Mapping from the row key to the classification at each noise level.
    for row in rows:  # Group the classifications by data type, kernel and parameter.
        key = (row["data_type"], row["kernel"], row["parameter"])  # Key of this row of Table 6.
        # Store the classification of this row under its noise level.
        grouped.setdefault(key, {})[f"{float(row['noise_level']):.2f}"] = row["classification"]
    records = []  # Accumulator for the rows of Table 6.
    for key in sorted(grouped):  # Format one row of Table 6 per key, in a stable order.
        data_type, kernel, parameter = key  # Components of the key of this row.
        record = {  # Formatted row of Table 6.
            "data_type": data_type.capitalize(),  # Transient or stationary data.
            "kernel": "Top-hat" if kernel == "tophat" else "Gaussian",  # Kernel family.
            "parameter": parameter.replace("_", " "),  # Name of the profiled parameter.
        }
        for level in noise_levels:  # One column per noise level of the study.
            record[f"eta_{level}"] = grouped[key].get(level, "")  # Classification at that level.
        records.append(record)  # Row appended to Table 6.
    # Column names of the output, with one column per noise level of the study.
    columns = ["data_type", "kernel", "parameter"] + [f"eta_{level}" for level in noise_levels]
    # Column headers of the typeset table, written in mathematical notation.
    headers = ["Data type", "Kernel", "Parameter"] + [f"$\\eta={level}$" for level in noise_levels]
    directory = ensure_dir(arguments.output_dir)  # Directory that receives the table files.
    csv_path = write_csv(directory / "Table6.csv", records, columns)  # Comma separated output.
    # LaTeX output in the booktabs style expected by the journal template.
    tex_path = write_latex(directory / "Table6.tex", records, columns, headers, CAPTION, LABEL)
    print(f"wrote {csv_path} and {tex_path} with {len(records)} row(s)")  # Report the outputs.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the table and propagate the exit status.
