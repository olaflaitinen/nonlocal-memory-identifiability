# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Table 5 of the manuscript, listing the relative error, the observed
# order of convergence, the mass conservation error and the cost of one forward
# solve at each grid size of the one-dimensional convergence study.
# Manuscript: Section 5.1, Table 5; Experiment 1.1.
# Inputs: the tracked summaries under results/summary.
# Outputs: results/tables/Table5.csv and results/tables/Table5.tex.

import argparse  # Command line interface of the table script.
import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the table script.
from pathlib import Path  # Portable filesystem paths.

from nmi.io import ensure_dir, write_csv, write_latex  # Output writers of the table script.

# Caption of the table, reproduced in the LaTeX output.
CAPTION = "Convergence of the one-dimensional forward solver"  # Caption of Table 5.
# LaTeX label of the table, used for cross references in the manuscript.
LABEL = "tab:convergence"  # Label of Table 5.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Table 5")  # Argument parser of the script.
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
    rows = find_summary(arguments.summary_dir, "*_convergence_1d.csv")  # Convergence study.
    if not rows:  # The convergence study has not produced a summary yet.
        print("no convergence summary was found, run exp_1_1_convergence_1d.py first")  # Report.
        return 0  # Signal success, since the absence of the input is not an error here.
    records = []  # Accumulator for the rows of Table 5.
    for row in rows:  # Format one row of Table 5 per row of the summary.
        if row["refinement"] != "grid":  # Table 5 reports the grid refinement study only.
            continue  # Continue with the next row of the summary.
        order = row["observed_order"]  # Observed order of convergence, empty on the first grid.
        records.append(  # One row of Table 5.
            {  # Formatted values of the current grid size.
                "kernel": "Top-hat" if row["kernel"] == "tophat" else "Gaussian",  # Kernel.
                "n_points": row["n_points"],  # Grid size of the run.
                "relative_error": f"{float(row['relative_error']):.3e}",  # Relative error.
                "observed_order": f"{float(order):.2f}" if order else "",  # Observed order.
                "mass_error": f"{float(row['mass_error']):.3e}",  # Mass conservation error.
                "wall_time": f"{float(row['wall_time']):.3f}",  # Cost of one forward solve.
            }
        )  # Row appended to Table 5.
    # Column names of the output, in the order used by the manuscript.
    columns = ["kernel", "n_points", "relative_error", "observed_order", "mass_error", "wall_time"]
    # Column headers of the typeset table, written in mathematical notation.
    headers = ["Kernel", "$N$", "Relative $L^2$ error", "Observed order", "Mass error", "Time (s)"]
    directory = ensure_dir(arguments.output_dir)  # Directory that receives the table files.
    csv_path = write_csv(directory / "Table5.csv", records, columns)  # Comma separated output.
    # LaTeX output in the booktabs style expected by the journal template.
    tex_path = write_latex(directory / "Table5.tex", records, columns, headers, CAPTION, LABEL)
    print(f"wrote {csv_path} and {tex_path} with {len(records)} row(s)")  # Report the outputs.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the table and propagate the exit status.
