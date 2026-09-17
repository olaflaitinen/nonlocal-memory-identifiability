# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Table 8 of the manuscript, listing the posterior summaries of the
# aggregation ratio and the perceptual range together with the model comparison
# for each included wolf.
# Manuscript: Section 6, Table 8; Experiment 4.3.
# Inputs: the tracked summaries under results/summary.
# Outputs: results/tables/Table8.csv and results/tables/Table8.tex.

import argparse  # Command line interface of the table script.
import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the table script.
from pathlib import Path  # Portable filesystem paths.

from nmi.io import ensure_dir, write_csv, write_latex  # Output writers of the table script.

# Caption of the table, reproduced in the LaTeX output.
CAPTION = "Posterior summaries and model comparison for the wolf data"  # Caption of Table 8.
# LaTeX label of the table, used for cross references in the manuscript.
LABEL = "tab:wolfposteriors"  # Label of Table 8.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Table 8")  # Argument parser of the script.
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
    posteriors = find_summary(arguments.summary_dir, "*_wolf_posteriors.csv")  # Posteriors.
    comparison = find_summary(arguments.summary_dir, "*_wolf_comparison.csv")  # Comparison.
    if not posteriors:  # The wolf fits have not produced a posterior summary yet.
        print("no wolf posteriors were found, run exp_4_3_wolf_fit.py first")  # Report the gap.
        return 0  # Signal success, since the absence of the input is not an error here.
    index = {}  # Mapping from the individual and model to the comparison row.
    for row in comparison:  # Index the comparison rows by individual and model.
        index[(row["individual_id"], row["model"])] = row  # Store the row under its key.
    names = {  # Human readable names of the candidate models of Section 6.
        "no_advection": "No nonlocal advection",  # Model with a uniform stationary density.
        "tophat": "Top-hat detection",  # Model with the top-hat detection kernel.
        "gaussian": "Gaussian detection",  # Model with the Gaussian detection kernel.
    }  # End of the model name table.
    records = []  # Accumulator for the rows of Table 8.
    for row in posteriors:  # Format one row of Table 8 per fitted model and individual.
        entry = index.get((row["individual_id"], row["model"]), {})  # Comparison of this fit.
        difference = entry.get("elpd_difference", "")  # Difference in expected log density.
        error = entry.get("difference_se", "")  # Standard error of that difference.
        records.append(  # One row of Table 8.
            {  # Formatted posterior summary and comparison of the current fit.
                "individual_id": row["individual_id"],  # Identifier of the animal.
                "model": names.get(row["model"], row["model"]),  # Name of the candidate model.
                "kappa": _interval(row, "aggregation_ratio", 3),  # Aggregation ratio kappa.
                "radius_km": _interval(row, "radius", 2, 1000.0),  # Perceptual range in km.
                # Difference in expected log density with its standard error, when available.
                "elpd_difference": f"{float(difference):.1f} ({float(error):.1f})" if difference else "",
                "max_rhat": f"{float(row['max_rhat']):.3f}",  # Largest R-hat of the fit.
            }
        )  # Row appended to Table 8.
    # Column names of the output, in the order used by the manuscript.
    columns = ["individual_id", "model", "kappa", "radius_km", "elpd_difference", "max_rhat"]
    # Column headers of the typeset table, written in mathematical notation.
    headers = [
        "Individual",  # Header of the identifier column.
        "Model",  # Header of the model column.
        "$\\kappa$, median (95 per cent CrI)",  # Header of the aggregation ratio column.
        "$R$ (km), median (95 per cent CrI)",  # Header of the perceptual range column.
        "$\\Delta$ELPD (SE)",  # Header of the model comparison column.
        "Maximum $\\hat R$",  # Header of the convergence column.
    ]  # Column headers of the typeset table.
    directory = ensure_dir(arguments.output_dir)  # Directory that receives the table files.
    csv_path = write_csv(directory / "Table8.csv", records, columns)  # Comma separated output.
    # LaTeX output in the booktabs style expected by the journal template.
    tex_path = write_latex(directory / "Table8.tex", records, columns, headers, CAPTION, LABEL)
    print(f"wrote {csv_path} and {tex_path} with {len(records)} row(s)")  # Report the outputs.
    return 0  # Signal success to the caller.


# Format a posterior median and its credible interval as a single cell.
# Arguments:
#   row (dict): one row of the posterior summary table.
#   name (str): the name of the summarised parameter.
#   digits (int): the number of decimal places of the formatted values.
#   scale (float): a divisor applied before formatting, for unit conversion.
# Returns:
#   str: the formatted cell, or an empty string when the values are absent.
def _interval(row, name, digits, scale=1.0):
    if f"median_{name}" not in row or not row[f"median_{name}"]:  # The summary is absent.
        return ""  # Report the absence as an empty cell.
    median = float(row[f"median_{name}"]) / scale  # Posterior median in the reported units.
    lower = float(row[f"lower_{name}"]) / scale  # Lower credible bound in the reported units.
    upper = float(row[f"upper_{name}"]) / scale  # Upper credible bound in the reported units.
    return f"{median:.{digits}f} ({lower:.{digits}f}, {upper:.{digits}f})"  # Formatted cell.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the table and propagate the exit status.
