# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Table 7 of the manuscript, summarising the telemetry and the estimated
# home ranges of the wolves that satisfy the inclusion rule of Section 4.5.
# Manuscript: Section 6, Table 7; Experiment 4.1.
# Inputs: the tracked summaries under results/summary.
# Outputs: results/tables/Table7.csv and results/tables/Table7.tex.

import argparse  # Command line interface of the table script.
import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the table script.
from pathlib import Path  # Portable filesystem paths.

from nmi.io import ensure_dir, write_csv, write_latex  # Output writers of the table script.

# Caption of the table, reproduced in the LaTeX output.
CAPTION = "Summary of the GPS data for the wolves that satisfy the inclusion rule"  # Caption of Table 7.
# LaTeX label of the table, used for cross references in the manuscript.
LABEL = "tab:wolfdata"  # Label of Table 7.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Table 7")  # Argument parser of the script.
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
    rows = find_summary(arguments.summary_dir, "*_wolf_data.csv")  # Summary of Experiment 4.1.
    if not rows:  # The preprocessing has not produced a summary yet.
        print("no wolf summary was found, run exp_4_1_wolf_preprocess.py first")  # Report the gap.
        return 0  # Signal success, since the absence of the input is not an error here.
    records = []  # Accumulator for the rows of Table 7.
    for row in rows:  # Format one row of Table 7 per included individual.
        if str(row["included"]).lower() != "true":  # The individual failed the inclusion rule.
            continue  # Table 7 lists the included individuals only.
        home_range = row["home_range_km2"]  # Ninety five per cent home range area, in km squared.
        records.append(  # One row of Table 7.
            {  # Formatted telemetry summary of the current individual.
                "individual_id": row["individual_id"],  # Identifier of the animal.
                "pack_sex": f"{row['pack']}, {row['sex']}",  # Pack name and sex of the animal.
                "period": f"{row['first_fix'][:10]} to {row['last_fix'][:10]}",  # Monitoring period.
                "n_fixes": row["n_fixes"],  # Number of retained fixes of the animal.
                "median_interval_hours": f"{float(row['median_interval_hours']):.1f}",  # Interval.
                "n_eff": f"{float(row['n_eff']):.1f}" if row["n_eff"] else "",  # Sample size.
                "home_range_km2": f"{float(home_range):.1f}" if home_range else "",  # Home range.
                "study_area_cells": row["study_area_cells"],  # Interior cells of the study area.
            }
        )  # Row appended to Table 7.
    if not records:  # No individual satisfied the inclusion rule of Section 4.5.
        print("no individual satisfied the inclusion rule")  # Report the empty selection.
        return 0  # Signal success, since an empty selection is a possible outcome.
    # Column names of the output, in the order used by the manuscript.
    columns = [
        "individual_id",  # Identifier of the animal.
        "pack_sex",  # Pack name and sex of the animal.
        "period",  # Monitoring period of the animal.
        "n_fixes",  # Number of retained fixes of the animal.
        "median_interval_hours",  # Median interval between two consecutive fixes.
        "n_eff",  # Effective sample size for area.
        "home_range_km2",  # Ninety five per cent home range area.
        "study_area_cells",  # Interior cells of the study area.
    ]  # Column names of the output, in the order used by the manuscript.
    # Column headers of the typeset table, written in mathematical notation.
    headers = [
        "Animal ID",  # Header of the identifier column.
        "Pack, sex",  # Header of the pack and sex column.
        "Monitoring period",  # Header of the monitoring period column.
        "$n$",  # Header of the number of fixes column.
        "Median interval (h)",  # Header of the fix interval column.
        "$n_{\\mathrm{eff}}$",  # Header of the effective sample size column.
        "95 per cent AKDE (km$^2$)",  # Header of the home range column.
        "Study area cells",  # Header of the study area column.
    ]  # Column headers of the typeset table.
    directory = ensure_dir(arguments.output_dir)  # Directory that receives the table files.
    csv_path = write_csv(directory / "Table7.csv", records, columns)  # Comma separated output.
    # LaTeX output in the booktabs style expected by the journal template.
    tex_path = write_latex(directory / "Table7.tex", records, columns, headers, CAPTION, LABEL)
    print(f"wrote {csv_path} and {tex_path} with {len(records)} row(s)")  # Report the outputs.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the table and propagate the exit status.
