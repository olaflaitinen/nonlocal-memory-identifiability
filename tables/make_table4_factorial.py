# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Table 4 of the manuscript, listing the factors and the levels of the
# full factorial design of the one-dimensional synthetic experiments.
# Manuscript: Section 4.2, Table 4.
# Inputs: the tracked summaries under results/summary.
# Outputs: results/tables/Table4.csv and results/tables/Table4.tex.

import argparse  # Command line interface of the table script.
import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the table script.
from pathlib import Path  # Portable filesystem paths.

from nmi.io import ensure_dir, write_csv, write_latex  # Output writers of the table script.

# Caption of the table, reproduced in the LaTeX output.
CAPTION = "Factorial design of the one-dimensional synthetic experiments"  # Caption of Table 4.
# LaTeX label of the table, used for cross references in the manuscript.
LABEL = "tab:factorial"  # Label of Table 4.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Table 4")  # Argument parser of the script.
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


# Describe one named sampling design as a level of the corresponding factor.
# Arguments:
#   name (str): the name of the sampling design.
#   block (dict): the number of spatial and of temporal observation points.
# Returns:
#   str: a textual description of that level, as printed in Table 4.
def _describe_sampling(name, block):
    return f"{name}: {block['n_space']} by {block['n_time']}"  # Level of the sampling factor.


# Entry point of the table script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
    from nmi.design import design_1d  # Construction of the factorial design.

    config = load_experiment_config("configs/design_1d.yaml")  # Design of the manuscript.
    design = config["design"]  # Block that describes the synthetic design.
    conditions = design_1d(config)  # Factorial design of the one-dimensional experiments.
    records = [  # Rows of Table 4, one per factor of the design.
        {  # Relative noise level of the observation model of equation (9).
            "factor": "Relative noise level eta",  # Name of the factor.
            "n_levels": len(design["noise_levels"]),  # Number of levels of the factor.
            # Levels of the factor, formatted as in Table 4 of the manuscript.
            "levels": ", ".join(f"{float(value):.2f}" for value in design["noise_levels"]),
        },
        {  # Sampling design, given as the number of spatial and temporal points.
            "factor": "Sampling design (spatial points by time points)",  # Name of the factor.
            "n_levels": len(design["sampling_designs"]),  # Number of levels of the factor.
            # Levels of the factor, one entry per named sampling design.
            "levels": "; ".join(
                _describe_sampling(name, block)  # One level of the sampling design factor.
                for name, block in sorted(design["sampling_designs"].items())  # Every design.
            ),
        },
        {  # Perceptual range of the detection kernel.
            "factor": "Perceptual range R",  # Name of the factor.
            "n_levels": len(design["perceptual_ranges"]),  # Number of levels of the factor.
            # Levels of the factor, formatted as in Table 4 of the manuscript.
            "levels": ", ".join(f"{float(value):.3f}" for value in design["perceptual_ranges"]),
        },
        {  # Detection kernel family of the model.
            "factor": "Detection kernel",  # Name of the factor.
            "n_levels": len(design["kernels"]),  # Number of levels of the factor.
            "levels": ", ".join(sorted(design["kernels"])),  # Levels of the factor.
        },
        {  # Total number of transient conditions of the factorial design.
            "factor": "Total transient conditions",  # Name of the derived quantity.
            "n_levels": "",  # A total has no level count of its own.
            "levels": str(len(conditions["transient"])),  # Number of transient conditions.
        },
        {  # Number of profile likelihood conditions reported in Table 6.
            "factor": "Profile likelihood conditions (Table 6)",  # Name of the derived quantity.
            "n_levels": "",  # A total has no level count of its own.
            "levels": str(len(conditions["stationary"])),  # Number of profile conditions.
        },
    ]  # Rows of Table 4.
    columns = ["factor", "n_levels", "levels"]  # Column names of the output.
    headers = ["Factor", "Number of levels", "Levels"]  # Column headers of the typeset table.
    directory = ensure_dir(arguments.output_dir)  # Directory that receives the table files.
    csv_path = write_csv(directory / "Table4.csv", records, columns)  # Comma separated output.
    # LaTeX output in the booktabs style expected by the journal template.
    tex_path = write_latex(directory / "Table4.tex", records, columns, headers, CAPTION, LABEL)
    print(f"wrote {csv_path} and {tex_path} with {len(records)} row(s)")  # Report the outputs.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the table and propagate the exit status.
