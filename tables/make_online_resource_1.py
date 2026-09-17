# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Online Resource 1 of the manuscript, holding the posterior summaries
# of every condition of the one-dimensional factorial design, with a header
# block that carries the article title, the journal, the author, the
# affiliation and the corresponding electronic mail address.
# Manuscript: Section 5.3, Online Resource 1.
# Inputs: the summary of Experiment 2.4 under results/summary.
# Outputs: results/tables/ESM_1.csv.

import argparse  # Command line interface of the electronic supplement script.
import csv  # Reader and writer of the comma separated tables.
import sys  # Process exit status of the script.
from pathlib import Path  # Portable filesystem paths.

from nmi.io import ensure_dir  # Creation of the output directory.

# Title of the article, reproduced in the header block of the supplement.
ARTICLE_TITLE = (
    "Structural and practical identifiability of perception and memory in "  # First half of the title.
    "nonlocal advection-diffusion models of animal movement"  # Second half of the title.
)  # Title of the article.
# Name of the journal to which the article is submitted.
JOURNAL = "Journal of Mathematical Biology"  # Target journal of the article.
# Name of the author of the article.
AUTHOR = "Olaf Yunus Laitinen Imanov"  # Sole author of the article.
# Affiliation of the author of the article.
AFFILIATION = "Department of Mathematics and Statistics, University of Helsinki, Helsinki, Finland"
# Electronic mail address of the corresponding author.
EMAIL = "yunus.imanov@helsinki.fi"  # Corresponding author address required by the journal.
# Citation of the data package used in Section 6 of the article.
DATA_CITATION = "Latham ADM, Boutin S (2019) Movebank Data Repository, https://doi.org/10.5441/001/1.7vr1k987"

# Parameters summarised in the supplement, in the order used by the samplers.
PARAMETERS = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameter vector.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Online Resource 1")  # Argument parser.
    parser.add_argument("--summary-dir", default="results/summary", help="summaries")  # Input.
    parser.add_argument("--output-dir", default="results/tables", help="output directory")  # Output.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Locate the first non-empty summary of the one-dimensional inference.
# Arguments:
#   folder (pathlib.Path): the directory that holds the tracked summaries.
# Returns:
#   list: the rows of the first summary that exists, possibly empty.
def find_summary(folder):
    for path in sorted(Path(folder).glob("*_mcmc_1d.csv")):  # Inspect every matching summary.
        with path.open(encoding="utf-8", newline="") as handle:  # Open the summary for reading.
            rows = list(csv.DictReader(handle))  # Rows of the current candidate summary.
        if rows:  # The candidate summary carries at least one row.
            return rows  # Use the first non-empty summary that was found.
    return []  # No matching summary exists yet.


# Entry point of the electronic supplement script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    rows = find_summary(arguments.summary_dir)  # Rows of the one-dimensional inference summary.
    if not rows:  # The one-dimensional inference has not produced a summary yet.
        print("no inference summary was found, run exp_2_4_mcmc_1d.py first")  # Report the gap.
        return 0  # Signal success, since the absence of the input is not an error here.
    directory = ensure_dir(arguments.output_dir)  # Directory that receives the supplement.
    path = Path(directory) / "ESM_1.csv"  # File name required by the journal for the supplement.
    header = [  # Header block required by the journal at the top of the supplement.
        ["Article title", ARTICLE_TITLE],  # Title of the article.
        ["Journal", JOURNAL],  # Name of the target journal.
        ["Author", AUTHOR],  # Name of the author of the article.
        ["Affiliation", AFFILIATION],  # Affiliation of the author.
        ["Corresponding author", EMAIL],  # Electronic mail address of the corresponding author.
        # Identification of the supplement and of its contents.
        ["Online Resource", "1: posterior summaries for all conditions of the design"],
        ["Data citation", DATA_CITATION],  # Citation of the data package used in Section 6.
        [],  # Blank line that separates the header block from the table itself.
    ]  # Header block of the supplement.
    columns = [  # Column names of the table of the supplement.
        "condition_index",  # Stable integer index of the condition.
        "identifier",  # Stable textual label of the condition.
        "kernel",  # Detection kernel family of the condition.
        "radius",  # Perceptual range R of the condition.
        "noise_level",  # Relative noise level eta of the condition.
        "sampling",  # Name of the sampling design of the condition.
        "converged",  # Whether the convergence criteria of Section 4.4 hold.
    ]  # Columns that describe the condition itself.
    for name in PARAMETERS:  # Add the posterior summary columns of every parameter.
        columns.extend(  # Five columns per parameter, as reported in Section 4.4.
            [  # Names of the summary columns of the current parameter.
                f"truth_{name}",  # True value of the parameter in this condition.
                f"median_{name}",  # Posterior median of the parameter.
                f"lower_{name}",  # Lower end of the credible interval.
                f"upper_{name}",  # Upper end of the credible interval.
                f"width_{name}",  # Relative width of the credible interval.
            ]
        )  # Summary columns of the current parameter.
    with path.open("w", encoding="utf-8", newline="") as handle:  # Open the supplement file.
        writer = csv.writer(handle)  # Plain writer used for the header block.
        writer.writerows(header)  # Write the header block required by the journal.
        writer.writerow(columns)  # Write the column names of the table.
        for row in sorted(rows, key=lambda item: int(item["condition_index"])):  # Stable order.
            writer.writerow([row.get(name, "") for name in columns])  # One line per condition.
    print(f"wrote {path} with {len(rows)} condition(s)")  # Report the location of the supplement.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the supplement and propagate the exit status.
