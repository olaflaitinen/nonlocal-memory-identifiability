# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure 5 of the manuscript, showing the relative width of the ninety
# five per cent credible interval of every parameter across the noise levels
# and the sampling designs, separately for each perceptual range and kernel.
# Manuscript: Section 5.3, Fig. 5.
# Inputs: the summary of Experiment 2.4 under results/summary.
# Outputs: results/figures/Fig5.eps and a preview in portable document format.

import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the figure script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.plotting.style import (  # Figure style of the Journal of Mathematical Biology.
    PALETTE,  # Colour-blind-safe palette of the manuscript.
    WIDTH_FULL_MM,  # Full text width of the journal page.
    new_figure,  # Creation of a styled figure and its axes.
    panel_label,  # Lowercase panel labels required by the journal.
    save_figure,  # Writer of the Encapsulated PostScript output.
)

# Line style of each sampling design, as stated in the caption of Fig. 5.
SAMPLING_STYLES = {"sparse": ":", "intermediate": "--", "dense": "-"}  # Dotted, dashed and solid.
# Parameters drawn in the figure, in the order of the manuscript.
PARAMETERS = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameter vector.


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


# Locate the first available summary of the one-dimensional inference.
# Arguments:
#   none.
# Returns:
#   list: the rows of the first summary that exists, possibly empty.
def find_summary():
    folder = Path("results/summary")  # Directory that holds the tracked summary tables.
    for path in sorted(folder.glob("*_mcmc_1d.csv")):  # Inspect every matching summary in turn.
        rows = read_summary(path)  # Rows of the current candidate summary.
        if rows:  # The candidate summary carries at least one row.
            return rows  # Use the first non-empty summary that was found.
    return []  # No matching summary exists yet.


# Entry point of the figure script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    rows = find_summary()  # Rows of the one-dimensional inference summary.
    if not rows:  # The one-dimensional inference has not produced a summary yet.
        print("no inference summary was found, run exp_2_4_mcmc_1d.py first")  # Report the gap.
        return 0  # Signal success, since the absence of the input is not an error here.
    kernels = sorted({row["kernel"] for row in rows})  # Kernel families present in the summary.
    radii = sorted({float(row["radius"]) for row in rows})  # Perceptual ranges of the summary.
    height = min(220.0, 45.0 * max(1, len(kernels)))  # Height that keeps the journal limit.
    figure, axes = new_figure(len(kernels), len(radii), WIDTH_FULL_MM, height)  # Panel grid.
    letters = "abcdefghijkl"  # Lowercase panel labels used by the journal.
    for row_index, kernel in enumerate(kernels):  # One row of panels per kernel family.
        for column_index, radius in enumerate(radii):  # One column of panels per perceptual range.
            axis = axes[row_index, column_index]  # Axis that receives this combination.
            for sampling, style in SAMPLING_STYLES.items():  # One line per sampling design.
                for position, name in enumerate(PARAMETERS):  # One colour per parameter.
                    selected = [  # Rows of this kernel, range and sampling design.
                        item  # Row of the summary table that matches the combination.
                        for item in rows  # Every row of the one-dimensional summary.
                        if item["kernel"] == kernel  # Kernel family of the row matches.
                        and abs(float(item["radius"]) - radius) < 1.0e-12  # Range matches.
                        and item["sampling"] == sampling  # Sampling design matches.
                    ]  # Rows that enter this line of the panel.
                    if not selected:  # No row of the summary matches this combination.
                        continue  # Continue with the next parameter of this line.
                    order = np.argsort([float(item["noise_level"]) for item in selected])  # Order.
                    noise = np.array([float(selected[i]["noise_level"]) for i in order])  # Levels.
                    width = np.array([float(selected[i][f"width_{name}"]) for i in order])  # Widths.
                    axis.semilogy(  # Relative width against the noise level, on a log axis.
                        noise,  # Relative noise levels along the horizontal axis.
                        np.maximum(width, 1.0e-12),  # Relative interval widths, guarded at zero.
                        linestyle=style,  # Line style that identifies the sampling design.
                        color=PALETTE[position],  # Colour that identifies the parameter.
                        marker="o",  # Circles that mark the four noise levels.
                        label=f"{name} {sampling}",  # Legend entry of this line.
                    )  # Line of this parameter and sampling design.
            axis.set_xlabel("noise level eta")  # Label of the horizontal axis of this panel.
            axis.set_ylabel("relative width")  # Label of the vertical axis of this panel.
            index = row_index * len(radii) + column_index  # Position of this panel in the grid.
            panel_label(axis, letters[index % len(letters)])  # Lowercase label of this panel.
    axes[0, 0].legend(frameon=False, fontsize=5, ncol=2)  # Compact legend in the first panel.
    figure.tight_layout()  # Remove the surplus white space around the panels.
    path = save_figure(figure, "Fig5")  # Write the Encapsulated PostScript output.
    print(f"wrote {path}")  # Report the location of the figure.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the figure and propagate the exit status.
