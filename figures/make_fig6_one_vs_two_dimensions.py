# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure 6 of the manuscript, comparing the relative credible interval
# width in two space dimensions with the corresponding width in one dimension
# for the selected conditions, with one symbol per parameter and open or filled
# symbols for the two detection kernels.
# Manuscript: Section 5.4, Fig. 6.
# Inputs: the summary of Experiment 3.3 under results/summary.
# Outputs: results/figures/Fig6.eps and a preview in portable document format.

import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the figure script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.plotting.style import (  # Figure style of the Journal of Mathematical Biology.
    PALETTE,  # Colour-blind-safe palette of the manuscript.
    WIDTH_COLUMN_MM,  # Single column width of the journal page.
    new_figure,  # Creation of a styled figure and its axes.
    save_figure,  # Writer of the Encapsulated PostScript output.
)

# Marker of each parameter, as stated in the caption of Fig. 6.
PARAMETER_MARKERS = {  # Circles, squares, triangles and diamonds.
    "diffusion": "o",  # Circles mark the diffusion rate d.
    "advection": "s",  # Squares mark the combined advection strength gamma.
    "memory_decay": "^",  # Triangles mark the memory decay rate mu.
    "radius": "D",  # Diamonds mark the perceptual range R.
}  # End of the marker table.


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


# Entry point of the figure script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    folder = Path("results/summary")  # Directory that holds the tracked summary tables.
    rows = []  # Accumulator for the rows of the comparison table.
    for path in sorted(folder.glob("*_analysis_2d.csv")):  # Inspect every comparison summary.
        rows = read_summary(path)  # Rows of the current candidate summary.
        if rows:  # The candidate summary carries at least one row.
            break  # Use the first non-empty summary that was found.
    if not rows:  # The two-dimensional analysis has not produced a summary yet.
        print("no two-dimensional comparison was found, run exp_3_3_analysis_2d.py first")  # Log.
        return 0  # Signal success, since the absence of the input is not an error here.
    figure, axes = new_figure(1, 1, WIDTH_COLUMN_MM, 80.0)  # Single column figure of one panel.
    axis = axes[0, 0]  # Axis that receives the comparison of the two widths.
    values = []  # Accumulator for the plotted widths, used to place the identity line.
    for name, marker in PARAMETER_MARKERS.items():  # One marker per compared parameter.
        for kernel, filled in (("tophat", False), ("gaussian", True)):  # Open and filled symbols.
            selected = [  # Rows of this parameter and kernel family with both widths present.
                row  # Row of the comparison table.
                for row in rows  # Every row of the comparison summary.
                # Keep only the rows whose one-dimensional counterpart exists.
                if row["parameter"] == name and row["kernel"] == kernel and row["width_1d"]
            ]  # Rows that enter this group of symbols.
            if not selected:  # No row of the summary matches this combination.
                continue  # Continue with the next kernel family of this parameter.
            width_1d = np.array([float(row["width_1d"]) for row in selected])  # Widths in one.
            width_2d = np.array([float(row["width_2d"]) for row in selected])  # Widths in two.
            values.extend(list(width_1d) + list(width_2d))  # Record the plotted widths.
            axis.loglog(  # Two-dimensional width against the one-dimensional width.
                width_1d,  # Relative widths in one dimension along the horizontal axis.
                width_2d,  # Relative widths in two dimensions along the vertical axis.
                marker=marker,  # Marker that identifies the compared parameter.
                linestyle="none",  # Only the markers are drawn, since the values are scattered.
                markerfacecolor=PALETTE[0] if filled else "none",  # Filled for the Gaussian kernel.
                markeredgecolor=PALETTE[0],  # Common edge colour of both kernel families.
                label=f"{name} {kernel}",  # Legend entry of this group of symbols.
            )  # Markers of this parameter and kernel family.
    if values:  # The identity line is drawn only when at least one symbol was plotted.
        span = np.array([min(values), max(values)])  # Extent of the identity line.
        axis.loglog(span, span, linestyle="--", color="black", linewidth=0.8)  # Identity line.
    axis.set_xlabel("relative width in one dimension")  # Label of the horizontal axis.
    axis.set_ylabel("relative width in two dimensions")  # Label of the vertical axis.
    axis.legend(frameon=False, fontsize=5, ncol=2)  # Compact legend without a frame.
    figure.tight_layout()  # Remove the surplus white space around the panel.
    path = save_figure(figure, "Fig6")  # Write the Encapsulated PostScript output.
    print(f"wrote {path}")  # Report the location of the figure.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the figure and propagate the exit status.
