# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure 4 of the manuscript, showing the profile likelihood of each
# free parameter under transient data and under stationary data, at a low and a
# high noise level, with the threshold of the approximate confidence interval.
# Manuscript: Section 5.3, Fig. 4.
# Inputs: the archives of Experiment 2.3 under results/raw.
# Outputs: results/figures/Fig4.eps and a preview in portable document format.

import sys  # Process exit status of the figure script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.io import read_hdf5  # Reader of the archives of the profiles.
from nmi.plotting.style import (  # Figure style of the Journal of Mathematical Biology.
    PALETTE,  # Colour-blind-safe palette of the manuscript.
    WIDTH_FULL_MM,  # Full text width of the journal page.
    new_figure,  # Creation of a styled figure and its axes.
    panel_label,  # Lowercase panel labels required by the journal.
    save_figure,  # Writer of the Encapsulated PostScript output.
)

# Threshold of the approximate ninety five per cent confidence interval.
THRESHOLD = 1.9207296556006175  # One half of the chi squared quantile with one degree of freedom.
# Parameters drawn in the panels of the figure, in the order of the manuscript.
PANEL_PARAMETERS = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameters.


# Collect the archives of the profile likelihood study.
# Arguments:
#   none.
# Returns:
#   dict: one entry per data type, holding the archives of that data type.
def collect_archives():
    root = Path("results/raw")  # Directory that holds the raw experiment output.
    archives = {"transient": [], "stationary": []}  # Archives grouped by data type.
    if not root.is_dir():  # No experiment has written raw output yet.
        return archives  # Report the absence as two empty groups.
    for path in sorted(root.glob("*/profile_*.h5")):  # Inspect every profile archive in turn.
        for data_type in archives:  # Assign the archive to the group named in its file name.
            if path.stem.endswith(data_type):  # The file name ends with the data type.
                archives[data_type].append(path)  # Store the archive in that group.
    return archives  # Archives of the profile likelihood study, grouped by data type.


# Entry point of the figure script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    archives = collect_archives()  # Archives of the profile likelihood study.
    if not any(archives.values()):  # The profile likelihood study has not been run yet.
        print("no profile archive was found, run exp_2_3_profile_likelihood.py first")  # Report.
        return 0  # Signal success, since the absence of the input is not an error here.
    figure, axes = new_figure(2, 2, WIDTH_FULL_MM, 110.0)  # Four panel figure at full width.
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]  # Panels in reading order.
    for position, name in enumerate(PANEL_PARAMETERS):  # Draw one parameter per panel.
        axis = flat_axes[position]  # Axis that receives the profiles of this parameter.
        for data_type, style in (("transient", "-"), ("stationary", "--")):  # Both data types.
            drawn = False  # Whether a profile of this data type was drawn in this panel.
            for path in archives[data_type]:  # Inspect every archive of this data type.
                arrays, _ = read_hdf5(path)  # Profiles stored in the current archive.
                if f"grid_{name}" not in arrays:  # The archive carries no profile of this name.
                    continue  # Continue with the next archive of this data type.
                grid = np.exp(arrays[f"grid_{name}"])  # Grid of the profiled parameter.
                profile = arrays[f"profile_{name}"]  # Profile log-likelihood at the grid values.
                finite = np.isfinite(profile)  # Grid values at which the profile is finite.
                if not finite.any():  # The profile could not be evaluated anywhere.
                    continue  # Continue with the next archive of this data type.
                relative = np.where(finite, np.max(profile[finite]) - profile, np.nan)  # Drop.
                colour = PALETTE[0] if data_type == "transient" else PALETTE[1]  # Data type.
                axis.plot(grid, relative, linestyle=style, color=colour, label=data_type)  # Curve.
                drawn = True  # A profile of this data type has been drawn in this panel.
                break  # One representative archive per data type is enough for the figure.
            if not drawn:  # No archive of this data type carried a profile of this parameter.
                continue  # Continue with the next data type of this panel.
        axis.axhline(THRESHOLD, linestyle=":", color="black", linewidth=0.8)  # Threshold line.
        axis.set_xscale("log")  # Logarithmic horizontal axis, matching the log-space grid.
        axis.set_xlabel(name.replace("_", " "))  # Label of the horizontal axis of this panel.
        axis.set_ylabel("profile drop")  # Label of the vertical axis of this panel.
        axis.set_ylim(0.0, 10.0)  # Vertical extent that shows the threshold clearly.
        axis.legend(frameon=False)  # Legend without a frame, as the journal prefers.
        panel_label(axis, "abcd"[position])  # Panel label placed outside the plotting area.
    figure.tight_layout()  # Remove the surplus white space around the panels.
    path = save_figure(figure, "Fig4")  # Write the Encapsulated PostScript output.
    print(f"wrote {path}")  # Report the location of the figure.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the figure and propagate the exit status.
