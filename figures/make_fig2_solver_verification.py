# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure 2 of the manuscript, showing the convergence of the forward
# solver, the mass conservation error and the transient solution approaching
# the steady state.
# Manuscript: Section 5.1, Fig. 2.
# Inputs: the summaries of Experiments 1.1 and 1.2 under results/summary and
# the snapshots under results/raw. Outputs: results/figures/Fig2.eps.

import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the figure script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.io import read_hdf5  # Reader of the archives of the snapshots.
from nmi.plotting.style import (  # Figure style of the Journal of Mathematical Biology.
    PALETTE,  # Colour-blind-safe palette of the manuscript.
    WIDTH_FULL_MM,  # Full text width of the journal page.
    new_figure,  # Creation of a styled figure and its axes.
    panel_label,  # Lowercase panel labels required by the journal.
    save_figure,  # Writer of the Encapsulated PostScript output.
)

# Experiment identifier whose summaries the figure reads by default.
DEFAULT_EXPERIMENT = "exp_1_1_convergence_1d"  # Convergence study of Experiment 1.1.


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


# Locate the first available summary of the convergence study.
# Arguments:
#   suffix (str): the file name suffix of the summary.
# Returns:
#   list: the rows of the first summary that exists, possibly empty.
def find_summary(suffix):
    folder = Path("results/summary")  # Directory that holds the tracked summary tables.
    for path in sorted(folder.glob(f"*{suffix}")):  # Inspect every matching summary in turn.
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
    convergence = find_summary("_convergence_1d.csv")  # Rows of the convergence study.
    if not convergence:  # The convergence study has not been run yet.
        print("no convergence summary was found, run exp_1_1_convergence_1d.py first")  # Report.
        return 0  # Signal success, since the absence of the input is not an error here.
    figure, axes = new_figure(1, 3, WIDTH_FULL_MM, 58.0)  # Three panel figure at full width.
    markers = {"tophat": "o", "gaussian": "s"}  # Circles and squares, as stated in the caption.
    colours = {"tophat": PALETTE[0], "gaussian": PALETTE[1]}  # Colours of the two kernels.
    for kernel in ("tophat", "gaussian"):  # Draw one curve per detection kernel family.
        # Rows of the grid refinement study that belong to this kernel family.
        rows = [row for row in convergence if row["kernel"] == kernel and row["refinement"] == "grid"]
        if not rows:  # No grid refinement row exists for this kernel family.
            continue  # Continue with the next detection kernel family.
        grids = np.array([float(row["n_points"]) for row in rows])  # Grid sizes of the study.
        errors = np.array([max(float(row["relative_error"]), 1.0e-16) for row in rows])  # Errors.
        axes[0, 0].loglog(  # Relative error against the grid size, on logarithmic axes.
            grids,  # Grid sizes along the horizontal axis.
            errors,  # Relative errors along the vertical axis.
            marker=markers[kernel],  # Marker that identifies the kernel family.
            linestyle="-",  # Solid line joining the marked values.
            color=colours[kernel],  # Colour that identifies the kernel family.
            label=kernel,  # Legend entry of this kernel family.
        )  # Curve of the grid refinement study.
    # Grid sizes of the refinement study, used to place the reference slope.
    reference_grid = np.array([float(row["n_points"]) for row in convergence if row["refinement"] == "grid"])
    if reference_grid.size:  # A reference slope is drawn only when grid rows exist.
        span = np.array([reference_grid.min(), reference_grid.max()])  # Extent of the slope line.
        axes[0, 0].loglog(  # Reference slope of second order, drawn with a dotted line.
            span,  # Extent of the reference slope along the horizontal axis.
            1.0e-3 * (span / span[0]) ** (-2.0),  # Second order decay of the reference slope.
            linestyle=":",  # Dotted line, as stated in the caption of Fig. 2.
            color="black",  # Neutral colour of the reference slope.
            label="second order",  # Legend entry of the reference slope.
        )  # Reference slope of the first panel.
    axes[0, 0].set_xlabel("grid size N")  # Label of the horizontal axis of panel (a).
    axes[0, 0].set_ylabel("relative L2 error")  # Label of the vertical axis of panel (a).
    axes[0, 0].legend(frameon=False)  # Legend without a frame, as the journal prefers.
    for kernel in ("tophat", "gaussian"):  # Draw the mass conservation error of each kernel.
        rows = [row for row in convergence if row["kernel"] == kernel]  # Rows of this kernel.
        if not rows:  # No row exists for this kernel family.
            continue  # Continue with the next detection kernel family.
        grids = np.array([float(row["n_points"]) for row in rows])  # Grid sizes of the study.
        mass = np.array([max(float(row["mass_error"]), 1.0e-18) for row in rows])  # Mass errors.
        axes[0, 1].semilogy(  # Mass error against the grid size, on a logarithmic vertical axis.
            grids,  # Grid sizes along the horizontal axis.
            mass,  # Relative mass conservation errors along the vertical axis.
            marker=markers[kernel],  # Marker that identifies the kernel family.
            linestyle="none",  # Only the markers are drawn, since the values are scattered.
            color=colours[kernel],  # Colour that identifies the kernel family.
            label=kernel,  # Legend entry of this kernel family.
        )  # Markers of the mass conservation study.
    axes[0, 1].set_xlabel("grid size N")  # Label of the horizontal axis of panel (b).
    axes[0, 1].set_ylabel("relative mass error")  # Label of the vertical axis of panel (b).
    axes[0, 1].legend(frameon=False)  # Legend without a frame, as the journal prefers.
    snapshot = _find_snapshot()  # Archive of the transient snapshots of Experiment 1.2.
    if snapshot is not None:  # The steady state study has produced an archive.
        arrays, _ = read_hdf5(snapshot)  # Snapshots and steady state of that archive.
        grid = np.linspace(0.0, 1.0, arrays["steady_state"].shape[0], endpoint=False)  # Grid.
        for row in arrays["snapshots"]:  # Draw each transient snapshot with a thin line.
            axes[0, 2].plot(grid, row, linewidth=0.4, color=PALETTE[4])  # Thin transient line.
        axes[0, 2].plot(  # Steady state of the fixed-point solver, drawn with a thick line.
            grid,  # Grid points of the periodic domain.
            arrays["steady_state"],  # Steady state computed by the fixed-point solver.
            linewidth=1.4,  # Thick line, as stated in the caption of Fig. 2.
            color=PALETTE[0],  # First colour of the palette of the manuscript.
            label="steady state",  # Legend entry of the steady state.
        )  # Curve of the steady state.
        axes[0, 2].legend(frameon=False)  # Legend without a frame, as the journal prefers.
    axes[0, 2].set_xlabel("position x")  # Label of the horizontal axis of panel (c).
    axes[0, 2].set_ylabel("density u")  # Label of the vertical axis of panel (c).
    for position, letter in enumerate("abc"):  # Add the lowercase label of each panel.
        panel_label(axes[0, position], letter)  # Panel label placed outside the plotting area.
    figure.tight_layout()  # Remove the surplus white space around the panels.
    path = save_figure(figure, "Fig2")  # Write the Encapsulated PostScript output.
    print(f"wrote {path}")  # Report the location of the figure.
    return 0  # Signal success to the caller.


# Locate an archive of the transient snapshots of Experiment 1.2.
# Arguments:
#   none.
# Returns:
#   pathlib.Path: the first archive that exists, or None when none exists.
def _find_snapshot():
    root = Path("results/raw")  # Directory that holds the raw experiment output.
    if not root.is_dir():  # No experiment has written raw output yet.
        return None  # Report the absence to the caller.
    candidates = sorted(root.glob("*/steady_*.h5"))  # Archives of the steady state study.
    return candidates[0] if candidates else None  # First archive, or None when none exists.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the figure and propagate the exit status.
