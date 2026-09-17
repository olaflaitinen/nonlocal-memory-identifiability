# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure 3 of the manuscript, showing the log-likelihood surface over
# the advection strength and the memory uptake rate for transient data, and the
# surface over the diffusion rate and the memory decay rate for stationary data
# at fixed combined advection strength.
# Manuscript: Section 5.2, Fig. 3; Proposition 1 and Theorem 1(i).
# Inputs: the archives of Experiment 2.1 under results/raw.
# Outputs: results/figures/Fig3.eps and a preview in portable document format.

import sys  # Process exit status of the figure script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.io import read_hdf5  # Reader of the archives of the surfaces.
from nmi.plotting.style import (  # Figure style of the Journal of Mathematical Biology.
    WIDTH_FULL_MM,  # Full text width of the journal page.
    new_figure,  # Creation of a styled figure and its axes.
    panel_label,  # Lowercase panel labels required by the journal.
    save_figure,  # Writer of the Encapsulated PostScript output.
)


# Locate the archive of the structural checks of Experiment 2.1.
# Arguments:
#   none.
# Returns:
#   pathlib.Path: the first archive that exists, or None when none exists.
def find_archive():
    root = Path("results/raw")  # Directory that holds the raw experiment output.
    if not root.is_dir():  # No experiment has written raw output yet.
        return None  # Report the absence to the caller.
    candidates = sorted(root.glob("*/structural_*.h5"))  # Archives of the structural checks.
    return candidates[0] if candidates else None  # First archive, or None when none exists.


# Draw one log-likelihood surface with its ridge curve and its true parameters.
# Arguments:
#   axis (matplotlib.axes.Axes): the axis that receives the surface.
#   axis_x (numpy.ndarray): the logarithmic axis of the first parameter.
#   axis_y (numpy.ndarray): the logarithmic axis of the second parameter.
#   surface (numpy.ndarray): the evaluated log-likelihood values.
#   product (float): the product that defines the ridge curve.
#   truth (tuple): the true parameters in log space.
# Returns:
#   None: the surface is drawn as a side effect.
def draw_surface(axis, axis_x, axis_y, surface, product, truth):
    finite = np.isfinite(surface)  # Grid points at which the surface could be evaluated.
    levels = 20  # Number of contour levels of the surface.
    if finite.any():  # A surface is drawn only when at least one value is finite.
        shifted = np.where(finite, surface - np.max(surface[finite]), np.nan)  # Relative values.
        axis.contourf(np.exp(axis_y), np.exp(axis_x), shifted, levels=levels)  # Filled contours.
    ridge_x = np.exp(axis_x)  # First parameter of the ridge curve, in natural units.
    ridge_y = product / ridge_x  # Second parameter implied by the fixed product.
    axis.plot(ridge_y, ridge_x, linestyle="--", color="white", linewidth=1.0)  # Ridge curve.
    axis.plot(np.exp(truth[1]), np.exp(truth[0]), marker="x", color="white", markersize=6)  # Truth.
    axis.set_xscale("log")  # Logarithmic horizontal axis, matching the log-space grid.
    axis.set_yscale("log")  # Logarithmic vertical axis, matching the log-space grid.


# Entry point of the figure script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    archive = find_archive()  # Archive of the structural checks of Experiment 2.1.
    if archive is None:  # The structural checks have not been run yet.
        print("no structural archive was found, run exp_2_1_structural_checks.py first")  # Report.
        return 0  # Signal success, since the absence of the input is not an error here.
    arrays, _ = read_hdf5(archive)  # Surfaces and true parameters of the structural checks.
    figure, axes = new_figure(1, 2, WIDTH_FULL_MM, 70.0)  # Two panel figure at full width.
    full_truth = arrays["truth_full"]  # True full parameter vector, in log space.
    advection = float(np.exp(full_truth[1]) * np.exp(full_truth[2]))  # True product alpha beta.
    draw_surface(  # Panel (a), the surface over the advection strength and the uptake rate.
        axes[0, 0],  # Axis that receives the first surface.
        arrays["axis_alpha"],  # Logarithmic axis of the advection strength.
        arrays["axis_beta"],  # Logarithmic axis of the memory uptake rate.
        arrays["surface_alpha_beta"],  # Evaluated log-likelihood surface of panel (a).
        advection,  # Product that defines the ridge curve of Proposition 1.
        (full_truth[1], full_truth[2]),  # True advection strength and uptake rate in log space.
    )  # Surface that exhibits the symmetry of Proposition 1.
    axes[0, 0].set_xlabel("memory uptake rate beta")  # Label of the horizontal axis of panel (a).
    axes[0, 0].set_ylabel("advection strength alpha")  # Label of the vertical axis of panel (a).
    reduced_truth = arrays["truth_reduced"]  # True reduced parameter vector, in log space.
    stationary_truth = arrays["truth_stationary"]  # True stationary parameters, in log space.
    product = float(np.exp(reduced_truth[1]) / np.exp(stationary_truth[0]))  # Product d mu.
    draw_surface(  # Panel (b), the surface over the diffusion rate and the memory decay rate.
        axes[0, 1],  # Axis that receives the second surface.
        arrays["axis_diffusion"],  # Logarithmic axis of the diffusion rate.
        arrays["axis_memory_decay"],  # Logarithmic axis of the memory decay rate.
        arrays["surface_diffusion_decay"],  # Evaluated log-likelihood surface of panel (b).
        product,  # Product that defines the ridge curve of Theorem 1(i).
        (reduced_truth[0], reduced_truth[2]),  # True diffusion rate and decay rate in log space.
    )  # Surface that exhibits the degeneracy of Theorem 1(i).
    axes[0, 1].set_xlabel("memory decay rate mu")  # Label of the horizontal axis of panel (b).
    axes[0, 1].set_ylabel("diffusion rate d")  # Label of the vertical axis of panel (b).
    for position, letter in enumerate("ab"):  # Add the lowercase label of each panel.
        panel_label(axes[0, position], letter)  # Panel label placed outside the plotting area.
    figure.tight_layout()  # Remove the surplus white space around the panels.
    path = save_figure(figure, "Fig3")  # Write the Encapsulated PostScript output.
    print(f"wrote {path}")  # Report the location of the figure.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the figure and propagate the exit status.
