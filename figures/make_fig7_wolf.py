# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure 7 of the manuscript, showing the estimated utilisation
# distribution of a representative wolf with its fixes and study area, the
# fitted steady state at the posterior median, the posterior of the perceptual
# range under both detection kernels and the differences in expected log
# pointwise predictive density.
# Manuscript: Section 6, Fig. 7.
# Data: Latham ADM, Boutin S (2019), Movebank Data Repository,
# https://doi.org/10.5441/001/1.7vr1k987, licensed CC0 1.0.
# Inputs: the outputs of Experiments 4.1 and 4.3.
# Outputs: results/figures/Fig7.eps and a preview in portable document format.

import csv  # Reader of the tracked summary tables.
import sys  # Process exit status of the figure script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.
import pandas  # Tabular handling of the cleaned table of location fixes.

from nmi.plotting.style import (  # Figure style of the Journal of Mathematical Biology.
    PALETTE,  # Colour-blind-safe palette of the manuscript.
    WIDTH_FULL_MM,  # Full text width of the journal page.
    new_figure,  # Creation of a styled figure and its axes.
    panel_label,  # Lowercase panel labels required by the journal.
    save_figure,  # Writer of the Encapsulated PostScript output.
)
from nmi.wolf.masked_solver import masked_steady_state  # Stationary density on the masked grid.


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
#   pattern (str): the glob pattern of the summary file name.
# Returns:
#   list: the rows of the first summary that exists, possibly empty.
def find_summary(pattern):
    folder = Path("results/summary")  # Directory that holds the tracked summary tables.
    for path in sorted(folder.glob(pattern)):  # Inspect every matching summary in turn.
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
    posteriors = find_summary("*_wolf_posteriors.csv")  # Posterior summaries of Experiment 4.3.
    comparison = find_summary("*_wolf_comparison.csv")  # Model comparison of Experiment 4.3.
    source = Path("results/raw/exp_4_1")  # Directory that holds the output of Experiment 4.1.
    fixes_path = source / "cleaned_fixes.csv"  # Internal cleaned table of location fixes.
    if not posteriors or not fixes_path.is_file():  # The wolf fits have not been produced yet.
        print("no wolf posteriors were found, run exp_4_3_wolf_fit.py first")  # Report the gap.
        return 0  # Signal success, since the absence of the input is not an error here.
    identifier = posteriors[0]["individual_id"]  # Representative individual shown in the figure.
    area_path = source / f"study_area_{identifier}.npz"  # Study area of that individual.
    if not area_path.is_file():  # The study area of the individual has not been built.
        print(f"missing study area for individual {identifier}")  # Report the missing input.
        return 0  # Signal success, since the absence of the input is not an error here.
    with np.load(area_path) as archive:  # Open the archive of the study area.
        mask = np.asarray(archive["mask"], dtype=bool)  # Boolean mask of the study area.
        x_centres = np.asarray(archive["x_centres"], dtype=float)  # Cell centres, first axis.
        y_centres = np.asarray(archive["y_centres"], dtype=float)  # Cell centres, second axis.
        cell = float(np.asarray(archive["cell"], dtype=float)[0])  # Cell width, in metres.
    fixes = pandas.read_csv(fixes_path)  # Cleaned and projected table of location fixes.
    group = fixes[fixes["individual_id"].astype(str) == str(identifier)]  # Fixes of the animal.
    figure, axes = new_figure(2, 2, WIDTH_FULL_MM, 120.0)  # Four panel figure at full width.
    extent = [x_centres[0], x_centres[-1], y_centres[0], y_centres[-1]]  # Extent of the panels.
    uniform = mask.astype(float) / (float(np.sum(mask)) * cell * cell)  # Uniform reference density.
    axes[0, 0].imshow(uniform, origin="lower", extent=extent, aspect="equal")  # Study area.
    axes[0, 0].plot(  # Retained fixes of the representative individual.
        group["easting_m"].to_numpy(dtype=float),  # Projected first coordinate of the fixes.
        group["northing_m"].to_numpy(dtype=float),  # Projected second coordinate of the fixes.
        marker=".",  # Small points that mark the individual fixes.
        linestyle="none",  # Only the points are drawn, without a connecting line.
        markersize=1.0,  # Small markers, so that the underlying density stays visible.
        color="white",  # Neutral colour that contrasts with the density.
    )  # Points of the retained fixes.
    axes[0, 0].contour(  # Boundary of the study area, drawn as a single contour line.
        np.linspace(extent[0], extent[1], mask.shape[1]),  # Coordinates along the first axis.
        np.linspace(extent[2], extent[3], mask.shape[0]),  # Coordinates along the second axis.
        mask.astype(float),  # Indicator of the study area.
        levels=[0.5],  # Single contour that traces the boundary of the study area.
        colors="black",  # Neutral colour of the boundary line.
        linewidths=0.6,  # Line width above the lower limit of the journal.
    )  # Boundary of the study area.
    axes[0, 0].set_xlabel("easting (m)")  # Label of the horizontal axis of panel (a).
    axes[0, 0].set_ylabel("northing (m)")  # Label of the vertical axis of panel (a).
    # Fit of the top-hat model for the representative individual, when it exists.
    tophat = [row for row in posteriors if row["individual_id"] == identifier and row["model"] == "tophat"]
    if tophat:  # A fitted top-hat model is available for this individual.
        fitted = masked_steady_state(  # Fitted steady state at the posterior median.
            float(tophat[0]["median_aggregation_ratio"]),  # Posterior median aggregation ratio.
            float(tophat[0]["median_radius"]),  # Posterior median perceptual range, in metres.
            "tophat",  # Detection kernel family of the fitted model.
            mask,  # Boolean mask of the study area.
            cell,  # Cell width of the Cartesian grid, in metres.
        )  # Stationary density at the posterior median of the top-hat model.
        axes[0, 1].imshow(fitted["density"], origin="lower", extent=extent, aspect="equal")  # Map.
    axes[0, 1].set_xlabel("easting (m)")  # Label of the horizontal axis of panel (b).
    axes[0, 1].set_ylabel("northing (m)")  # Label of the vertical axis of panel (b).
    for position, model in enumerate(("tophat", "gaussian")):  # Both detection kernel families.
        rows = [row for row in posteriors if row["model"] == model]  # Fits of this kernel family.
        if not rows:  # No fit of this kernel family is available.
            continue  # Continue with the next detection kernel family.
        medians = np.array([float(row["median_radius"]) for row in rows])  # Median ranges.
        lower = np.array([float(row["lower_radius"]) for row in rows])  # Lower credible bounds.
        upper = np.array([float(row["upper_radius"]) for row in rows])  # Upper credible bounds.
        style = "-" if model == "tophat" else "--"  # Solid for top-hat, dashed for Gaussian.
        axes[1, 0].errorbar(  # Median perceptual range with its credible interval per individual.
            np.arange(medians.size) + 0.1 * position,  # Individual index along the horizontal axis.
            medians,  # Median perceptual range along the vertical axis.
            yerr=[medians - lower, upper - medians],  # Extent of the credible interval.
            marker="o",  # Circles that mark the posterior medians.
            linestyle=style,  # Line style that identifies the detection kernel family.
            color=PALETTE[position],  # Colour that identifies the detection kernel family.
            capsize=2.0,  # Small caps on the error bars, as the journal prefers.
            label=model,  # Legend entry of this detection kernel family.
        )  # Error bars of this detection kernel family.
    axes[1, 0].set_xlabel("individual")  # Label of the horizontal axis of panel (c).
    axes[1, 0].set_ylabel("perceptual range R (m)")  # Label of the vertical axis of panel (c).
    axes[1, 0].legend(frameon=False)  # Legend without a frame, as the journal prefers.
    if comparison:  # The model comparison is drawn only when it is available.
        models = sorted({row["model"] for row in comparison})  # Candidate models of Section 6.
        differences = []  # Mean difference in expected log pointwise predictive density.
        errors = []  # Mean standard error of those differences.
        for model in models:  # Summarise the comparison of each candidate model.
            rows = [row for row in comparison if row["model"] == model]  # Rows of this model.
            # Mean difference from the preferred model over the fitted individuals.
            differences.append(float(np.mean([float(row["elpd_difference"]) for row in rows])))
            # Mean standard error of that difference over the fitted individuals.
            errors.append(float(np.mean([float(row["difference_se"]) for row in rows])))
        axes[1, 1].errorbar(  # Differences in expected log pointwise predictive density.
            np.arange(len(models)),  # Candidate model index along the horizontal axis.
            differences,  # Mean differences along the vertical axis.
            yerr=errors,  # One standard error of the mean difference.
            marker="s",  # Squares that mark the mean differences.
            linestyle="none",  # Only the markers and the error bars are drawn.
            color=PALETTE[2],  # Third colour of the palette of the manuscript.
            capsize=2.0,  # Small caps on the error bars, as the journal prefers.
        )  # Error bars of the model comparison.
        axes[1, 1].set_xticks(np.arange(len(models)))  # One tick per candidate model.
        axes[1, 1].set_xticklabels(models, rotation=20)  # Names of the candidate models.
    axes[1, 1].set_ylabel("difference in ELPD")  # Label of the vertical axis of panel (d).
    for position, letter in enumerate("abcd"):  # Add the lowercase label of each panel.
        panel_label([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]][position], letter)  # Label.
    figure.tight_layout()  # Remove the surplus white space around the panels.
    path = save_figure(figure, "Fig7")  # Write the Encapsulated PostScript output.
    print(f"wrote {path}")  # Report the location of the figure.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the figure and propagate the exit status.
