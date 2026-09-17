# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure 1 of the manuscript, showing the structure of model (1), the
# two detection kernels and their Fourier transforms at the torus wavenumbers.
# The figure is analytic and needs no experiment output.
# Manuscript: Section 2.2, Fig. 1.
# Inputs: none. Outputs: results/figures/Fig1.eps and a preview in portable
# document format.

import sys  # Process exit status of the figure script.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.kernels import kernel_hat_1d, kernel_values_1d  # Kernel profiles and transforms.
from nmi.plotting.style import (  # Figure style of the Journal of Mathematical Biology.
    PALETTE,  # Colour-blind-safe palette of the manuscript.
    WIDTH_FULL_MM,  # Full text width of the journal page.
    new_figure,  # Creation of a styled figure and its axes.
    panel_label,  # Lowercase panel labels required by the journal.
    save_figure,  # Writer of the Encapsulated PostScript output.
)


# Draw the schematic of model (1) in the first panel.
# Arguments:
#   axis (matplotlib.axes.Axes): the axis that receives the schematic.
# Returns:
#   None: the schematic is drawn as a side effect.
def draw_schematic(axis):
    axis.set_xlim(0.0, 1.0)  # Horizontal extent of the schematic, in axis units.
    axis.set_ylim(0.0, 1.0)  # Vertical extent of the schematic, in axis units.
    axis.axis("off")  # The schematic carries no axes of its own.
    axis.text(0.22, 0.78, "density u", ha="center", va="center", color=PALETTE[0])  # Density box.
    axis.text(0.78, 0.78, "map k", ha="center", va="center", color=PALETTE[1])  # Cognitive map box.
    # Box that represents the spatially averaged cognitive map of model (1).
    axis.text(0.50, 0.30, "perceived map G_R * k", ha="center", va="center", color=PALETTE[2])
    axis.annotate(  # Arrow that represents the uptake of presence into the map.
        "",  # The arrow carries its label separately, below.
        xy=(0.66, 0.78),  # Head of the arrow, at the cognitive map.
        xytext=(0.34, 0.78),  # Tail of the arrow, at the density.
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},  # Plain arrow.
    )  # Arrow from the density to the cognitive map.
    axis.text(0.50, 0.86, "uptake beta u", ha="center", va="center")  # Label of that arrow.
    axis.annotate(  # Arrow that represents the advection up the perceived gradient.
        "",  # The arrow carries its label separately, below.
        xy=(0.22, 0.70),  # Head of the arrow, at the density.
        xytext=(0.42, 0.36),  # Tail of the arrow, at the perceived map.
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},  # Plain arrow.
    )  # Arrow from the perceived map to the density.
    axis.annotate(  # Arrow that represents the spatial averaging of the map.
        "",  # The arrow carries its label separately, below.
        xy=(0.58, 0.36),  # Head of the arrow, at the perceived map.
        xytext=(0.78, 0.70),  # Tail of the arrow, at the cognitive map.
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},  # Plain arrow.
    )  # Arrow from the cognitive map to the perceived map.
    axis.text(0.14, 0.50, "advection alpha", ha="center", va="center", rotation=60)  # Label.
    axis.text(0.88, 0.50, "decay mu k", ha="center", va="center", rotation=-60)  # Label.
    axis.text(0.50, 0.10, "diffusion d", ha="center", va="center")  # Label of the diffusion term.


# Entry point of the figure script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    figure, axes = new_figure(1, 3, WIDTH_FULL_MM, 58.0)  # Three panel figure at full width.
    draw_schematic(axes[0, 0])  # Panel (a), the structure of model (1).
    radius = 0.15  # Perceptual range used in the second and third panels.
    grid = np.linspace(-0.5, 0.5, 2001)  # Displacements at which the kernels are drawn.
    axes[0, 1].plot(  # Top-hat kernel of equation (2), drawn with a solid line.
        grid,  # Displacements at which the kernel is evaluated.
        kernel_values_1d(grid, radius, "tophat"),  # Values of the top-hat kernel.
        linestyle="-",  # Solid line, as stated in the caption of Fig. 1.
        color=PALETTE[0],  # First colour of the palette of the manuscript.
        label="top-hat",  # Legend entry of the top-hat kernel.
    )  # Curve of the top-hat kernel.
    axes[0, 1].plot(  # Gaussian kernel of equation (2), drawn with a dashed line.
        grid,  # Displacements at which the kernel is evaluated.
        kernel_values_1d(grid, radius, "gaussian"),  # Values of the Gaussian kernel.
        linestyle="--",  # Dashed line, as stated in the caption of Fig. 1.
        color=PALETTE[1],  # Second colour of the palette of the manuscript.
        label="Gaussian",  # Legend entry of the Gaussian kernel.
    )  # Curve of the Gaussian kernel.
    axes[0, 1].set_xlabel("displacement y")  # Label of the horizontal axis of panel (b).
    axes[0, 1].set_ylabel("kernel G_R(y)")  # Label of the vertical axis of panel (b).
    axes[0, 1].legend(frameon=False)  # Legend without a frame, as the journal prefers.
    modes = np.arange(0, 13)  # Mode indices at which the transforms are marked.
    wavenumbers = 2.0 * np.pi * modes  # Wavenumbers xi_m of those modes on the unit torus.
    axes[0, 2].axhline(0.0, color="black", linewidth=0.5)  # Zero line that marks the sign changes.
    axes[0, 2].plot(  # Transform of the top-hat kernel, marked with circles.
        modes,  # Mode indices along the horizontal axis.
        kernel_hat_1d(radius * wavenumbers, "tophat"),  # Values of the top-hat transform.
        marker="o",  # Circles, as stated in the caption of Fig. 1.
        linestyle="-",  # Solid line joining the marked values.
        color=PALETTE[0],  # First colour of the palette of the manuscript.
        label="top-hat",  # Legend entry of the top-hat transform.
    )  # Marked values of the top-hat transform.
    axes[0, 2].plot(  # Transform of the Gaussian kernel, marked with squares.
        modes,  # Mode indices along the horizontal axis.
        kernel_hat_1d(radius * wavenumbers, "gaussian"),  # Values of the Gaussian transform.
        marker="s",  # Squares, as stated in the caption of Fig. 1.
        linestyle="--",  # Dashed line joining the marked values.
        color=PALETTE[1],  # Second colour of the palette of the manuscript.
        label="Gaussian",  # Legend entry of the Gaussian transform.
    )  # Marked values of the Gaussian transform.
    axes[0, 2].set_xlabel("mode index m")  # Label of the horizontal axis of panel (c).
    axes[0, 2].set_ylabel("transform G_hat(R xi_m)")  # Label of the vertical axis of panel (c).
    axes[0, 2].legend(frameon=False)  # Legend without a frame, as the journal prefers.
    for position, letter in enumerate("abc"):  # Add the lowercase label of each panel.
        panel_label(axes[0, position], letter)  # Panel label placed outside the plotting area.
    figure.tight_layout()  # Remove the surplus white space around the panels.
    path = save_figure(figure, "Fig1")  # Write the Encapsulated PostScript output.
    print(f"wrote {path}")  # Report the location of the figure.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the figure and propagate the exit status.
