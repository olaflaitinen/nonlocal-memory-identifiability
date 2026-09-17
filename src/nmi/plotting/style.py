# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Figure style for the Journal of Mathematical Biology. The module
# sets the Matplotlib parameters required by the journal, provides the figure
# widths, a colour-blind-safe palette with distinct line styles and markers,
# and writes each figure as Encapsulated PostScript with an additional preview
# in portable document format.
# Manuscript: Section 4.6; the style is documented in docs/figure_style.md.
# Inputs: none. Outputs: configured Matplotlib parameters and saved figures.

import warnings  # Reporting of a missing preferred typeface.
from pathlib import Path  # Portable filesystem paths.

import matplotlib  # Configuration of the plotting backend and its parameters.

matplotlib.use("Agg")  # Non-interactive backend suitable for batch jobs on the cluster.

import matplotlib.pyplot as plt  # Figure and axes creation.  # noqa: E402

# Conversion factor from millimetres to inches, used by the figure sizes.
MM_TO_INCH = 1.0 / 25.4  # One inch equals 25.4 millimetres exactly.
# Width of a full width figure of the journal, in millimetres.
WIDTH_FULL_MM = 174.0  # Full text width of the journal page.
# Width of a single column figure of the journal, in millimetres.
WIDTH_COLUMN_MM = 84.0  # Single column width of the journal page.
# Largest admissible figure height of the journal, in millimetres.
HEIGHT_MAX_MM = 234.0  # Largest height that fits on one journal page.
# Typefaces accepted by the journal, in order of preference.
PREFERRED_FONTS = ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")  # Sans-serif faces.
# Colour-blind-safe palette used for every figure of the manuscript.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")  # Okabe and Ito.
# Line styles used in addition to colour, so that figures remain readable in grey.
LINE_STYLES = ("-", "--", ":", "-.")  # Solid, dashed, dotted and dash-dotted lines.
# Markers used in addition to colour, so that figures remain readable in grey.
MARKERS = ("o", "s", "^", "D", "v", "P")  # Circle, square, triangle, diamond and further shapes.


# Select the first available typeface among those accepted by the journal.
# Arguments:
#   none.
# Returns:
#   str: the name of the typeface that Matplotlib will use.
def select_font():
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}  # Installed.
    for name in PREFERRED_FONTS:  # Inspect the accepted typefaces in order of preference.
        if name in available:  # The typeface is installed in the active environment.
            if name not in ("Arial", "Helvetica"):  # A metric compatible substitute is used.
                # Record that a metric compatible substitute is used instead of Arial.
                warnings.warn(f"Using the substitute typeface {name} instead of Arial", stacklevel=2)
            return name  # Use the first accepted typeface that is installed.
    # Record that no accepted typeface is installed in the active environment.
    warnings.warn("No accepted sans-serif typeface was found, using the Matplotlib default", stacklevel=2)
    return "sans-serif"  # Fall back to whatever sans-serif face Matplotlib provides.


# Apply the figure style required by the journal.
# Arguments:
#   base_size (float): base font size in points, between eight and twelve.
# Returns:
#   None: the Matplotlib parameters are set as a side effect.
def apply_style(base_size=9.0):
    font = select_font()  # Typeface accepted by the journal, or a metric compatible substitute.
    plt.rcParams["font.family"] = "sans-serif"  # The journal requires sans-serif lettering.
    plt.rcParams["font.sans-serif"] = [font]  # Selected typeface, with no serif fallback.
    plt.rcParams["font.size"] = float(base_size)  # Base font size of the figure text.
    plt.rcParams["axes.labelsize"] = float(base_size)  # Font size of the axis labels.
    plt.rcParams["axes.titlesize"] = float(base_size)  # Font size of any axis title.
    plt.rcParams["xtick.labelsize"] = float(base_size) - 1.0  # Font size of the tick labels.
    plt.rcParams["ytick.labelsize"] = float(base_size) - 1.0  # Font size of the tick labels.
    plt.rcParams["legend.fontsize"] = float(base_size) - 1.0  # Font size of the legend entries.
    plt.rcParams["axes.linewidth"] = 0.6  # Width of the axis frame, above the lower limit.
    plt.rcParams["lines.linewidth"] = 1.0  # Width of the plotted lines, above the lower limit.
    plt.rcParams["lines.markersize"] = 3.5  # Size of the plot markers in points.
    plt.rcParams["xtick.major.width"] = 0.6  # Width of the major ticks, above the lower limit.
    plt.rcParams["ytick.major.width"] = 0.6  # Width of the major ticks, above the lower limit.
    plt.rcParams["ps.fonttype"] = 42  # Embed the fonts as Type 42, as the journal requires.
    plt.rcParams["pdf.fonttype"] = 42  # Use the same embedding in the portable document preview.
    plt.rcParams["ps.useafm"] = False  # Embed the selected typeface rather than a core font.
    plt.rcParams["savefig.dpi"] = 600  # Resolution of any rasterised element of a figure.
    plt.rcParams["savefig.bbox"] = "standard"  # Keep the requested figure size exactly.
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=list(PALETTE))  # Palette of the paper.
    plt.rcParams["figure.facecolor"] = "white"  # White background, as required for print.
    plt.rcParams["axes.grid"] = False  # The journal figures carry no background grid.


# Figure size in inches from a width and a height given in millimetres.
# Arguments:
#   width_mm (float): figure width in millimetres.
#   height_mm (float): figure height in millimetres.
# Returns:
#   tuple: the width and height in inches, as Matplotlib expects.
def figure_size(width_mm, height_mm):
    if float(height_mm) > HEIGHT_MAX_MM:  # The journal limits the height of a figure.
        # A taller figure would not fit on one page of the journal.
        raise ValueError(f"The figure height {height_mm} mm exceeds the limit of {HEIGHT_MAX_MM} mm")
    return (float(width_mm) * MM_TO_INCH, float(height_mm) * MM_TO_INCH)  # Size in inches.


# Create a figure and a grid of axes with the style of the journal.
# Arguments:
#   n_rows (int): number of rows of the panel grid.
#   n_cols (int): number of columns of the panel grid.
#   width_mm (float): figure width in millimetres.
#   height_mm (float): figure height in millimetres.
# Returns:
#   tuple: the figure and the array of axes.
def new_figure(n_rows=1, n_cols=1, width_mm=WIDTH_FULL_MM, height_mm=60.0):
    apply_style()  # Apply the journal style before the figure is created.
    size = figure_size(width_mm, height_mm)  # Figure size in inches.
    figure, axes = plt.subplots(int(n_rows), int(n_cols), figsize=size, squeeze=False)  # Grid.
    return figure, axes  # Figure and the two-dimensional array of axes.


# Add a lowercase panel label to an axis, as the journal requires.
# Arguments:
#   axis (matplotlib.axes.Axes): the axis that receives the label.
#   letter (str): the lowercase letter of the panel.
# Returns:
#   None: the label is added as a side effect.
def panel_label(axis, letter):
    # Place the lowercase letter just outside the upper left corner of the axis.
    axis.text(
        -0.16,  # Horizontal position in axis coordinates, just left of the axis.
        1.06,  # Vertical position in axis coordinates, just above the axis.
        str(letter),  # Lowercase letter that identifies the panel.
        transform=axis.transAxes,  # Interpret the coordinates in axis units.
        fontweight="bold",  # Panel labels are set in bold face.
        va="top",  # Align the top of the text with the requested position.
        ha="right",  # Align the right of the text with the requested position.
    )  # Panel label placed outside the plotting area.


# Save a figure as Encapsulated PostScript with a portable document preview.
# Arguments:
#   figure (matplotlib.figure.Figure): the figure to save.
#   stem (str): the file name without a suffix, for example "Fig2".
#   directory (str or pathlib.Path): the directory that receives the files.
# Returns:
#   pathlib.Path: the Encapsulated PostScript file that was written.
def save_figure(figure, stem, directory="results/figures"):
    location = Path(directory)  # Normalise the argument into a path object.
    location.mkdir(parents=True, exist_ok=True)  # Create the directory when it is absent.
    eps_path = location / f"{stem}.eps"  # Vector file submitted to the journal.
    pdf_path = location / f"{stem}.pdf"  # Preview used while the manuscript is drafted.
    figure.savefig(eps_path, format="eps")  # Write the Encapsulated PostScript file.
    figure.savefig(pdf_path, format="pdf")  # Write the portable document preview.
    plt.close(figure)  # Release the figure so that batch jobs do not accumulate memory.
    return eps_path  # Return the vector file for convenience.
