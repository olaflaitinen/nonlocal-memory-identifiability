# Figure style

The figures follow the artwork requirements of the Journal of Mathematical
Biology. The style is implemented in `nmi.plotting.style` and is applied
automatically by `nmi.plotting.style.new_figure`.

## Format

- Vector output in Encapsulated PostScript, written by
  `nmi.plotting.style.save_figure` with `format="eps"`. A preview in portable
  document format is written alongside it for use while the manuscript is
  drafted; only the Encapsulated PostScript file is submitted.
- Fonts are embedded as Type 42 through the Matplotlib parameters
  `ps.fonttype = 42` and `pdf.fonttype = 42`.
- Files are named `Fig1.eps` to `Fig7.eps` under `results/figures/`.

## Size

- Width 174 mm for a full width figure, or 84 mm for a single column figure.
  The two widths are available as `WIDTH_FULL_MM` and `WIDTH_COLUMN_MM`.
- Height at most 234 mm, which `figure_size` enforces by raising an error.

## Lettering

- Sans-serif lettering in Arial or Helvetica. When neither typeface is
  installed, `select_font` falls back to the metric-compatible Liberation Sans
  and issues a warning that names the substitute.
- Font size between 8 and 12 pt, applied consistently through the base size of
  `apply_style`, which defaults to 9 pt with tick and legend labels one point
  smaller.

## Lines, colour and structure

- All lines are at least 0.3 pt wide; the style sets the axis frame and the
  tick marks to 0.6 pt and the plotted lines to 1.0 pt.
- Colour is RGB, taken from the colour-blind-safe palette `PALETTE`.
- Every series is distinguished by a line style or a marker in addition to its
  colour, so that the figure remains readable when printed in grey. The line
  styles are collected in `LINE_STYLES` and the markers in `MARKERS`.
- Figures carry no title inside the artwork. Panels are labelled with lowercase
  letters placed outside the plotting area by `panel_label`.
- Legends are drawn without a frame.

## Panel content

| Figure | Panels |
|---|---|
| Fig. 1 | (a) schematic of model (1); (b) the two kernels, top-hat solid and Gaussian dashed; (c) their transforms at the torus wavenumbers, circles for the top-hat and squares for the Gaussian kernel, with a zero line |
| Fig. 2 | (a) relative error against N, circles for the top-hat and squares for the Gaussian kernel, with a dotted reference slope; (b) relative mass error; (c) transient snapshots as thin lines approaching the steady state drawn as a thick line |
| Fig. 3 | (a) log-likelihood surface over (alpha, beta) with the ridge curve dashed and the true parameters marked by a cross; (b) surface over (d, mu) at fixed gamma with the same annotations |
| Fig. 4 | profile likelihoods of d, gamma, mu and R, transient data solid and stationary data dashed, with the threshold as a dotted horizontal line |
| Fig. 5 | relative interval widths against the noise level, sampling designs as dotted, dashed and solid lines, one column per perceptual range and one row per kernel |
| Fig. 6 | two-dimensional against one-dimensional relative widths, one marker per parameter, open symbols for the top-hat and filled symbols for the Gaussian kernel, with a dashed identity line |
| Fig. 7 | (a) utilisation distribution with the fixes and the study-area boundary; (b) fitted steady state at the posterior median; (c) posterior perceptual range under both kernels; (d) differences in expected log pointwise predictive density with error bars |
