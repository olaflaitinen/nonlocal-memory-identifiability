# nonlocal-memory-identifiability

Research code accompanying the manuscript

> Laitinen Imanov OY. *Structural and practical identifiability of perception
> and memory in nonlocal advection-diffusion models of animal movement.* In
> preparation for the Journal of Mathematical Biology, topical collection
> "Nonlocal PDE Models of Cognitive Movement: Mathematical Analysis and
> Biological Applications".

## Summary

Animals move using perception and spatial memory, and nonlocal
advection-diffusion equations have become a standard description of such
cognition-driven movement. This repository implements a single-species model in
which a population advects up the gradient of a spatially averaged cognitive
map, sampled through a top-hat or Gaussian detection kernel with perceptual
range R, together with the numerical and statistical machinery needed to decide
which of its parameters different kinds of observation can determine. It
provides pseudo-spectral and positivity-preserving finite-volume solvers, the
fixed-point characterisation of the stationary states, the reconstruction of
the parameters from two Fourier modes, profile likelihood and Bayesian
inference on synthetic data across noise levels, sampling designs, perceptual
ranges and kernel shapes, and an application to publicly archived wolf Global
Positioning System location data.

**Status:** under development. The numerical results in this repository are not
final until the manuscript is submitted.

## System requirements

- Operating systems tested: Linux (Ubuntu 24.04) and the Red Hat based
  environment of the CSC Roihu supercomputer.
- Python 3.12, with the exact package versions pinned in `environment.yml`.
- R 4.4 with the package `ctmm`, required only by Experiment 4.1.
- Hardware: a workstation with at least 8 GB of memory runs the demonstration
  and the one-dimensional experiments; the full sampling experiments of
  Sections 5.3, 5.4 and 6 were run on a cluster allocation.

## Installation

```bash
git clone https://github.com/olaflaitinen/nonlocal-memory-identifiability.git
cd nonlocal-memory-identifiability
conda env create -f environment.yml
conda activate nmi
make install
```

Expected installation time: to be measured.

On the CSC Roihu supercomputer, build a containerised environment with Tykky as
described in `docs/roihu.md`.

## Demonstration

```bash
make smoke
```

The demonstration runs every stage of the pipeline once on a deliberately small
configuration: four transient conditions, a coarse grid and short chains. It
writes summaries named `smoke_*.csv` under `results/summary/`, figures under
`results/figures/` and tables under `results/tables/`. These outputs show that
the pipeline runs end to end; they are not the results reported in the
manuscript and are not tracked by version control.

Expected run time: to be measured.

## Full reproduction

`docs/reproducibility.md` gives the step-by-step route from a clean clone to
every table, figure and the electronic supplement, and `docs/experiments.md`
maps each experiment to its manuscript location, configuration, script, Slurm
job file, outputs and planned compute.

## Repository structure

| Path | Contents |
|---|---|
| `src/nmi/` | forward solvers, likelihoods, priors, samplers, diagnostics and the figure style |
| `src/nmi/wolf/` | schema validation, study areas, masked solver and model comparison for Section 6 |
| `R/` | autocorrelated kernel density estimation with `ctmm`, used by Experiment 4.1 |
| `configs/` | one configuration file per experiment, plus the shared design and the smoke settings |
| `experiments/` | one script per experiment, with a command line interface and resume support |
| `slurm/` | job files and the chaining helper for the CSC Roihu supercomputer |
| `figures/` | one script per figure of the manuscript |
| `tables/` | one script per numerical table, plus the electronic supplement |
| `precheck/` | numerical checks of the structural results and of the parameter design |
| `tests/` | the test suite, with expensive tests marked `slow` |
| `tools/` | the character, comment and licence header policy checkers |
| `docs/` | model equations, experiment map, reproduction route, cluster notes and standards |
| `results/summary/` | tracked comma separated summaries read by the figure and table scripts |
| `data/` | data documentation and checksums; the data files themselves are not tracked |

## Data

The empirical application of Section 6 uses the publicly archived data package

> Latham ADM, Boutin S (2019) Data from: Wolf ecology and caribou-primary
> prey-wolf spatial relationships in low productivity peatland complexes in
> northeastern Alberta. Movebank Data Repository.
> https://doi.org/10.5441/001/1.7vr1k987

released under the Creative Commons Zero (CC0 1.0) public domain dedication.
The raw files are not committed. Download them from the digital object
identifier above, place them under `data/raw/` and verify them against
`data/checksums.sha256`; `data/README.md` gives the full instructions, the
schema and the known properties of the verified files. All synthetic data can
be regenerated with the code in this repository.

## Code availability and archiving

The code is available at
https://github.com/olaflaitinen/nonlocal-memory-identifiability under the
Mozilla Public License 2.0. The version used for the article is archived at
Zenodo, which mints two persistent identifiers:

| Identifier | Resolves to |
|---|---|
| https://doi.org/10.5281/zenodo.22804930 | version 0.1.0, the code that produced the reported results |
| https://doi.org/10.5281/zenodo.22804041 | the software as a whole, always the most recent archived version |

Both are recorded in `CITATION.cff` and the concept identifier is recorded in
`.zenodo.json`. `docs/reproducibility.md` describes how a later version is
tagged, released and archived.

## Citation

Please cite the article and the archived software version. Cite the version
identifier 10.5281/zenodo.22804930 rather than the concept identifier, so that
the citation names the exact code that produced the reported results. Citation
metadata are provided in `CITATION.cff`.

## Licence

Mozilla Public License 2.0, see `LICENSE`.
