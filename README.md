# nonlocal-memory-identifiability

Code accompanying the manuscript

> Laitinen Imanov OY. *Structural and practical identifiability of perception and memory in nonlocal advection-diffusion models of animal movement.* In preparation for the Journal of Mathematical Biology.

The repository contains the forward solvers, likelihoods, profile likelihood and Markov chain Monte Carlo (MCMC) code, experiment scripts and summary results for the paper. The model is a nonlocal advection-diffusion equation for population density coupled to a cognitive map, with top-hat or Gaussian detection kernels.

**Status:** under development. Results in this repository are not final until the manuscript is submitted.

## Repository structure

| Folder | Contents |
|---|---|
| `src/` | Forward solvers (pseudo-spectral, fixed-point steady state, finite volume), likelihoods and inference code |
| `precheck/` | Numerical checks of the structural identifiability results and the pilot study used to choose the parameter design |
| `experiments/` | Experiment scripts and Slurm job files for the CSC Roihu supercomputer |
| `results/` | Summary outputs used for tables and figures (large raw outputs are not tracked) |
| `data/` | Local data directory, **not tracked** (see Data below) |

## Installation

Local installation with conda:

```bash
conda env create -f environment.yml
conda activate nmi
```

On CSC Roihu, build a containerised environment with Tykky (replace `project_XXXXXXX` with your project):

```bash
module load tykky
mkdir -p /projappl/project_XXXXXXX/nmi-env
conda-containerize new --prefix /projappl/project_XXXXXXX/nmi-env environment.yml
export PATH="/projappl/project_XXXXXXX/nmi-env/bin:$PATH"
```

## Pre-experiment checks

```bash
python precheck/check_proofs.py                          # Propositions 1 and 3, Lemma 1, Theorem 2, Corollary 1
python precheck/design_params.py 1.2 0.075,0.15,0.30     # Theorem 1 and parameter design (Table 3)
python precheck/timing.py                                # forward-solve cost at candidate resolutions
```

## Data

The wolf GPS data analysed in the paper were provided by the Natural Resources Institute Finland (Luke) and are **not included** in this repository, because precise locations of a protected large carnivore are sensitive. They are available from Luke on reasonable request and subject to its data sharing conditions. All synthetic data can be regenerated with the code in this repository.

## Citation

If you use this code, please cite the paper (reference will be added on publication) and the archived software version (Zenodo DOI will be added on release). Citation metadata are provided in `CITATION.cff`.

## License

MIT License, see `LICENSE`.
