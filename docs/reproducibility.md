# Reproducibility

This file describes how to reproduce every table, figure and supplementary file
of the manuscript from a clean clone. None of the experiments below has been run
yet, so the route is a specification rather than a record: only the test suite,
the pre-experiment checks and the demonstration configuration have been
executed so far.

## System requirements

- Operating system on which the test suite has been run: Linux (Ubuntu 24.04).
  No other operating system has been tried, and the Red Hat based environment of
  the CSC Roihu supercomputer is to be tested.
- Python 3.12 with the package versions recorded in `environment.yml`.
- R 4.4 with the package `ctmm`, required only by Experiment 4.1. That step has
  not yet been run.
- Hardware: a workstation with at least 8 GB of memory for the smoke
  configuration and for the one-dimensional experiments. A cluster allocation is
  planned for the one-dimensional and two-dimensional sampling experiments of
  Sections 5.3, 5.4 and 6, which have not yet been run.

## Installation

```bash
git clone https://github.com/olaflaitinen/nonlocal-memory-identifiability.git
cd nonlocal-memory-identifiability
conda env create -f environment.yml
conda activate nmi
make install
```

Expected installation time: to be measured.

Optional, required only for Experiment 4.1:

```bash
Rscript R/install_r_packages.R
```

## Checks

```bash
make policy     # character, comment and licence header policies
make lint       # ruff
make test       # pytest without the slow tests
make test-all   # pytest including the slow tests
```

Expected run time of `make test`: to be measured.
Expected run time of `make test-all`: to be measured.

## Pre-experiment checks

```bash
make precheck
```

This runs `precheck/check_proofs.py`, which verifies Propositions 1 and 3,
Lemma 1, Theorem 2 and Corollary 1; `precheck/design_params.py`, which
reproduces the values of Table 3 and verifies the stationary identity at the
resulting steady states; and `precheck/timing.py`, which measures the cost of
one forward solve at the candidate resolutions.

Expected run time: to be measured.

## Demonstration

```bash
make smoke
```

The smoke configuration runs every stage of the pipeline once on a small
design: four transient conditions, a coarse grid and short chains. It writes
untracked raw output under `results/raw/smoke/`, summaries named `smoke_*.csv`
under `results/summary/`, and the figures and tables under `results/figures/`
and `results/tables/`. The smoke outputs demonstrate that the pipeline runs end
to end; they are not the results reported in the manuscript and are not tracked
by version control.

Expected run time: to be measured.

## Full reproduction

The order below reflects the dependencies between the experiments. The
configuration files are listed in `docs/experiments.md`.

1. Generate the synthetic data of every condition.

```bash
python experiments/make_synthetic_data.py --config configs/exp_2_4_mcmc_1d.yaml
```

2. Verify the solver and the steady states.

```bash
python experiments/exp_0_3_checkpoint_test.py --config configs/exp_0_3_checkpoint_test.yaml
python experiments/exp_1_1_convergence_1d.py --config configs/exp_1_1_convergence_1d.yaml
python experiments/exp_1_2_steady_state.py --config configs/exp_1_2_steady_state.yaml
python experiments/exp_1_3_convergence_2d.py --config configs/exp_1_3_convergence_2d.yaml
```

3. Run the structural checks and the sampler pilot.

```bash
python experiments/exp_2_1_structural_checks.py --config configs/exp_2_1_structural_checks.yaml
python experiments/exp_2_2_mcmc_pilot.py --config configs/exp_2_2_mcmc_pilot.yaml
```

4. Run the profile likelihood and the one-dimensional sampling on the cluster.

```bash
bash slurm/submit_chain.sh slurm/exp_2_3_profile_likelihood.sbatch 1
bash slurm/submit_chain.sh slurm/exp_2_4_mcmc_1d.sbatch 4
```

5. Collect the diagnostics and rerun any condition that failed the criteria.

```bash
python experiments/exp_2_5_diagnostics.py --config configs/exp_2_4_mcmc_1d.yaml
```

6. Run the kernel discrimination study and the two-dimensional experiments.

```bash
python experiments/exp_2_6_kernel_discrimination.py --config configs/exp_2_6_kernel_discrimination.yaml
python experiments/exp_3_1_pilot_2d.py --config configs/exp_3_1_pilot_2d.yaml
bash slurm/submit_chain.sh slurm/exp_3_2_mcmc_2d.sbatch 4
python experiments/exp_3_3_analysis_2d.py --config configs/exp_3_2_mcmc_2d.yaml
```

7. Run the wolf application. Download the data package first, as described in
   `data/README.md`.

```bash
python experiments/exp_4_1_wolf_preprocess.py --config configs/exp_4_wolf.yaml
python experiments/exp_4_2_wolf_pilot.py --config configs/exp_4_wolf.yaml
bash slurm/submit_chain.sh slurm/exp_4_3_wolf_fit.sbatch 4
python experiments/exp_4_4_wolf_summary.py --config configs/exp_4_wolf.yaml
```

8. Build the figures, the tables and the electronic supplement.

```bash
make tables
make figures
make online-resource
```

## Expected outputs

| Output | Location |
|---|---|
| Table 3 to Table 8 | `results/tables/TableN.csv` and `results/tables/TableN.tex` |
| Fig. 1 to Fig. 7 | `results/figures/FigN.eps` with a preview `FigN.pdf` |
| Online Resource 1 | `results/tables/ESM_1.csv` |
| Tracked summaries | `results/summary/*.csv` |
| Raw output and checkpoints | `results/raw/` and `checkpoints/`, both untracked |

## Release and archiving

The manuscript cites an archived version of this code with a persistent
identifier, as the research data and code sharing policy of the publisher
requires. The archive is produced from a tagged release.

Versions 0.0.1 and 0.1.0 are development snapshots, archived before any
experiment was run, and neither produced the numerical results of the
manuscript. The version that produces those results will be released as 1.0.0
once the experiments have been run, and the manuscript will cite its identifier.
Until then the concept identifier 10.5281/zenodo.22804041 is the identifier of
the software as a whole.

1. Connect the repository to Zenodo before the release is published. Sign in at
   https://zenodo.org with the GitHub account that owns the repository, open
   the GitHub tab of the Zenodo account settings and enable the switch for
   `olaflaitinen/nonlocal-memory-identifiability`. Zenodo archives a release
   only when the switch was already enabled at the moment the release was
   published.

2. Verify the working tree before tagging. The tag must point at a commit at
   which every check passes.

```bash
make policy
make lint
make test-all
make smoke
```

3. Create an annotated tag on that commit and push it.

```bash
git tag -a v1.0.0 -m "Version 1.0.0"
git push origin v1.0.0
```

4. Publish a release for the tag, either from the releases page of the
   repository or with the GitHub command line client. The release notes
   summarise the contents of the version, as recorded under the matching
   heading of `CHANGELOG.md`.

```bash
gh release create v1.0.0 --title "v1.0.0" --notes-file release-notes.md
```

5. Zenodo mints a digital object identifier for the release within a few
   minutes. Record it in three places and commit the change:

   - the `identifiers` block of `CITATION.cff`;
   - the `related_identifiers` block of `.zenodo.json`;
   - the code availability paragraph of `README.md` and the corresponding
     statement of the manuscript.

6. The concept identifier that Zenodo assigns alongside the version identifier
   resolves to the most recent version. Once 1.0.0 is released, cite its version
   identifier in the manuscript, so that the citation names the exact code that
   produced the reported results.

## Determinism

Every condition carries a seed spawned from the global seed of its
configuration with `numpy.random.SeedSequence`, in index order, and the seed is
recorded in the output metadata. Every output file also records the git commit
of the working tree, the hash of the configuration, the versions of the
scientific libraries and the time of the run, so that a result can be traced
back to the exact code and settings that produced it.
