# Experiments

Every experiment of the manuscript is listed below with its manuscript
location, its configuration file, its script, its Slurm job file, its outputs
and its planned compute. Run times marked "to be measured" are filled in once
the experiment has been executed on the named machine.

## Mapping table

| Experiment | Manuscript | Configuration | Script | Slurm | Outputs | Planned compute |
|---|---|---|---|---|---|---|
| 0.3 Checkpoint test | Section 4.6 | `configs/exp_0_3_checkpoint_test.yaml` | `experiments/exp_0_3_checkpoint_test.py` | not required | log only, plus `results/summary/*_checkpoint_test.csv` | seconds on a workstation |
| 1.1 One-dimensional convergence | Section 5.1 | `configs/exp_1_1_convergence_1d.yaml` | `experiments/exp_1_1_convergence_1d.py` | not required | Table 5, Fig. 2a,b | minutes on a workstation |
| 1.2 Steady-state agreement | Section 5.1 | `configs/exp_1_2_steady_state.yaml` | `experiments/exp_1_2_steady_state.py` | not required | Fig. 2c | tens of minutes on a workstation |
| 1.3 Two-dimensional convergence and cost | Section 5.1 | `configs/exp_1_3_convergence_2d.yaml` | `experiments/exp_1_3_convergence_2d.py` | not required | text values | tens of minutes on a workstation |
| 2.1 Structural checks | Section 5.2 | `configs/exp_2_1_structural_checks.yaml` | `experiments/exp_2_1_structural_checks.py` | not required | Fig. 3 | tens of minutes on a workstation |
| 2.2 Sampler pilot | Section 5.3 | `configs/exp_2_2_mcmc_pilot.yaml` | `experiments/exp_2_2_mcmc_pilot.py` | not required | sampler settings | hours on a workstation |
| 2.3 Profile likelihood | Section 5.3 | `configs/exp_2_3_profile_likelihood.yaml` | `experiments/exp_2_3_profile_likelihood.py` | `slurm/exp_2_3_profile_likelihood.sbatch` | Table 6, Fig. 4 | 16 array elements, 4 cores each |
| 2.4 One-dimensional MCMC, 72 conditions | Section 5.3 | `configs/exp_2_4_mcmc_1d.yaml` | `experiments/exp_2_4_mcmc_1d.py` | `slurm/exp_2_4_mcmc_1d.sbatch` | Fig. 5, Online Resource 1 | 288 array elements, 2 cores each, chained |
| 2.5 Diagnostics | Section 5.3 | `configs/exp_2_4_mcmc_1d.yaml` | `experiments/exp_2_5_diagnostics.py` | not required | rerun list | minutes on a workstation |
| 2.6 Kernel discrimination | Section 5.3 | `configs/exp_2_6_kernel_discrimination.yaml` | `experiments/exp_2_6_kernel_discrimination.py` | not required | text values | hours on a workstation |
| 3.1 Two-dimensional pilot | Section 5.4 | `configs/exp_3_1_pilot_2d.yaml` | `experiments/exp_3_1_pilot_2d.py` | not required | selected conditions and data | hours on a workstation |
| 3.2 Two-dimensional MCMC | Section 5.4 | `configs/exp_3_2_mcmc_2d.yaml` | `experiments/exp_3_2_mcmc_2d.py` | `slurm/exp_3_2_mcmc_2d.sbatch` | posterior chains | 12 array elements, 8 cores each, chained |
| 3.3 Two-dimensional analysis | Section 5.4 | `configs/exp_3_2_mcmc_2d.yaml` | `experiments/exp_3_3_analysis_2d.py` | not required | Fig. 6 | minutes on a workstation |
| 4.1 Wolf preprocessing | Section 6 | `configs/exp_4_wolf.yaml` | `experiments/exp_4_1_wolf_preprocess.py` | not required | Table 7 | tens of minutes, R required |
| 4.2 Wolf pilot | Section 6 | `configs/exp_4_wolf.yaml` | `experiments/exp_4_2_wolf_pilot.py` | not required | sampler settings | minutes on a workstation |
| 4.3 Wolf fits | Section 6 | `configs/exp_4_wolf.yaml` | `experiments/exp_4_3_wolf_fit.py` | `slurm/exp_4_3_wolf_fit.sbatch` | Table 8, Fig. 7 | 8 array elements, 8 cores each, chained |
| 4.4 Wolf summary | Section 6 | `configs/exp_4_wolf.yaml` | `experiments/exp_4_4_wolf_summary.py` | not required | text values | minutes on a workstation |
| Forward solve benchmark | Section 4.1 | any configuration | `experiments/benchmark_forward.py` | not required | cost per forward solve | minutes on a workstation |
| Fig. 1 | Section 2.2 | none | `figures/make_fig1_model_kernels.py` | not required | Fig. 1 | analytic, no experiment |

## Synthetic data

`experiments/make_synthetic_data.py` generates the reference solutions and the
noisy data of every condition at the reference resolution, so that inference at
the coarser resolution avoids the inverse crime. It must be run before
Experiments 2.1 to 2.6.

## Condition indices

The 72 transient conditions are enumerated in lexicographic order of the tuple
(kernel, perceptual range, noise level, sampling design), with numeric factors
in ascending order and textual factors in alphabetical order. The resulting
integer index is stable and is used by the `--condition-index` option of every
script and by the Slurm array files. Each condition also carries the textual
identifier `k-tophat_R-0.150_eta-0.05_s-intermediate`.

The 16 profile likelihood conditions of Table 6 are the eight settings with the
intermediate sampling design and the perceptual range 0.15, analysed once with
stationary data and once with transient data for comparison. They carry the
suffix `_d-stationary` or `_d-transient` in their identifier.

## Chained jobs

Samplers stop cleanly and write a checkpoint at least thirty minutes before the
wall time limit of the job, which is configured through `GUARD_SECONDS` in
`slurm/config.sh` and the `wall_time_limit` entry of each configuration. Chains
longer than one job are continued with

```bash
bash slurm/submit_chain.sh slurm/exp_2_4_mcmc_1d.sbatch 4
```

which submits the first block and four dependent continuation blocks.
