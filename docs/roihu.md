# Running on the CSC Roihu supercomputer

The experiments of Sections 5 and 6 were executed on the Roihu supercomputer of
CSC, IT Center for Science, Finland. This file records the environment build,
the directory layout, the submission of chained jobs and the monitoring.

## Before you start

Edit `PROJECT` in `slurm/config.sh` and the `--account` line of every file in
`slurm/` so that they name your billing project. Every other path in the Slurm
files is derived from that project.

## Building the environment with Tykky

A containerised environment keeps the number of files on the shared filesystem
small, which the site policy requires.

```bash
module load tykky
mkdir -p /projappl/project_XXXXXXX/nmi-env
conda-containerize new --prefix /projappl/project_XXXXXXX/nmi-env environment.yml
export PATH="/projappl/project_XXXXXXX/nmi-env/bin:$PATH"
```

Verify the environment before submitting any job.

```bash
python -c "import numpy, scipy, emcee, arviz; print('environment ready')"
```

## Directory layout

| Path | Contents |
|---|---|
| `/projappl/project_XXXXXXX/nmi-env` | the containerised Python environment |
| `/scratch/project_XXXXXXX/nonlocal-memory-identifiability` | the working copy of this repository |
| `/scratch/project_XXXXXXX/results` | raw experiment output |
| `/scratch/project_XXXXXXX/checkpoints` | sampler checkpoints |
| `/scratch/project_XXXXXXX/logs` | one log file per array element |

Scratch is not backed up and is subject to a cleaning policy, so the tracked
summaries under `results/summary/` are copied back to a permanent location and
committed once an experiment has finished.

## Submitting chained jobs

A single job may run for at most twelve hours on the `small` partition, while a
full chain of the one-dimensional sampler needs longer. Each sampler therefore
checkpoints its state and stops cleanly before the limit, and the next block
resumes from that checkpoint.

```bash
bash slurm/submit_chain.sh slurm/exp_2_4_mcmc_1d.sbatch 4
```

This submits the first block and four dependent continuation blocks, each of
which starts only after the previous block has finished cleanly.

## Monitoring

```bash
squeue --me                       # queued and running jobs of the account
seff <jobid>                      # efficiency report of a finished job
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # accounting record
tail -f /scratch/project_XXXXXXX/logs/nmi-mcmc1d-*.out  # live log of an element
```

## Billing units

Central processing unit partitions are billed by allocated cores and by memory.
The partitions used here are `small` for the production runs, `test` for short
verification runs of a few minutes, and `interactive` for exploratory work. The
memory reservation of two gigabytes per core in the Slurm files is the smallest
value at which the two-dimensional runs fit, and raising it raises the billing
proportionally. Check the remaining allocation before submitting a long chain.

```bash
csc-projects
```

Record the total consumption reported by `sacct` for the manuscript, since
Section 4.6 quotes the number of core hours used.
