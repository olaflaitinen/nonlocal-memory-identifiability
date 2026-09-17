# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: One-dimensional inference, one array element per condition and chain of the seventy two conditions.
# Manuscript: Section 5.3, Experiment 2.4; Fig. 5 and Online Resource 1.
# Inputs: the configuration named below and the output of the earlier
# experiments. Outputs: raw output and checkpoints on the scratch filesystem.

# Name under which the job appears in the scheduler queue.
#SBATCH --job-name=nmi-mcmc1d
# Partition of the central processing unit nodes.
#SBATCH --partition=small
# Billing project of the allocation, edited together with slurm/config.sh.
#SBATCH --account=project_XXXXXXX
# Wall time of one block of the chain, below the limit of the partition.
#SBATCH --time=12:00:00
# One task per array element, since each element is an independent run.
#SBATCH --ntasks=1
# Cores allocated to one array element.
#SBATCH --cpus-per-task=2
# Memory reserved per allocated core.
#SBATCH --mem-per-cpu=2G
# Array of independent runs, one per condition or chain.
#SBATCH --array=0-287
# Standard output of each array element.
#SBATCH --output=logs/nmi-mcmc1d-%A_%a.out
# Standard error of each array element.
#SBATCH --error=logs/nmi-mcmc1d-%A_%a.err

# Stop on the first error, on an undefined variable and on a failed pipe.
set -euo pipefail
# Load the shared settings of the Slurm job files.
source "$(dirname "$0")/config.sh"
# Work from the repository copy on the scratch filesystem.
cd "${REPO}"
# Create the directory that receives one log file per array element.
mkdir -p logs
# Restrict every numerical library to the cores allocated to this element.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# Report the array element that this block is processing.
echo "array element ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID}"
# Run the experiment, resuming from the checkpoint of the previous block.
CONDITION=$(( SLURM_ARRAY_TASK_ID / 4 ))
# Index of the chain within the condition, between zero and three.
CHAIN=$(( SLURM_ARRAY_TASK_ID % 4 ))
# Run one chain of one condition, resuming from its checkpoint.
python experiments/exp_2_4_mcmc_1d.py \
  --config configs/exp_2_4_mcmc_1d.yaml \
  --condition-index "${CONDITION}" \
  --chain-index "${CHAIN}" \
  --resume
# Report that this block finished cleanly, so that the next block may start.
echo "array element ${SLURM_ARRAY_TASK_ID} finished"
