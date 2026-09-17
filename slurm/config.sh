# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Shared settings of the Slurm job files, sourced by every batch
# script and by the chaining helper. The author edits PROJECT once, and every
# path and default follows from it.
# Manuscript: Section 4.6, computational resources; see docs/roihu.md.
# Inputs: none. Outputs: exported shell variables used by the job files.

# Billing project of the CSC allocation, to be edited by the author.
PROJECT="project_XXXXXXX"
# Scratch directory of the project, which holds the raw experiment output.
SCRATCH="/scratch/${PROJECT}"
# Persistent application directory of the project, which holds the environment.
PROJAPPL="/projappl/${PROJECT}"
# Containerised Python environment built with Tykky, see docs/roihu.md.
TYKKY_ENV="${PROJAPPL}/nmi-env"
# Working copy of this repository on the scratch filesystem.
REPO="${SCRATCH}/nonlocal-memory-identifiability"
# Directory that receives the raw experiment output of every job.
RESULTS="${SCRATCH}/results"
# Directory that receives the sampler checkpoints of every job.
CHECKPOINTS="${SCRATCH}/checkpoints"
# Directory that receives one log file per array task.
LOGS="${SCRATCH}/logs"
# Default partition of the central processing unit nodes.
PARTITION="small"
# Default wall time of one block of a chained job.
WALLTIME="12:00:00"
# Default memory reserved per allocated core.
MEMPERCPU="2G"
# Wall time in seconds after which a sampler checkpoints and stops cleanly.
GUARD_SECONDS=39600
# Make the settings visible to the job steps started by the batch scripts.
export PROJECT SCRATCH PROJAPPL TYKKY_ENV REPO RESULTS CHECKPOINTS LOGS
# Make the scheduling defaults visible as well.
export PARTITION WALLTIME MEMPERCPU GUARD_SECONDS
# Place the containerised environment first on the executable search path.
export PATH="${TYKKY_ENV}/bin:${PATH}"
# Make the package importable without an editable installation inside the job.
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"
