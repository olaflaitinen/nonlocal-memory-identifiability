# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Submit one batch job followed by a configurable number of dependent
# continuation jobs, so that a sampler chain longer than the wall time limit of
# a single job proceeds in blocks that resume from their checkpoints.
# Manuscript: Section 4.6, computational resources; see docs/roihu.md.
# Inputs: the batch file and the number of continuation jobs.
# Outputs: the identifiers of the submitted jobs on standard output.

# Stop on the first error, on an undefined variable and on a failed pipe.
set -euo pipefail
# Directory that contains this script, used to locate the shared settings.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load the shared settings of the Slurm job files.
source "${HERE}/config.sh"
# Reject an invocation that does not name a batch file.
if [ "$#" -lt 1 ]; then
  # Print the usage of the script and stop, rather than submitting nothing.
  echo "Usage: submit_chain.sh <batch_file> [number_of_continuations]" >&2
  # Signal the incorrect invocation to the caller.
  exit 2
fi
# Batch file that is submitted first and then continued.
BATCH="$1"
# Number of dependent continuation jobs, defaulting to three.
CONTINUATIONS="${2:-3}"
# Create the directory that receives one log file per array task.
mkdir -p "${LOGS}"
# Submit the first block of the chain and capture its job identifier.
JOBID="$(sbatch --parsable "${BATCH}")"
# Report the identifier of the first block of the chain.
echo "submitted ${BATCH} as job ${JOBID}"
# Submit the requested number of dependent continuation jobs.
for INDEX in $(seq 1 "${CONTINUATIONS}"); do
  # Each continuation starts only after the previous block has finished cleanly.
  JOBID="$(sbatch --parsable --dependency=afterok:"${JOBID}" "${BATCH}")"
  # Report the identifier of this continuation block.
  echo "submitted continuation ${INDEX} as job ${JOBID}"
done
# Report the identifier of the last block of the chain.
echo "chain complete, final job ${JOBID}"
