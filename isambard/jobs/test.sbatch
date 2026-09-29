#!/bin/bash
# Same cheap check as smoke, but assumes the venv already exists (no pip).
# 10 min cap. Does not download ProteinGym or ESM2-650M.

#SBATCH --job-name=plmlof-test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source isambard/env.sh
bash isambard/pipeline.sh --smoke
