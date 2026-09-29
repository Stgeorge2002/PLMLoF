#!/bin/bash
# First-time venv + ESM2 download. 1 GH200 Superchip, billed as 0.25 NHR per wall-hour.
# Do not add --exclusive (that charges a full node = 1 NHR/hour for unused GPUs).
# Do not add --partition (workq is the default). Login nodes have no GPU.

#SBATCH --job-name=plmlof-setup
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source isambard/env.sh
bash isambard/setup.sh
