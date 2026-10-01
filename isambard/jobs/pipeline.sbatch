#!/bin/bash
# Full v2 pipeline on 1 GH200. Tables must already be in data/processed/v2/
# (built on a laptop). This job does not download ProteinGym/CARD/GBFF.
# 1 GPU-hour = 0.25 NHR. Do not add --exclusive.

#SBATCH --job-name=plmlof-pipeline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source isambard/env.sh
bash isambard/pipeline.sh "$@"
