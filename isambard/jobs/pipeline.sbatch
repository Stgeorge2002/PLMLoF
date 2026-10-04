#!/bin/bash
# Full pipeline on 1 GH200. Tables must already be in data/processed/{lof,mlof,...}
# (built on a laptop). This job does not download ProteinGym/CARD/GBFF.
# 1 GPU-hour = 0.25 NHR. Do not add --exclusive.
# --mem=0 takes the host RAM attached to this GH200 (needed for 650M site windows).

#SBATCH --job-name=plmlof-pipeline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source isambard/env.sh
bash isambard/pipeline.sh "$@"
