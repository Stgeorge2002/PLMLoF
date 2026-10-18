#!/bin/bash
# Distill LoF ESM2-35M teacher → ESM2-8M student (1 GH200).
# Needs existing 35M teacher ensemble + teacher embeddings + lof parquet.

#SBATCH --job-name=plmlof-distill
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --time=08:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source isambard/env.sh
bash isambard/distill.sh "$@"
