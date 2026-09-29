#!/bin/bash
# Short smoke test (ESM2-8M, synthetic data). Requires setup to have finished.

#SBATCH --job-name=plmlof-test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source isambard/env.sh
bash isambard/pipeline.sh --test
