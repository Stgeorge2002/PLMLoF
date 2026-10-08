#!/bin/bash
# Batch-score every gene produced by scripts/build_gene_fasta_pairs.py with the
# frozen best-models/ LoF + MLoF heads, loading ESM2 once for the whole run.
# 1 GPU-hour = 0.25 NHR. Do not add --exclusive.
#
# Inputs must already be on this filesystem (rsync from a laptop, not git —
# neither N44_ST258_results/plmlof_input/ nor best-models/*.pt are in git):
#   - plmlof_input/ is a large, regenerable tree of small files (per-gene
#     ref/var FASTA pairs) -> belongs on $SCRATCHDIR ("working space ...
#     designed for active job I/O"; 60-day purge is fine, it's cheap to rebuild).
#   - best-models/ is small and shared across the project -> belongs on
#     $PROJECTDIR (persistent, not purged).
#   - predictions.tsv (the actual result) is written back to $PROJECTDIR so
#     it isn't at risk from the scratch 60-day inactivity purge.
#
#   rsync -avP plmlof_input.tar.gz HOST:$SCRATCHDIR/plmlof/N44_ST258_results/
#   rsync -avP best-models/        HOST:$PROJECTDIR/$USER/PLMLoF/best-models/
#
#   bash isambard/submit.sh predict
#   bash isambard/submit.sh predict /custom/input_dir /custom/model_dir /custom/out.tsv

#SBATCH --job-name=plmlof-predict
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: predict.sbatch must run under Slurm on a compute node." >&2
    echo "  From the repo root:  bash isambard/submit.sh predict" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"
cd "$PLMLOF_ROOT"

if [[ ! -x "${PLMLOF_VENV}/bin/python" ]]; then
    echo "ERROR: venv missing at $PLMLOF_VENV" >&2
    echo "  Submit setup first:  bash isambard/submit.sh setup" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${PLMLOF_VENV}/bin/activate"

INPUT_DIR="${1:-${PLMLOF_SCRATCH}/N44_ST258_results/plmlof_input}"
MODEL_DIR="${2:-${PLMLOF_ROOT}/best-models}"
OUT="${3:-${PLMLOF_ROOT}/N44_ST258_results/predictions.tsv}"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: $INPUT_DIR missing. rsync a tar of N44_ST258_results/plmlof_input/ to \$SCRATCHDIR/plmlof/N44_ST258_results/ and extract it first." >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/lof/ensemble.json" ]]; then
    echo "ERROR: ${MODEL_DIR}/lof/ensemble.json missing. rsync best-models/ from the laptop first (weights are gitignored)." >&2
    exit 1
fi

mkdir -p "$PLMLOF_LOG_DIR" "$(dirname "$OUT")"
LOG_FILE="$PLMLOF_LOG_DIR/predict_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"
echo "Job: $SLURM_JOB_ID  node=$(hostname)  gpus=${SLURM_GPUS:-$SLURM_GPUS_ON_NODE}"
echo "Input:  $INPUT_DIR"
echo "Model:  $MODEL_DIR"
echo "Output: $OUT"

python scripts/run_predict_batch.py \
    --input-dir "$INPUT_DIR" \
    --model "$MODEL_DIR" \
    --output "$OUT" \
    --device cuda

echo "Done. $OUT"
