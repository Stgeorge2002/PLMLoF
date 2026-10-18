#!/bin/bash
# Batch-score every gene produced by scripts/build_gene_fasta_pairs.py /
# scripts/build_from_lof_matrix.py with the frozen best-models/ heads,
# loading ESM2 once for the whole run.
# 1 GPU-hour = 0.25 NHR. Do not add --exclusive.
#
# Inputs must already be on this filesystem (rsync from a laptop, not git —
# neither plmlof_input/ nor best-models/*.pt are in git):
#   - plmlof_input/ is a large, regenerable tree of small files (per-gene
#     ref/var FASTA pairs) -> belongs on $SCRATCHDIR ("working space ...
#     designed for active job I/O"; 60-day purge is fine, it's cheap to rebuild).
#   - best-models/ is small and shared across the project -> belongs on
#     $PROJECTDIR (persistent, not purged).
#
#   bash isambard/submit.sh predict
#   bash isambard/submit.sh predict /custom/input_dir /custom/model_dir /custom/out.tsv
#   bash isambard/submit.sh predict /custom/input_dir /custom/model_dir /custom/out.tsv --lof-only

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

# Slurm executes a spooled copy; submit.sh sets the submission directory to the repo.
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck source=../env.sh
source isambard/env.sh
cd "$PLMLOF_ROOT"

if [[ ! -x "${PLMLOF_VENV}/bin/python" ]]; then
    echo "ERROR: venv missing at $PLMLOF_VENV" >&2
    echo "  Submit setup first:  bash isambard/submit.sh setup" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${PLMLOF_VENV}/bin/activate"

INPUT_DIR="${PLMLOF_SCRATCH}/N44_ST258_results/plmlof_input"
MODEL_DIR="${PLMLOF_ROOT}/best-models"
OUT="${PLMLOF_ROOT}/N44_ST258_results/predictions.tsv"
LOF_ONLY=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lof-only)
            LOF_ONLY=1
            shift
            ;;
        -h|--help)
            echo "Usage: predict.sbatch [INPUT_DIR] [MODEL_DIR] [OUT.tsv] [--lof-only]" >&2
            exit 0
            ;;
        -*)
            echo "ERROR: unknown flag: $1" >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then INPUT_DIR="${POSITIONAL[0]}"; fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then MODEL_DIR="${POSITIONAL[1]}"; fi
if [[ ${#POSITIONAL[@]} -ge 3 ]]; then OUT="${POSITIONAL[2]}"; fi
if [[ ${#POSITIONAL[@]} -gt 3 ]]; then
    echo "ERROR: too many positional args (expected ≤3 + optional --lof-only)" >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: $INPUT_DIR missing. rsync a tar of plmlof_input/ to scratch and extract it first." >&2
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
echo "LoF-only: $LOF_ONLY"

EXTRA_ARGS=()
if [[ "$LOF_ONLY" -eq 1 ]]; then
    EXTRA_ARGS+=(--lof-only)
fi

python scripts/run_predict_batch.py \
    --input-dir "$INPUT_DIR" \
    --model "$MODEL_DIR" \
    --output "$OUT" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "Done. $OUT"
