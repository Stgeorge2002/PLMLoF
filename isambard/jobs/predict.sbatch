#!/bin/bash
# Batch-score every gene produced by scripts/build_gene_fasta_pairs.py /
# scripts/build_from_lof_matrix.py with a production LoF ensemble
# (1–2 seeds, torch.compile, length-sorted unique embeds, rule short-circuit).
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
#   bash isambard/submit.sh predict ... $PLMLOF_OUTPUT_DIR/lof_8m predictions_8m.tsv
#
# Optional env overrides:
#   PLMLOF_PREDICT_BATCH=256
#   PLMLOF_PREDICT_MEMBERS=1
#   PLMLOF_PREDICT_QUANTIZE=none|int8|fp8
#   PLMLOF_PREDICT_COMPILE=1
#   PLMLOF_PREDICT_SHORT_CIRCUIT=1

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
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: predict.sbatch [INPUT_DIR] [MODEL_DIR] [OUT.tsv]" >&2
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
    echo "ERROR: too many positional args (expected ≤3)" >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: $INPUT_DIR missing. rsync a tar of plmlof_input/ to scratch and extract it first." >&2
    exit 1
fi
# Accept best-models/ (with lof/ subdir) or a distilled ensemble dir (ensemble.json at root).
if [[ -f "${MODEL_DIR}/lof/ensemble.json" ]]; then
    PREDICT_MODEL="$MODEL_DIR"
elif [[ -f "${MODEL_DIR}/ensemble.json" ]]; then
    PREDICT_MODEL="$MODEL_DIR"
else
    echo "ERROR: no ensemble.json under ${MODEL_DIR}/lof/ or ${MODEL_DIR}/." >&2
    echo "  Use best-models/ (35M) or outputs/lof_8m (distilled)." >&2
    exit 1
fi

BATCH="${PLMLOF_PREDICT_BATCH:-256}"
MEMBERS="${PLMLOF_PREDICT_MEMBERS:-1}"
QUANTIZE="${PLMLOF_PREDICT_QUANTIZE:-none}"
COMPILE="${PLMLOF_PREDICT_COMPILE:-1}"
SHORT="${PLMLOF_PREDICT_SHORT_CIRCUIT:-1}"

COMPILE_FLAG=(--compile)
if [[ "$COMPILE" == "0" ]]; then COMPILE_FLAG=(--no-compile); fi
SHORT_FLAG=(--rule-short-circuit)
if [[ "$SHORT" == "0" ]]; then SHORT_FLAG=(--no-rule-short-circuit); fi

mkdir -p "$PLMLOF_LOG_DIR" "$(dirname "$OUT")"
LOG_FILE="$PLMLOF_LOG_DIR/predict_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"
echo "Job: $SLURM_JOB_ID  node=$(hostname)  gpus=${SLURM_GPUS:-$SLURM_GPUS_ON_NODE}"
echo "Input:  $INPUT_DIR"
echo "Model:  $PREDICT_MODEL"
echo "Output: $OUT"
echo "Speed:  batch=$BATCH members=$MEMBERS quantize=$QUANTIZE compile=$COMPILE short_circuit=$SHORT"

python scripts/run_predict_batch.py \
    --input-dir "$INPUT_DIR" \
    --model "$PREDICT_MODEL" \
    --output "$OUT" \
    --device cuda \
    --batch-size "$BATCH" \
    --max-ensemble-members "$MEMBERS" \
    --quantize "$QUANTIZE" \
    "${COMPILE_FLAG[@]}" \
    "${SHORT_FLAG[@]}"

echo "Done. $OUT"
