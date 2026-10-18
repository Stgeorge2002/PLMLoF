#!/usr/bin/env bash
# Distill LoF 35M teacher → 8M student on an Isambard compute node.
#
#   bash isambard/submit.sh distill
#   bash isambard/submit.sh distill --teacher-dir best-models/lof
#
# Steps: precompute 8M embeddings (if missing) → distill → eval student test split.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: distill.sh must run under Slurm on a compute node." >&2
    echo "  From the repo root:  bash isambard/submit.sh distill" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
cd "$PLMLOF_ROOT"

if [[ ! -x "${PLMLOF_VENV}/bin/python" ]]; then
    echo "ERROR: venv missing at $PLMLOF_VENV" >&2
    echo "  Submit setup first:  bash isambard/submit.sh setup" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${PLMLOF_VENV}/bin/activate"

TEACHER_DIR="${PLMLOF_ROOT}/best-models/lof"
OUT_DIR="${PLMLOF_OUTPUT_DIR}/lof_8m"
EMB_8M="${PLMLOF_EMB_DIR_8M}"
DISTILL_ALPHA="0.85"
MAX_EPOCHS=""
SEEDS=""

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case $arg in
        --teacher-dir) i=$((i+1)); TEACHER_DIR="${ARGS[$i]}" ;;
        --output-dir)  i=$((i+1)); OUT_DIR="${ARGS[$i]}" ;;
        --student-emb) i=$((i+1)); EMB_8M="${ARGS[$i]}" ;;
        --distill-alpha) i=$((i+1)); DISTILL_ALPHA="${ARGS[$i]}" ;;
        --max-epochs)  i=$((i+1)); MAX_EPOCHS="${ARGS[$i]}" ;;
        --seeds)       i=$((i+1)); SEEDS="${ARGS[$i]}" ;;
        --help|-h)
            echo "Usage: bash isambard/distill.sh [--teacher-dir DIR] [--output-dir DIR] [--student-emb DIR] [--distill-alpha A]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
    i=$((i+1))
done

if [[ ! -d "$TEACHER_DIR" ]]; then
    # Fall back to pipeline outputs
    if [[ -d "${PLMLOF_OUTPUT_DIR}/lof" ]]; then
        TEACHER_DIR="${PLMLOF_OUTPUT_DIR}/lof"
    else
        echo "ERROR: teacher ensemble not found at $TEACHER_DIR" >&2
        echo "  Point --teacher-dir at best-models/lof or outputs/lof" >&2
        exit 1
    fi
fi

if [[ ! -f "${PLMLOF_DATA_DIR}/lof/train.parquet" ]]; then
    echo "ERROR: missing ${PLMLOF_DATA_DIR}/lof/train.parquet" >&2
    exit 1
fi

if [[ ! -f "${PLMLOF_EMB_DIR}/lof/train_embeddings.pt" ]]; then
    echo "ERROR: missing teacher embeddings ${PLMLOF_EMB_DIR}/lof/train_embeddings.pt" >&2
    echo "  Run the 35M pipeline first:  bash isambard/submit.sh pipeline" >&2
    exit 1
fi

DEVICE="cuda"
PRECISION="bf16"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available." >&2
    exit 1
fi

echo "=============================================="
echo " PLMLoF distill  35M → 8M"
echo " Teacher:  $TEACHER_DIR"
echo " Teacher emb: $PLMLOF_EMB_DIR"
echo " Student emb: $EMB_8M"
echo " Output:   $OUT_DIR"
echo "=============================================="

mkdir -p "$EMB_8M" "$OUT_DIR"

NEED_PRE=0
if [[ ! -f "$EMB_8M/lof/train_embeddings.pt" || ! -f "$EMB_8M/lof/encoder.json" ]]; then
    NEED_PRE=1
else
    got=$(python -c "import json; print(json.load(open('$EMB_8M/lof/encoder.json'))['esm2_model_name'])")
    if [[ "$got" != "facebook/esm2_t6_8M_UR50D" ]]; then
        echo "Student emb encoder is $got — rebuilding for 8M"
        NEED_PRE=1
    fi
fi

if [[ "$NEED_PRE" -eq 1 ]]; then
    echo "──────── Precompute ESM2-8M embeddings ────────"
    python scripts/precompute.py \
        --data-dir "$PLMLOF_DATA_DIR" \
        --output-dir "$EMB_8M" \
        --model-config "${PLMLOF_ROOT}/configs/model_8m.yaml" \
        --device "$DEVICE" \
        --batch-size 128 \
        --num-workers 2 \
        --tasks lof
fi

EPOCH_FLAG=()
[[ -n "$MAX_EPOCHS" ]] && EPOCH_FLAG=(--max-epochs "$MAX_EPOCHS")
SEED_FLAG=()
if [[ -n "$SEEDS" ]]; then
    # shellcheck disable=SC2206
    SEED_FLAG=(--seeds $SEEDS)
fi

echo "──────── Distill LoF head ────────"
python scripts/distill_lof.py \
    --teacher-dir "$TEACHER_DIR" \
    --teacher-emb "$PLMLOF_EMB_DIR" \
    --student-emb "$EMB_8M" \
    --output-dir "$OUT_DIR" \
    --config "$PLMLOF_TRAIN_CFG" \
    --model-config "${PLMLOF_ROOT}/configs/model_8m.yaml" \
    --device "$DEVICE" \
    --mixed-precision "$PRECISION" \
    --distill-alpha "$DISTILL_ALPHA" \
    "${EPOCH_FLAG[@]}" \
    "${SEED_FLAG[@]}"

if [[ -f "$EMB_8M/lof/test_embeddings.pt" ]]; then
    echo "──────── Eval student / test ────────"
    python scripts/evaluate.py \
        --task lof \
        --ensemble-dir "$OUT_DIR" \
        --embeddings "$EMB_8M/lof/test_embeddings.pt" \
        --device "$DEVICE" \
        --json-out "$OUT_DIR/metrics_test.json"
fi

echo "=============================================="
echo " Distill complete → $OUT_DIR"
echo " Fast predict: python scripts/predict.py --model $OUT_DIR --reference ref.fasta --variants var.fasta --device cuda"
echo " Optional: rsync/cp $OUT_DIR → best-models/lof_8m"
echo "=============================================="
