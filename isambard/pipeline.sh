#!/usr/bin/env bash
# PLMLoF v2 pipeline for Isambard-AI compute nodes.
# Training tables are prepared on a laptop (scripts/prepare_v2_local.sh) and
# copied to data/processed/v2/. This job NEVER downloads ProteinGym/CARD/GBFF.
#
#   bash isambard/submit.sh pipeline
#   bash isambard/submit.sh pipeline --train-only
#   bash isambard/submit.sh pipeline --eval-only
#   bash isambard/submit.sh pipeline --task lof
#   bash isambard/submit.sh smoke          # still ESM2-8M env check (not v2)

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: pipeline.sh must run under Slurm on a compute node." >&2
    echo "  From the repo root:  bash isambard/submit.sh pipeline" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
cd "$PLMLOF_ROOT"

if [[ ! -f plmlof/v2/model.py ]]; then
    echo "ERROR: plmlof/v2 is missing from this clone." >&2
    exit 1
fi

if [[ ! -x "${PLMLOF_VENV}/bin/python" ]]; then
    echo "ERROR: venv missing at $PLMLOF_VENV" >&2
    echo "  Submit setup first:  bash isambard/submit.sh setup" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${PLMLOF_VENV}/bin/activate"

mkdir -p "$PLMLOF_LOG_DIR"
LOG_FILE="$PLMLOF_LOG_DIR/pipeline_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"
echo "Job: $SLURM_JOB_ID  node=$(hostname)  gpus=${SLURM_GPUS:-$SLURM_GPUS_ON_NODE}"

MODE="full"
TASK="all"
SMOKE=false
MAX_EPOCHS=""
SEEDS="0 1 2 3 4"

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case $arg in
        --smoke)       MODE="test"; SMOKE=true ;;
        --test)        MODE="test" ;;
        --train-only)  MODE="train" ;;
        --eval-only)   MODE="eval" ;;
        --embed-only)  MODE="embed" ;;
        --task)        i=$((i+1)); TASK="${ARGS[$i]}" ;;
        --seeds)       i=$((i+1)); SEEDS="${ARGS[$i]}" ;;
        --max-epochs)  i=$((i+1)); MAX_EPOCHS="${ARGS[$i]}" ;;
        --help|-h)
            echo "Usage: bash isambard/pipeline.sh [--smoke|--test|--train-only|--eval-only|--embed-only|--task lof|growth_gof|amr_gof|all|--max-epochs N]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
    i=$((i+1))
done

V2_DATA="${PLMLOF_V2_DATA_DIR:-$PLMLOF_ROOT/data/processed/v2}"
V2_EMB="${PLMLOF_V2_EMB_DIR:-$PLMLOF_EMB_DIR/v2}"
V2_OUT="${PLMLOF_V2_OUTPUT_DIR:-$PLMLOF_ROOT/outputs/v2}"
TRAIN_CFG="${PLMLOF_TRAIN_CFG}"
MODEL_CFG="${PLMLOF_MODEL_CFG}"

echo "=============================================="
echo " PLMLoF v2  mode=$MODE  task=$TASK"
echo " Data:       $V2_DATA"
echo " Embeddings: $V2_EMB"
echo " Output:     $V2_OUT"
echo "=============================================="

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available." >&2
    exit 1
fi
DEVICE="cuda"
echo "GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"

PRECISION="bf16"
if python -c "import torch; cap = torch.cuda.get_device_capability(); raise SystemExit(0 if cap >= (8, 0) else 1)" 2>/dev/null; then
    echo "  mixed precision $PRECISION"
else
    PRECISION="fp16"
    echo "  mixed precision $PRECISION"
fi

# ── Smoke: original tiny 3-class env check (no v2 data, no 650M) ──────────
if [[ "$MODE" == "test" ]]; then
    TEST_EPOCHS=2
    [[ "$SMOKE" == true ]] && TEST_EPOCHS=1
    echo "Smoke/tiny train (ESM2-8M, synthetic 3-class — env check only)"
    python scripts/train.py \
        --tiny \
        --max-epochs "$TEST_EPOCHS" \
        --device "$DEVICE" \
        --output-dir "${PLMLOF_OUTPUT_DIR%/production}/test_run/"
    echo "Smoke complete."
    exit 0
fi

require_task() {
    local t="$1"
    if [[ ! -f "$V2_DATA/$t/train.parquet" || ! -f "$V2_DATA/$t/val.parquet" ]]; then
        echo "ERROR: missing $V2_DATA/$t/{train,val}.parquet" >&2
        echo "Prepare tables on a laptop, then rsync data/processed/v2/ here:" >&2
        echo "  bash scripts/prepare_v2_local.sh" >&2
        exit 1
    fi
}

TASKS=()
if [[ "$TASK" == "all" ]]; then
    for t in lof growth_gof amr_gof; do
        if [[ -f "$V2_DATA/$t/train.parquet" ]]; then
            n=$(python -c "import pandas as pd; print(len(pd.read_parquet('$V2_DATA/$t/train.parquet')))")
            if [[ "$n" -gt 0 ]]; then
                TASKS+=("$t")
            else
                echo "Skipping empty task $t"
            fi
        else
            echo "Skipping missing task $t"
        fi
    done
else
    require_task "$TASK"
    TASKS=("$TASK")
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "ERROR: no v2 tasks with training parquet under $V2_DATA" >&2
    echo "This pipeline does not download data. Run on a laptop:" >&2
    echo "  bash scripts/prepare_v2_local.sh" >&2
    echo "Then rsync data/processed/v2/ onto this clone." >&2
    exit 1
fi
require_task "lof"

# ── Embed (once, all tasks) ──────────────────────────────────────────────
if [[ "$MODE" == "full" || "$MODE" == "embed" || "$MODE" == "train" ]]; then
    NEED_EMBED=0
    for t in "${TASKS[@]}"; do
        if [[ ! -f "$V2_EMB/$t/train_embeddings.pt" || ! -f "$V2_EMB/$t/val_embeddings.pt" ]]; then
            NEED_EMBED=1
        elif [[ "$V2_DATA/$t/train.parquet" -nt "$V2_EMB/$t/train_embeddings.pt" ]]; then
            NEED_EMBED=1
        fi
    done
    if [[ "$NEED_EMBED" -eq 1 ]]; then
        echo "──────── Precompute v2 embeddings (no downloads) ────────"
        mkdir -p "$V2_EMB"
        python scripts/precompute_v2.py \
            --data-dir "$V2_DATA" \
            --output-dir "$V2_EMB" \
            --device "$DEVICE" \
            --batch-size 128 \
            --num-workers 8 \
            --tasks "${TASKS[@]}"
    else
        echo "Embeddings up to date, skipping precompute."
    fi
fi

if [[ "$MODE" == "embed" ]]; then
    echo "Embed-only complete."
    exit 0
fi

EPOCH_FLAG=""
[[ -n "$MAX_EPOCHS" ]] && EPOCH_FLAG="--max-epochs $MAX_EPOCHS"

train_task() {
    local t="$1"
    echo "──────── Train $t  (seeds: $SEEDS) ────────"
    mkdir -p "$V2_OUT/$t"
    local members=()
    for seed in $SEEDS; do
        local out="$V2_OUT/$t/seed${seed}"
        python scripts/train_v2.py \
            --task "$t" \
            --seed "$seed" \
            --precomputed "$V2_EMB" \
            --output-dir "$out" \
            --config "$TRAIN_CFG" \
            --model-config "$MODEL_CFG" \
            --device "$DEVICE" \
            --mixed-precision "$PRECISION" \
            --num-workers 8 \
            $EPOCH_FLAG
        members+=("seed${seed}/checkpoints/model_best.pt")
    done
    python -c "
import json
from pathlib import Path
task_dir = Path('$V2_OUT/$t')
members = [str(p.relative_to(task_dir)) for p in sorted(task_dir.glob('seed*/checkpoints/model_best.pt'))]
(task_dir / 'ensemble.json').write_text(json.dumps({'task': '$t', 'members': members}, indent=2))
print('ensemble.json', members)
gal = task_dir / 'seed0' / 'train_gallery.pt'
dest = task_dir / 'train_gallery.pt'
if gal.exists() and not dest.exists():
    dest.write_bytes(gal.read_bytes())
cfg = task_dir / 'seed0' / 'model_config.json'
root_cfg = Path('$V2_OUT') / 'model_config.json'
if cfg.exists():
    root_cfg.write_bytes(cfg.read_bytes())
"
    if [[ -f "$V2_EMB/$t/null_embeddings.pt" ]]; then
        python scripts/score_nulls.py \
            --task "$t" \
            --ensemble-dir "$V2_OUT/$t" \
            --embeddings "$V2_EMB/$t/null_embeddings.pt" \
            --device "$DEVICE"
    else
        echo "No null embeddings for $t — empirical p will be skipped until null.parquet is embedded."
    fi
}

if [[ "$MODE" == "full" || "$MODE" == "train" ]]; then
    for t in "${TASKS[@]}"; do
        train_task "$t"
    done
fi

eval_task() {
    local t="$1"
    local split="$2"
    local emb="$V2_EMB/$t/${split}_embeddings.pt"
    if [[ ! -f "$emb" ]]; then
        echo "No $emb — skip $t $split eval"
        return
    fi
    echo "──────── Eval $t / $split ────────"
    python scripts/evaluate_v2.py \
        --task "$t" \
        --ensemble-dir "$V2_OUT/$t" \
        --embeddings "$emb" \
        --device "$DEVICE" \
        --json-out "$V2_OUT/$t/metrics_${split}.json"
}

if [[ "$MODE" == "full" || "$MODE" == "eval" || "$MODE" == "train" ]]; then
    for t in "${TASKS[@]}"; do
        eval_task "$t" test
        eval_task "$t" val
    done
fi

echo "=============================================="
echo " v2 pipeline complete"
echo " Checkpoints: $V2_OUT"
echo " Predict: python scripts/predict.py --model $V2_OUT --reference ref.fasta --variants var.fasta --device cuda"
echo " Dewachter: python scripts/evaluate_dewachter.py --model-dir $V2_OUT --reference ... --variants ... --scores ..."
echo "=============================================="
