#!/usr/bin/env bash
# PLMLoF pipeline for Isambard-AI compute nodes.
# Training tables are prepared on a laptop (scripts/prepare_local.sh) and
# copied to data/processed/{lof,mlof,growth_gof,amr_gof}/.
# This job NEVER downloads ProteinGym/CARD/GBFF.
#
#   bash isambard/submit.sh pipeline
#   bash isambard/submit.sh pipeline --train-only
#   bash isambard/submit.sh pipeline --eval-only
#   bash isambard/submit.sh pipeline --task lof
#   bash isambard/submit.sh smoke          # ESM2-8M forward + TaskNet (no training data)

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

if [[ ! -f plmlof/model.py ]]; then
    echo "ERROR: plmlof/model.py is missing from this clone." >&2
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
SEEDS="0 1 2 3 4"
MAX_EPOCHS=""

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case $arg in
        --smoke|--test) MODE="smoke" ;;
        --train-only)  MODE="train" ;;
        --eval-only)   MODE="eval" ;;
        --embed-only)  MODE="embed" ;;
        --task)        i=$((i+1)); TASK="${ARGS[$i]}" ;;
        --seeds)       i=$((i+1)); SEEDS="${ARGS[$i]}" ;;
        --max-epochs)  i=$((i+1)); MAX_EPOCHS="${ARGS[$i]}" ;;
        --help|-h)
            echo "Usage: bash isambard/pipeline.sh [--smoke|--train-only|--eval-only|--embed-only|--task lof|mlof|growth_gof|amr_gof|all|--max-epochs N]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
    i=$((i+1))
done

DATA_DIR="${PLMLOF_DATA_DIR}"
EMB_DIR="${PLMLOF_EMB_DIR}"
OUT_DIR="${PLMLOF_OUTPUT_DIR}"
TRAIN_CFG="${PLMLOF_TRAIN_CFG}"
MODEL_CFG="${PLMLOF_MODEL_CFG}"

echo "=============================================="
echo " PLMLoF  mode=$MODE  task=$TASK"
echo " Data:       $DATA_DIR"
echo " Embeddings: $EMB_DIR"
echo " Output:     $OUT_DIR"
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

if [[ "$MODE" == "smoke" ]]; then
    echo "Smoke: ESM2-8M forward + TaskNet (no training tables, no 650M)"
    python - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer
from plmlof.model import TaskNet

device = "cuda"
name = "facebook/esm2_t6_8M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
enc = AutoModel.from_pretrained(name).to(device).eval()
batch = tok(["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERTFTSLPFF"], return_tensors="pt").to(device)
with torch.no_grad():
    hidden = enc(**batch).last_hidden_state
print(f"  ESM2-8M hidden {tuple(hidden.shape)}")
d = hidden.shape[-1]
net = TaskNet(hidden_size=d, task="lof").to(device).eval()
b = 2
mean = torch.randn(b, d, device=device)
mx = torch.randn(b, d, device=device)
nuc = torch.zeros(b, 12, device=device)
with torch.no_grad():
    score = net.forward_from_pooled(mean, mx, mean, mx, nuc)
print(f"  TaskNet LoF {tuple(score.shape)}  {score.tolist()}")
assert score.shape == (b,)
print("  smoke: OK")
PY
    echo "Smoke complete."
    exit 0
fi

require_task() {
    local t="$1"
    if [[ ! -f "$DATA_DIR/$t/train.parquet" || ! -f "$DATA_DIR/$t/val.parquet" ]]; then
        echo "ERROR: missing $DATA_DIR/$t/{train,val}.parquet" >&2
        echo "Prepare tables on a laptop, then rsync task dirs here:" >&2
        echo "  bash scripts/prepare_local.sh" >&2
        exit 1
    fi
}

TASKS=()
if [[ "$TASK" == "all" ]]; then
    for t in lof mlof growth_gof amr_gof; do
        if [[ -f "$DATA_DIR/$t/train.parquet" ]]; then
            n=$(python -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/$t/train.parquet')))")
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
    echo "ERROR: no tasks with training parquet under $DATA_DIR" >&2
    echo "This pipeline does not download data. Run on a laptop:" >&2
    echo "  bash scripts/prepare_local.sh" >&2
    echo "Then rsync data/processed/{lof,mlof,growth_gof,amr_gof} onto this clone." >&2
    exit 1
fi
require_task "lof"

if [[ "$MODE" == "full" || "$MODE" == "embed" || "$MODE" == "train" ]]; then
    echo "──────── Precompute embeddings (no downloads) ────────"
    mkdir -p "$EMB_DIR"
    python scripts/precompute.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$EMB_DIR" \
        --model-config "$MODEL_CFG" \
        --device "$DEVICE" \
        --batch-size 128 \
        --num-workers 8 \
        --tasks "${TASKS[@]}"
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
    mkdir -p "$OUT_DIR/$t"
    for seed in $SEEDS; do
        local out="$OUT_DIR/$t/seed${seed}"
        python scripts/train.py \
            --task "$t" \
            --seed "$seed" \
            --precomputed "$EMB_DIR" \
            --output-dir "$out" \
            --config "$TRAIN_CFG" \
            --model-config "$MODEL_CFG" \
            --device "$DEVICE" \
            --mixed-precision "$PRECISION" \
            --num-workers 8 \
            $EPOCH_FLAG
    done
    python -c "
import json
from pathlib import Path
task_dir = Path('$OUT_DIR/$t')
root_cfg = Path('$OUT_DIR') / 'model_config.json'
members = [str(p.relative_to(task_dir)) for p in sorted(task_dir.glob('seed*/checkpoints/model_best.pt'))]
(task_dir / 'ensemble.json').write_text(json.dumps({'task': '$t', 'members': members}, indent=2))
print('ensemble.json', members)
gal = task_dir / 'seed0' / 'train_gallery.pt'
dest = task_dir / 'train_gallery.pt'
if gal.exists() and not dest.exists():
    dest.write_bytes(gal.read_bytes())
cfg = task_dir / 'seed0' / 'model_config.json'
if cfg.exists():
    (task_dir / 'model_config.json').write_bytes(cfg.read_bytes())
    payload = json.loads(cfg.read_text())
    root = json.loads(root_cfg.read_text()) if root_cfg.exists() else {}
    root.setdefault('encoders', {})
    root['encoders'][payload.get('task', '$t')] = payload.get('esm2_model_name')
    if payload.get('task') == 'lof':
        root['esm2_model_name_lof'] = payload.get('esm2_model_name')
    else:
        root['esm2_model_name'] = payload.get('esm2_model_name')
    root_cfg.write_text(json.dumps(root, indent=2))
"
    if [[ -f "$EMB_DIR/$t/null_embeddings.pt" ]]; then
        python scripts/score_nulls.py \
            --task "$t" \
            --ensemble-dir "$OUT_DIR/$t" \
            --embeddings "$EMB_DIR/$t/null_embeddings.pt" \
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
    local emb="$EMB_DIR/$t/${split}_embeddings.pt"
    if [[ ! -f "$emb" ]]; then
        echo "No $emb — skip $t $split eval"
        return
    fi
    echo "──────── Eval $t / $split ────────"
    python scripts/evaluate.py \
        --task "$t" \
        --ensemble-dir "$OUT_DIR/$t" \
        --embeddings "$emb" \
        --device "$DEVICE" \
        --json-out "$OUT_DIR/$t/metrics_${split}.json"
}

if [[ "$MODE" == "full" || "$MODE" == "eval" || "$MODE" == "train" ]]; then
    for t in "${TASKS[@]}"; do
        eval_task "$t" test
        eval_task "$t" val
        eval_task "$t" protein_test
    done
fi

echo "=============================================="
echo " Pipeline complete"
echo " Checkpoints: $OUT_DIR"
echo " Predict: python scripts/predict.py --model $OUT_DIR --reference ref.fasta --variants var.fasta --device cuda"
echo " Dewachter: python scripts/evaluate_dewachter.py --model-dir $OUT_DIR --reference ... --variants ... --scores ..."
echo "=============================================="
