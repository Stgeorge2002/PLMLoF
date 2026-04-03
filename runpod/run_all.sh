#!/usr/bin/env bash
# Full PLMLoF pipeline — setup, data, precompute embeddings, train, evaluate.
#
# Usage:
#   bash runpod/run_all.sh              # Full pipeline (setup → data → embed → train → eval)
#   bash runpod/run_all.sh --test       # Quick smoke test (ESM2-8M, synthetic, 2 epochs)
#   bash runpod/run_all.sh --data-only  # Download + curate data only
#   bash runpod/run_all.sh --train-only # Train + eval only (assumes embeddings exist)
#   bash runpod/run_all.sh --eval-only  # Evaluate only (assumes checkpoint exists)
#   bash runpod/run_all.sh --skip-setup # Skip pip install + model download

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Parse arguments ──
MODE="full"
SKIP_SETUP=false
for arg in "$@"; do
    case $arg in
        --test)        MODE="test" ;;
        --data-only)   MODE="data" ;;
        --train-only)  MODE="train" ;;
        --eval-only)   MODE="eval" ;;
        --skip-setup)  SKIP_SETUP=true ;;
        --help|-h)
            echo "Usage: bash runpod/run_all.sh [--test|--data-only|--train-only|--eval-only] [--skip-setup]"
            exit 0
            ;;
    esac
done

# ── Env setup ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

# Paths
DATA_DIR="data/processed"
EMB_DIR="data/embeddings"
OUTPUT_DIR="outputs/production"
CHECKPOINT="$OUTPUT_DIR/checkpoints/model_best.pt"
TRAIN_CFG="configs/runpod_training.yaml"
MODEL_CFG="configs/runpod_model.yaml"

echo "=============================================="
echo " PLMLoF Pipeline — Mode: $MODE"
echo "=============================================="
echo ""

# ── STEP 0: Setup (install deps + download ESM2) ──
if [[ "$SKIP_SETUP" == false && ("$MODE" == "full" || "$MODE" == "test") ]]; then
    echo "──────── Step 0: Setup ────────"
    pip install -q -r requirements.txt
    pip install -q -e ".[dev]"

    echo "Pre-downloading ESM2 weights..."
    if [[ "$MODE" == "test" ]]; then
        bash runpod/download_models.sh --tiny
    else
        bash runpod/download_models.sh
    fi
    echo "Setup complete."
    echo ""
fi

# Auto-detect device (after install so torch is available)
DEVICE="cpu"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}')")
    echo "GPU: $GPU_NAME (${GPU_MEM} GB) — device=cuda"
else
    echo "No GPU detected, using CPU (will be slow)"
fi
echo ""

# ── STEP 1: Data preparation ──
if [[ "$MODE" == "full" || "$MODE" == "data" || "$MODE" == "test" ]]; then
    echo "──────── Step 1: Data Preparation ────────"

    if [[ "$MODE" == "test" ]]; then
        echo "Generating synthetic data (test mode)..."
        python data/scripts/generate_synthetic.py
    else
        echo "Downloading ProteinGym data..."
        python data/scripts/download_proteingym.py || echo "  ProteinGym download failed, continuing..."

        echo "Curating dataset (150K balanced)..."
        python data/scripts/curate_dataset.py

        echo "Data files:"
        wc -l "$DATA_DIR"/*.parquet 2>/dev/null || true
        for f in "$DATA_DIR"/{train,val,test}.parquet; do
            if [[ -f "$f" ]]; then
                ROWS=$(python -c "import pandas as pd; print(len(pd.read_parquet('$f')))")
                echo "  $(basename $f): $ROWS rows"
            fi
        done
    fi
    echo ""
fi

if [[ "$MODE" == "data" ]]; then
    echo "Data-only mode complete."
    exit 0
fi

# ── STEP 2: Precompute ESM2 embeddings ──
if [[ "$MODE" == "full" || "$MODE" == "test" ]]; then
    echo "──────── Step 2: Precompute Embeddings ────────"

    if [[ "$MODE" == "test" ]]; then
        echo "Running test precompute (ESM2-8M, synthetic)..."
        python scripts/train.py \
            --tiny \
            --max-epochs 2 \
            --device "$DEVICE" \
            --output-dir outputs/test_run/
        # Test mode skips precompute — uses tiny inline training
    else
        # Skip if embeddings already exist and are newer than the data
        if [[ -f "$EMB_DIR/train_embeddings.pt" && -f "$EMB_DIR/val_embeddings.pt" && \
              "$EMB_DIR/train_embeddings.pt" -nt "$DATA_DIR/train.parquet" ]]; then
            echo "Embeddings already up-to-date, skipping precompute."
        else
            mkdir -p "$EMB_DIR"
            python scripts/precompute_embeddings.py \
                --train-data "$DATA_DIR/train.parquet" \
                --val-data "$DATA_DIR/val.parquet" \
                --output-dir "$EMB_DIR" \
                --device "$DEVICE" \
                --batch-size 128
        fi
        echo "Embeddings: $(du -sh "$EMB_DIR" 2>/dev/null | cut -f1)"
    fi
    echo ""
fi

# ── STEP 3: Training (cached — comparison + classifier only) ──
if [[ "$MODE" == "full" || "$MODE" == "train" || "$MODE" == "test" ]]; then
    echo "──────── Step 3: Training ────────"

    if [[ "$MODE" == "test" ]]; then
        # Already trained inline in step 2 for test mode
        echo "Test training already done in step 2."
    else
        echo "Training with cached embeddings (CE loss, cross-attn, LayerNorm)..."
        python scripts/train.py \
            --config "$TRAIN_CFG" \
            --model-config "$MODEL_CFG" \
            --precomputed "$EMB_DIR" \
            --device "$DEVICE" \
            --mixed-precision fp16 \
            --output-dir "$OUTPUT_DIR"
    fi
    echo ""
fi

if [[ "$MODE" == "train" ]]; then
    # Also run eval after training
    MODE="eval_after_train"
fi

# ── STEP 4: Evaluation ──
if [[ "$MODE" == "full" || "$MODE" == "eval" || "$MODE" == "eval_after_train" || "$MODE" == "test" ]]; then
    echo "──────── Step 4: Evaluation ────────"

    if [[ "$MODE" == "test" ]]; then
        CHECKPOINT="outputs/test_run/checkpoints/model_best.pt"
        if [[ -f "$CHECKPOINT" ]]; then
            python scripts/evaluate.py \
                --model "$CHECKPOINT" \
                --tiny \
                --device "$DEVICE"
        else
            echo "No test checkpoint found. Skipping."
        fi
    else
        if [[ -f "$CHECKPOINT" ]]; then
            echo "Evaluating on held-out test set..."
            python scripts/evaluate.py \
                --model "$CHECKPOINT" \
                --test-data "$DATA_DIR/test.parquet" \
                --device "$DEVICE"
        else
            echo "No checkpoint at $CHECKPOINT. Skipping evaluation."
        fi
    fi
    echo ""
fi

echo "=============================================="
echo " Pipeline complete!"
echo ""
echo " Checkpoint:   $CHECKPOINT"
echo " Predict:      python scripts/predict.py --model $CHECKPOINT --reference <ref.fasta> --variants <var.fasta> --device $DEVICE"
echo "==============================================" 

# ── STEP 5: Sync outputs to S3 ──
if [[ -n "${S3_BUCKET:-}" ]]; then
    echo ""
    echo "──────── Step 5: Sync to S3 ────────"
    S3_DEST="s3://${S3_BUCKET}/plmlof-runs/$(date +%Y%m%d_%H%M%S)"
    echo "Syncing outputs → $S3_DEST"
    aws s3 sync "$OUTPUT_DIR" "$S3_DEST" \
        --exclude '*.tmp' \
        --no-progress
    echo "S3 sync complete: $S3_DEST"
else
    echo ""
    echo "Tip: set S3_BUCKET=your-bucket-name in runpod/env.sh to auto-sync outputs to AWS."
fi
