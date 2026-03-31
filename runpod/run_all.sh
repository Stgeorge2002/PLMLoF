#!/usr/bin/env bash
# Full PLMLoF pipeline — download data, train model, evaluate.
#
# Usage:
#   bash runpod/run_all.sh              # Full production training (ESM2-650M)
#   bash runpod/run_all.sh --test       # Quick test run (ESM2-8M, synthetic data)
#   bash runpod/run_all.sh --data-only  # Download and prepare data only
#   bash runpod/run_all.sh --train-only # Train only (assumes data exists)
#   bash runpod/run_all.sh --eval-only  # Evaluate only (assumes model exists)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Parse arguments ──
MODE="full"
for arg in "$@"; do
    case $arg in
        --test)       MODE="test" ;;
        --data-only)  MODE="data" ;;
        --train-only) MODE="train" ;;
        --eval-only)  MODE="eval" ;;
        --help|-h)
            echo "Usage: bash runpod/run_all.sh [--test|--data-only|--train-only|--eval-only]"
            exit 0
            ;;
    esac
done

# ── Env setup ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

# Auto-detect device
DEVICE="cpu"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "Detected $NUM_GPUS GPU(s), using device=cuda"
else
    echo "No GPU detected, using CPU (will be slow)"
fi

echo "=============================================="
echo " PLMLoF Pipeline — Mode: $MODE"
echo " Device: $DEVICE"
echo "=============================================="
echo ""

# ── STEP 1: Data preparation ──
if [[ "$MODE" == "full" || "$MODE" == "data" || "$MODE" == "test" ]]; then
    echo "──────── Step 1: Data Preparation ────────"

    if [[ "$MODE" == "test" ]]; then
        echo "Generating synthetic data only (test mode)..."
        python data/scripts/generate_synthetic.py
    else
        echo "Downloading CARD data (GoF variants)..."
        python data/scripts/download_card.py || echo "CARD download failed (may need manual download), continuing..."

        echo "Downloading DEG data (LoF variants)..."
        python data/scripts/download_deg.py || echo "DEG download failed (may need manual download), continuing..."

        echo "Downloading ProteinGym data..."
        python data/scripts/download_proteingym.py || echo "ProteinGym download failed, continuing..."

        echo "Generating synthetic data..."
        python data/scripts/generate_synthetic.py

        echo "Curating and merging datasets..."
        python data/scripts/curate_dataset.py
    fi

    echo "Data preparation complete."
    echo ""
fi

if [[ "$MODE" == "data" ]]; then
    echo "Data-only mode complete. Exiting."
    exit 0
fi

# ── STEP 2: Training ──
if [[ "$MODE" == "full" || "$MODE" == "train" || "$MODE" == "test" ]]; then
    echo "──────── Step 2: Training ────────"

    if [[ "$MODE" == "test" ]]; then
        echo "Running test training (ESM2-8M, 2 epochs, synthetic data)..."
        python scripts/train.py \
            --tiny \
            --max-epochs 2 \
            --device "$DEVICE" \
            --output-dir outputs/test_run/
    else
        echo "Running production training (ESM2-650M)..."

        # Use multi-GPU if available
        if [[ "$NUM_GPUS" -gt 1 ]] 2>/dev/null; then
            echo "Multi-GPU detected: using accelerate with $NUM_GPUS GPUs"
            accelerate launch \
                --num_processes "$NUM_GPUS" \
                --mixed_precision fp16 \
                scripts/train.py \
                --config configs/runpod_training.yaml \
                --model-config configs/runpod_model.yaml \
                --train-data data/processed/train.parquet \
                --val-data data/processed/val.parquet \
                --device cuda \
                --output-dir outputs/production/
        else
            python scripts/train.py \
                --config configs/runpod_training.yaml \
                --model-config configs/runpod_model.yaml \
                --train-data data/processed/train.parquet \
                --val-data data/processed/val.parquet \
                --device "$DEVICE" \
                --mixed-precision fp16 \
                --output-dir outputs/production/
        fi
    fi

    echo "Training complete."
    echo ""
fi

if [[ "$MODE" == "train" ]]; then
    echo "Train-only mode complete. Exiting."
    exit 0
fi

# ── STEP 3: Evaluation ──
if [[ "$MODE" == "full" || "$MODE" == "eval" || "$MODE" == "test" ]]; then
    echo "──────── Step 3: Evaluation ────────"

    if [[ "$MODE" == "test" ]]; then
        CHECKPOINT="outputs/test_run/checkpoints/model_best.pt"
        echo "Evaluating test model..."
        python scripts/evaluate.py \
            --model "$CHECKPOINT" \
            --tiny \
            --device "$DEVICE"
    else
        CHECKPOINT="outputs/production/checkpoints/model_best.pt"
        if [[ -f "$CHECKPOINT" ]]; then
            echo "Evaluating production model..."
            python scripts/evaluate.py \
                --model "$CHECKPOINT" \
                --test-data data/processed/test.parquet \
                --device "$DEVICE"
        else
            echo "No checkpoint found at $CHECKPOINT. Skipping evaluation."
        fi
    fi

    echo ""
fi

echo "=============================================="
echo " Pipeline complete!"
echo ""
echo " Model checkpoint:  outputs/*/checkpoints/model_best.pt"
echo " Run predictions:   python scripts/predict.py --reference <ref.fasta> --variants <var.fasta> --model <checkpoint> --device $DEVICE"
echo "=============================================="
