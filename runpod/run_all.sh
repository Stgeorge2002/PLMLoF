#!/usr/bin/env bash
# Full PLMLoF pipeline — setup, data, precompute embeddings, train, evaluate.
#
# Usage:
#   bash runpod/run_all.sh              # Full pipeline (setup → data → embed → train → eval)
#   bash runpod/run_all.sh --test       # Quick smoke test (ESM2-8M, synthetic, 2 epochs)
#   bash runpod/run_all.sh --quick      # Fast validation run (30K samples, 20 S1 epochs, 2 S2 epochs)
#   bash runpod/run_all.sh --scale 60   # Custom scale: 60K samples (default 300 for full)
#   bash runpod/run_all.sh --s1-epochs 30 --s2-epochs 3  # Override epoch caps
#   bash runpod/run_all.sh --data-only  # Download + curate data only
#   bash runpod/run_all.sh --train-only # Train + eval only (assumes embeddings exist)
#   bash runpod/run_all.sh --eval-only  # Evaluate only (assumes checkpoint exists)
#   bash runpod/run_all.sh --skip-setup # Skip pip install + model download

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Auto-activate PyTorch env if present (AWS Deep Learning AMI) and python not already on PATH
if ! command -v python &>/dev/null && [[ -f /opt/pytorch/bin/activate ]]; then
    source /opt/pytorch/bin/activate
fi

# ── Logging setup ──
LOG_DIR="$PROJECT_DIR/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# ── Parse arguments ──
MODE="full"
SKIP_SETUP=false
# Scale controls how many thousand training samples to use.
# Default 300 (full 300K balanced dataset). --quick sets 30. --scale N overrides.
SCALE=300
S1_EPOCHS=""   # empty = use config default
S2_EPOCHS=""   # empty = use config default

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case $arg in
        --test)        MODE="test" ;;
        --quick)       SCALE=30; S1_EPOCHS=20; S2_EPOCHS=2 ;;
        --scale)       i=$((i+1)); SCALE="${ARGS[$i]}" ;;
        --s1-epochs)   i=$((i+1)); S1_EPOCHS="${ARGS[$i]}" ;;
        --s2-epochs)   i=$((i+1)); S2_EPOCHS="${ARGS[$i]}" ;;
        --data-only)   MODE="data" ;;
        --train-only)  MODE="train" ;;
        --eval-only)   MODE="eval" ;;
        --skip-setup)  SKIP_SETUP=true ;;
        --help|-h)
            echo "Usage: bash runpod/run_all.sh [--test|--quick|--scale N|--s1-epochs N|--s2-epochs N|--data-only|--train-only|--eval-only] [--skip-setup]"
            echo ""
            echo "  --quick          30K samples, 20 Stage-1 epochs, 2 Stage-2 epochs (~1–2 hrs)"
            echo "  --scale N        Use N*1000 samples total (default: 300 = full 300K)"
            echo "  --s1-epochs N    Override Stage 1 max epochs (default: from config)"
            echo "  --s2-epochs N    Override Stage 2 max epochs (default: from config)"
            exit 0
            ;;
    esac
    i=$((i+1))
done

# ── Env setup ──
# Default cache to $HOME/.cache so it works on both RunPod (/workspace) and EC2 (~)
DEFAULT_CACHE_DIR="${WORKSPACE_DIR:-$HOME}/.cache"
export HF_HOME="${HF_HOME:-$DEFAULT_CACHE_DIR/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$DEFAULT_CACHE_DIR/torch}"
# Reduce CUDA memory fragmentation (helps with variable-length protein sequences)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Paths
DATA_DIR="data/processed"
EMB_DIR="data/embeddings"
OUTPUT_DIR="outputs/production"
CHECKPOINT="$OUTPUT_DIR/checkpoints/model_best.pt"
TRAIN_CFG="configs/runpod_training.yaml"
MODEL_CFG="configs/runpod_model.yaml"

echo "=============================================="
echo " PLMLoF Pipeline — Mode: $MODE | Scale: ${SCALE}K samples"
[[ -n "$S1_EPOCHS" ]] && echo " Stage 1 epochs: $S1_EPOCHS"
[[ -n "$S2_EPOCHS" ]] && echo " Stage 2 epochs: $S2_EPOCHS"
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
        echo "Test mode: synthetic data generated inline by train.py --tiny"
    else
        echo "Downloading ProteinGym data..."
        python data/scripts/download_proteingym.py || echo "  ProteinGym download failed, continuing..."

        TOTAL_SAMPLES=$(( SCALE * 1000 ))
        echo "Curating dataset (${SCALE}K balanced = ${TOTAL_SAMPLES} samples)..."
        python data/scripts/curate_dataset.py --total-samples "$TOTAL_SAMPLES"

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
                --test-data "$DATA_DIR/test.parquet" \
                --output-dir "$EMB_DIR" \
                --device "$DEVICE" \
                --batch-size 64   # Conservative for A40 — long sequences (>800 aa) OOM at 256
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
        # Use bf16 on A100/H100 (larger dynamic range, no overflow spikes)
        # Use fp16 on older GPUs (V100, A40, T4)
        PRECISION="fp16"
        if python -c "import torch; cap = torch.cuda.get_device_capability(); exit(0 if cap >= (8,0) else 1)" 2>/dev/null; then
            PRECISION="bf16"
            echo "  Ampere+ GPU detected (compute cap ≥ 8.0) — using bf16"
        fi
        S1_EPOCH_FLAG=""
        [[ -n "$S1_EPOCHS" ]] && S1_EPOCH_FLAG="--max-epochs $S1_EPOCHS"
        python scripts/train.py \
            --config "$TRAIN_CFG" \
            --model-config "$MODEL_CFG" \
            --precomputed "$EMB_DIR" \
            --device "$DEVICE" \
            --mixed-precision "$PRECISION" \
            --output-dir "$OUTPUT_DIR" \
            $S1_EPOCH_FLAG

        # ── Step 3a: Evaluate Stage 1 model before LoRA fine-tuning ──────────
        # Run this now so Stage 2 cannot overwrite model_best.pt before we record
        # Stage 1 baseline numbers.  Also saves a permanent Stage 1 copy.
        echo "──────── Step 3a: Stage 1 Evaluation (pre-LoRA baseline) ────────"
        if [[ -f "$CHECKPOINT" ]]; then
            # Preserve Stage 1 checkpoint so the comparison is available after Stage 2
            S1_CHECKPOINT="$OUTPUT_DIR/checkpoints/model_stage1.pt"
            cp "$CHECKPOINT" "$S1_CHECKPOINT"
            echo "  Saved Stage 1 checkpoint → $S1_CHECKPOINT"

            echo "  Evaluating Stage 1 model on held-out test set..."
            python scripts/evaluate.py \
                --model "$S1_CHECKPOINT" \
                --test-data "$DATA_DIR/test.parquet" \
                --embeddings "$EMB_DIR/test_embeddings.pt" \
                --device "$DEVICE"

            # Per-species evaluation: E. coli, M. tuberculosis, S. aureus, Klebsiella, S. pneumoniae
            for SPECIES_TAG in ecoli myctu stau klepn strpn; do
                SPECIES_PARQUET="$DATA_DIR/test_${SPECIES_TAG}.parquet"
                SPECIES_EMB="$EMB_DIR/test_${SPECIES_TAG}_embeddings.pt"
                if [[ -f "$SPECIES_PARQUET" ]]; then
                    echo "  Evaluating Stage 1 — ${SPECIES_TAG} species..."
                    python scripts/evaluate.py \
                        --model "$S1_CHECKPOINT" \
                        --test-data "$SPECIES_PARQUET" \
                        --embeddings "$SPECIES_EMB" \
                        --device "$DEVICE"
                fi
            done
        else
            echo "  No Stage 1 checkpoint at $CHECKPOINT — skipping Stage 1 evaluation."
        fi
        echo ""

        # Stage 2: LoRA fine-tuning of ESM2 encoder (requires Stage 1 checkpoint)
        echo "──────── Step 3b: Stage 2 LoRA Fine-tuning ────────"
        if [[ -f "$CHECKPOINT" ]]; then
            S2_EPOCH_FLAG=""
            [[ -n "$S2_EPOCHS" ]] && S2_EPOCH_FLAG="--s2-max-epochs $S2_EPOCHS"
            python scripts/train.py \
                --config "$TRAIN_CFG" \
                --model-config "$MODEL_CFG" \
                --train-data "$DATA_DIR/train.parquet" \
                --val-data "$DATA_DIR/val.parquet" \
                --stage2-only \
                --checkpoint "$CHECKPOINT" \
                --device "$DEVICE" \
                --mixed-precision "$PRECISION" \
                --output-dir "$OUTPUT_DIR" \
                $S2_EPOCH_FLAG
        else
            echo "  No Stage 1 checkpoint at $CHECKPOINT — skipping Stage 2."
        fi
    fi
    echo ""
fi

if [[ "$MODE" == "train" ]]; then
    # Also run eval after training
    MODE="eval_after_train"
fi

# ── STEP 4: Evaluation (final / best model) ──
if [[ "$MODE" == "full" || "$MODE" == "eval" || "$MODE" == "eval_after_train" || "$MODE" == "test" ]]; then
    echo "──────── Step 4: Evaluation (final best model) ────────"

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
            echo "Evaluating final model on held-out test set..."
            python scripts/evaluate.py \
                --model "$CHECKPOINT" \
                --test-data "$DATA_DIR/test.parquet" \
                --embeddings "$EMB_DIR/test_embeddings.pt" \
                --device "$DEVICE"

            # Per-species evaluation: E. coli, M. tuberculosis, S. aureus, Klebsiella, S. pneumoniae
            for SPECIES_TAG in ecoli myctu stau klepn strpn; do
                SPECIES_PARQUET="$DATA_DIR/test_${SPECIES_TAG}.parquet"
                SPECIES_EMB="$EMB_DIR/test_${SPECIES_TAG}_embeddings.pt"
                if [[ -f "$SPECIES_PARQUET" ]]; then
                    echo "Evaluating final model — ${SPECIES_TAG} species..."
                    python scripts/evaluate.py \
                        --model "$CHECKPOINT" \
                        --test-data "$SPECIES_PARQUET" \
                        --embeddings "$SPECIES_EMB" \
                        --device "$DEVICE"
                fi
            done
        else
            echo "No checkpoint at $CHECKPOINT. Skipping evaluation."
        fi
    fi
    echo ""
fi

echo "=============================================="
echo " Pipeline complete!"
echo ""
echo " Log:          $LOG_FILE"
echo " Checkpoint:   $CHECKPOINT"
echo " Predict:      python scripts/predict.py --model $CHECKPOINT --reference <ref.fasta> --variants <var.fasta> --device $DEVICE"
echo "==============================================" 

# ── STEP 5: Sync outputs to S3 ──
if [[ -n "${S3_BUCKET:-}" ]]; then
    echo ""
    echo "──────── Step 5: Sync to S3 ────────"

    # Validate AWS CLI is installed
    if ! command -v aws &>/dev/null; then
        echo "ERROR: S3_BUCKET is set but 'aws' CLI is not installed. Skipping S3 sync."
        echo "  Install with: pip install awscli  or  apt-get install awscli"
    else
        # Validate credentials are working before attempting sync
        if ! aws sts get-caller-identity &>/dev/null; then
            echo "ERROR: AWS credentials are invalid or not configured. Skipping S3 sync."
            echo "  Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION in runpod/env.sh"
        else
            S3_DEST="s3://${S3_BUCKET}/plmlof-runs/$(date +%Y%m%d_%H%M%S)"
            echo "Syncing outputs → $S3_DEST"
            aws s3 sync "$OUTPUT_DIR" "$S3_DEST" \
                --exclude '*.tmp' \
                --no-progress
            echo "S3 sync complete: $S3_DEST"
        fi
    fi
else
    echo ""
    echo "Tip: set S3_BUCKET=your-bucket-name in runpod/env.sh to auto-sync outputs to AWS."
fi
