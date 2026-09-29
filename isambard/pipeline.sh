#!/usr/bin/env bash
# PLMLoF pipeline body for Isambard-AI. Invoked by sbatch, never from a login node.
#
#   bash isambard/pipeline.sh
#   bash isambard/pipeline.sh --test
#   bash isambard/pipeline.sh --quick
#   bash isambard/pipeline.sh --scale 60
#   bash isambard/pipeline.sh --data-only
#   bash isambard/pipeline.sh --train-only
#   bash isambard/pipeline.sh --eval-only

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

if [[ ! -f plmlof/data/dataset.py || ! -f data/scripts/download_proteingym.py ]]; then
    echo "ERROR: plmlof/data or data/scripts is missing from this clone." >&2
    echo "       Both must be copied to the cluster (they were previously hidden by a" >&2
    echo "       blanket data/ gitignore). rsync the full tree or add those paths to git." >&2
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
SCALE=0
S1_EPOCHS=""
S2_EPOCHS=""
SMOKE=false

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case $arg in
        --smoke)       MODE="test"; SMOKE=true ;;
        --test)        MODE="test" ;;
        --quick)       SCALE=30; S1_EPOCHS=20; S2_EPOCHS=2 ;;
        --scale)       i=$((i+1)); SCALE="${ARGS[$i]}" ;;
        --s1-epochs)   i=$((i+1)); S1_EPOCHS="${ARGS[$i]}" ;;
        --s2-epochs)   i=$((i+1)); S2_EPOCHS="${ARGS[$i]}" ;;
        --data-only)   MODE="data" ;;
        --train-only)  MODE="train" ;;
        --eval-only)   MODE="eval" ;;
        --help|-h)
            echo "Usage: bash isambard/pipeline.sh [--smoke|--test|--quick|--scale N|--s1-epochs N|--s2-epochs N|--data-only|--train-only|--eval-only]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
    i=$((i+1))
done

DATA_DIR="$PLMLOF_DATA_DIR"
EMB_DIR="$PLMLOF_EMB_DIR"
OUTPUT_DIR="$PLMLOF_OUTPUT_DIR"
CHECKPOINT="$OUTPUT_DIR/checkpoints/model_best.pt"
TRAIN_CFG="$PLMLOF_TRAIN_CFG"
MODEL_CFG="$PLMLOF_MODEL_CFG"

echo "=============================================="
if [[ "$MODE" == "test" ]]; then
    echo " PLMLoF Pipeline — Mode: $MODE (ESM2-8M, synthetic; no ProteinGym, no 650M)"
else
    echo " PLMLoF Pipeline — Mode: $MODE | Scale: $([[ "$SCALE" -eq 0 ]] && echo 'ALL Prokaryote variants' || echo "${SCALE}K samples")"
fi
[[ -n "$S1_EPOCHS" ]] && echo " Stage 1 epochs: $S1_EPOCHS"
[[ -n "$S2_EPOCHS" ]] && echo " Stage 2 epochs: $S2_EPOCHS"
echo " Data:       $DATA_DIR"
echo " Embeddings: $EMB_DIR"
echo " Output:     $OUTPUT_DIR"
echo "=============================================="
echo ""

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This job is not on a GH200 compute node." >&2
    exit 1
fi
DEVICE="cuda"
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}')")
echo "GPU: $GPU_NAME (${GPU_MEM} GB)"
echo ""

# Hopper (GH200, sm_90) and Ampere both have native bf16.
PRECISION="bf16"
if python -c "import torch; cap = torch.cuda.get_device_capability(); raise SystemExit(0 if cap >= (8, 0) else 1)" 2>/dev/null; then
    echo "  Compute cap ≥ 8.0 — mixed precision $PRECISION"
else
    PRECISION="fp16"
    echo "  Compute cap < 8.0 — mixed precision $PRECISION"
fi
echo ""

if [[ "$MODE" == "full" || "$MODE" == "data" || "$MODE" == "test" ]]; then
    echo "──────── Step 1: Data Preparation ────────"
    if [[ "$MODE" == "test" ]]; then
        echo "Test mode: synthetic data generated inline by train.py --tiny"
    else
        echo "Downloading ProteinGym data..."
        mkdir -p data/raw/proteingym data/processed
        python data/scripts/download_proteingym.py

        TOTAL_SAMPLES=$(( SCALE * 1000 ))
        if [[ "$SCALE" -eq 0 ]]; then
            echo "Curating dataset (ALL Prokaryote LoF/WT/GoF variants, no cap)..."
            python data/scripts/curate_dataset.py --total-samples 0
        else
            echo "Curating dataset (${SCALE}K balanced = ${TOTAL_SAMPLES} samples)..."
            python data/scripts/curate_dataset.py --total-samples "$TOTAL_SAMPLES"
        fi

        mkdir -p "$DATA_DIR"
        for f in "$DATA_DIR"/{train,val,test}.parquet; do
            if [[ -f "$f" ]]; then
                ROWS=$(python -c "import pandas as pd; print(len(pd.read_parquet('$f')))")
                echo "  $(basename "$f"): $ROWS rows"
            fi
        done
    fi
    echo ""
fi

if [[ "$MODE" == "data" ]]; then
    echo "Data-only mode complete."
    exit 0
fi

if [[ "$MODE" == "full" || "$MODE" == "test" ]]; then
    echo "──────── Step 2: Precompute Embeddings ────────"
    if [[ "$MODE" == "test" ]]; then
        TEST_EPOCHS=2
        [[ "$SMOKE" == true ]] && TEST_EPOCHS=1
        echo "Running tiny train (ESM2-8M, synthetic, ${TEST_EPOCHS} epoch(s))..."
        python scripts/train.py \
            --tiny \
            --max-epochs "$TEST_EPOCHS" \
            --device "$DEVICE" \
            --output-dir "${PLMLOF_OUTPUT_DIR%/production}/test_run/"
    else
        if [[ -f "$EMB_DIR/train_embeddings.pt" && -f "$EMB_DIR/val_embeddings.pt" && \
              -f "$DATA_DIR/train.parquet" && \
              "$EMB_DIR/train_embeddings.pt" -nt "$DATA_DIR/train.parquet" ]]; then
            echo "Embeddings already up-to-date, skipping precompute."
        else
            mkdir -p "$EMB_DIR"
            # GH200 120 GB: 128 is safe for ESM2-650M at max_seq_length 1024.
            # 256 OOM'd on 48 GB A40 for long sequences; raise only after a successful 128 run.
            python scripts/precompute_embeddings.py \
                --train-data "$DATA_DIR/train.parquet" \
                --val-data "$DATA_DIR/val.parquet" \
                --test-data "$DATA_DIR/test.parquet" \
                --output-dir "$EMB_DIR" \
                --device "$DEVICE" \
                --batch-size 128 \
                --num-workers 8
        fi
        echo "Embeddings: $(du -sh "$EMB_DIR" 2>/dev/null | cut -f1)"
    fi
    echo ""
fi

if [[ "$MODE" == "full" || "$MODE" == "train" || "$MODE" == "test" ]]; then
    echo "──────── Step 3: Training ────────"
    if [[ "$MODE" == "test" ]]; then
        echo "Test training already done in step 2."
    else
        echo "Training with cached embeddings..."
        S1_EPOCH_FLAG=""
        [[ -n "$S1_EPOCHS" ]] && S1_EPOCH_FLAG="--max-epochs $S1_EPOCHS"
        python scripts/train.py \
            --config "$TRAIN_CFG" \
            --model-config "$MODEL_CFG" \
            --precomputed "$EMB_DIR" \
            --device "$DEVICE" \
            --mixed-precision "$PRECISION" \
            --output-dir "$OUTPUT_DIR" \
            --num-workers 8 \
            $S1_EPOCH_FLAG

        echo "──────── Step 3a: Stage 1 Evaluation (pre-LoRA baseline) ────────"
        if [[ -f "$CHECKPOINT" ]]; then
            S1_CHECKPOINT="$OUTPUT_DIR/checkpoints/model_stage1.pt"
            cp "$CHECKPOINT" "$S1_CHECKPOINT"
            echo "  Saved Stage 1 checkpoint → $S1_CHECKPOINT"

            echo "  Evaluating Stage 1 model on held-out test set..."
            python scripts/evaluate.py \
                --model "$S1_CHECKPOINT" \
                --test-data "$DATA_DIR/test.parquet" \
                --embeddings "$EMB_DIR/test_embeddings.pt" \
                --device "$DEVICE"

            for SPECIES_TAG in ecoli myctu stau klepn strpn; do
                SPECIES_PARQUET="$DATA_DIR/test_${SPECIES_TAG}.parquet"
                SPECIES_EMB="$EMB_DIR/test_${SPECIES_TAG}_embeddings.pt"
                if [[ -f "$SPECIES_PARQUET" ]]; then
                    echo "  Evaluating Stage 1 — ${SPECIES_TAG}..."
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
                --num-workers 8 \
                $S2_EPOCH_FLAG
        else
            echo "  No Stage 1 checkpoint at $CHECKPOINT — skipping Stage 2."
        fi
    fi
    echo ""
fi

if [[ "$MODE" == "train" ]]; then
    MODE="eval_after_train"
fi

if [[ "$MODE" == "full" || "$MODE" == "eval" || "$MODE" == "eval_after_train" || "$MODE" == "test" ]]; then
    echo "──────── Step 4: Evaluation (final best model) ────────"
    if [[ "$MODE" == "test" ]]; then
        CHECKPOINT="${PLMLOF_OUTPUT_DIR%/production}/test_run/checkpoints/model_best.pt"
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

            for SPECIES_TAG in ecoli myctu stau klepn strpn; do
                SPECIES_PARQUET="$DATA_DIR/test_${SPECIES_TAG}.parquet"
                SPECIES_EMB="$EMB_DIR/test_${SPECIES_TAG}_embeddings.pt"
                if [[ -f "$SPECIES_PARQUET" ]]; then
                    echo "Evaluating final model — ${SPECIES_TAG}..."
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
echo " Log:          $LOG_FILE"
echo " Checkpoint:   $CHECKPOINT"
echo " Predict:      python scripts/predict.py --model $CHECKPOINT --reference <ref.fasta> --variants <var.fasta> --device $DEVICE"
echo "=============================================="
