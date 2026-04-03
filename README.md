# PLMLoF — Protein Language Model Loss/Gain-of-Function Variant Classifier

ESM2-based classifier for bacterial gene variants: **Loss-of-Function (LoF)**, **Wildtype (WT)**, or **Gain-of-Function (GoF)**.

---

## Table of Contents

- [RunPod Quick Start](#runpod-quick-start)
- [Architecture](#architecture)
- [Local Install](#local-install)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## RunPod Quick Start

### 1. First-Time Setup

```bash
cd /workspace
git clone https://github.com/Stgeorge2002/PLMLoF.git
cd PLMLoF
bash runpod/run_all.sh
bash runpod/setup.sh
```

`setup.sh` installs dependencies, pre-downloads ESM2-8M and ESM2-650M (~2.6 GB) to `/workspace/.cache/`, and runs smoke tests.

### 2. Prepare Data (ProteinGym only)

```bash
python data/scripts/download_proteingym.py
python data/scripts/curate_dataset.py
```

Outputs: `data/processed/train.parquet`, `val.parquet`, `test.parquet` (~50K balanced samples, equal thirds LoF/WT/GoF).

### 3. Pre-compute Embeddings (run ESM2 once)

This runs ESM2-650M over all data once and saves pooled embeddings to disk. Training then becomes MLP-only — minutes instead of hours.

> **Takes ~40 min on A40.** Use `nohup` so it survives SSH disconnections. Reconnect and run `tail -f precompute.log` to check progress.

```bash
nohup python scripts/precompute_embeddings.py \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --output-dir data/embeddings/ \
  --batch-size 128 \
  --device cuda > precompute.log 2>&1 &

tail -f precompute.log
```

Outputs: `data/embeddings/train_embeddings.pt` and `val_embeddings.pt`.

### 4. Train (fast cached mode)

```bash
python scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --precomputed data/embeddings/ \
  --mixed-precision fp16 \
  --output-dir outputs/production/
```

Checkpoint saved to `outputs/production/checkpoints/model_best.pt` on best validation macro-F1.

For A100/H100, use `--mixed-precision bf16`.

### 5. Evaluate

```bash
python scripts/evaluate.py \
  --model outputs/production/checkpoints/model_best.pt \
  --test-data data/processed/test.parquet \
  --device cuda
```

### 6. Predict

```bash
# Paired FASTA → TSV
python scripts/predict.py \
  --reference ref_genes.fasta \
  --variants var_genes.fasta \
  --model outputs/production/checkpoints/model_best.pt \
  --output predictions.tsv \
  --device cuda

# VCF input
python scripts/predict.py \
  --reference ref_genes.fasta \
  --vcf variants.vcf \
  --model outputs/production/checkpoints/model_best.pt \
  --output predictions.json \
  --format json \
  --device cuda
```

### 7. Returning to an Existing Pod

```bash
cd /workspace/PLMLoF
git pull
python scripts/train.py --config configs/runpod_training.yaml --model-config configs/runpod_model.yaml \
  --precomputed data/embeddings/ --output-dir outputs/production/
```

---

## Architecture

```
Reference protein ──► ┌──────────────────┐
                      │   Shared ESM2    │
                      │   Encoder        │──► Comparison ──► ┌─────────────┐
Variant protein  ──►  │   (+ LoRA        │    Module         │  Classifier │──► LoF / WT / GoF
                      │    adapters)      │    diff/product   │  Head (MLP) │
                      └──────────────────┘    mean+max pool  │             │
                                                             │  Regression │──► DMS z-score
                              12-dim engineered ────────────►│  Head (MLP) │    (multi-task)
                              features                       └─────────────┘
                              (length change, premature stop,
                               met-start lost, missense density,
                               truncation, region, sequence identity...)
```

**Pre-compute workflow:** ESM2 runs once to save pooled `[mean, max]` embeddings per sequence. The `CachedTrainer` then trains only the Comparison projection + Classifier MLP + Regression MLP using those cached tensors — no ESM2 forward pass per epoch.

**Multi-task training:** Classification loss (cross-entropy) + regression loss (Huber/SmoothL1 on DMS fitness z-scores) are jointly optimised with linear warmup. The regression head predicts the continuous fitness effect; weight is configurable via `regression_weight` in the training config.

---

## Local Install

```bash
git clone https://github.com/Stgeorge2002/PLMLoF.git
cd PLMLoF
pip install -e ".[dev]"
```

Requires Python 3.10+, PyTorch 2.0+, CUDA 11.8+ (CPU inference supported).

---

## Testing

All tests are CPU-compatible and use the tiny ESM2-8M model. No GPU or real data required.

```bash
pytest tests/ -v
```

---

## Troubleshooting

**Out of Memory (OOM)**
- Reduce `--batch-size`
- ESM2-650M Stage 2 (full training) requires ~18 GB VRAM minimum
- For the cached workflow (`--precomputed`), any GPU with 4 GB+ is sufficient

**pip appears frozen during setup**
Torch is ~2 GB. It will look frozen for 5–10 minutes. Wait.

**ProteinGym download returns 0 records**
External URLs change periodically — check the script's log output for the count.

**CUDA errors**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
CUDA_LAUNCH_BLOCKING=1 python scripts/train.py ...
```

**Slow training**
Use the `--precomputed` flag to run ESM2 once and cache embeddings. Training becomes MLP-only (minutes, not hours).
