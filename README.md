# PLMLoF — Protein Language Model Loss/Gain-of-Function Variant Classifier

PLMLoF is a deep learning tool for classifying bacterial gene variants as **Loss-of-Function (LoF)**, **Wildtype (WT)**, or **Gain-of-Function (GoF)** using protein language model embeddings from ESM2. It is designed to replace rule-based tools like SNPEff in bacterial genomics pipelines, providing learned variant effect predictions with attribution explanations.

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction / Inference](#prediction--inference)
- [RunPod GPU Cloud](#runpod-gpu-cloud)
- [All Commands Reference](#all-commands-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
Reference protein ──► ┌──────────────────┐
                      │   Shared ESM2    │
                      │   Encoder        │──► Comparison ──► ┌─────────────┐
Variant protein  ──►  │   (+ LoRA        │    Module         │  Classifier │──► LoF / WT / GoF
                      │    adapters)      │    diff/product   │  Head (MLP) │
                      └──────────────────┘    mean+max pool  │             │
                                                             │             │
                              12-dim engineered ────────────►│             │
                              nucleotide features            └─────────────┘
                              (frameshift, premature stop,
                               start codon loss, truncation...)
```

**Components:**

| Component | Description |
|-----------|-------------|
| **ESM2 Encoder** | Shared `facebook/esm2_t33_650M_UR50D` (650M params) embedding both ref and variant proteins |
| **Comparison Module** | Diff, product, mean+max pooling across residues → `4×D` comparison vector |
| **Nucleotide Features** | 12-dim rule-based vector: frameshift, premature stop, start codon loss, missense count, truncation fraction, etc. |
| **Classifier Head** | MLP `[512 → 128 → 3]` with dropout → LoF / WT / GoF logits |
| **LoRA Fine-tuning** | Rank-32 adapters on query/value projections of ESM2 (PEFT) |

**Two-Stage Training:**
1. **Stage 1** — ESM2 frozen, train classifier head only (fast, low memory)
2. **Stage 2** — Enable LoRA adapters on ESM2, fine-tune end-to-end (requires GPU)

**Datasets:**
- **CARD** — SNP-mediated AMR mutations → GoF labels
- **DEG** — Essential gene disruptions → LoF labels (synthetic variants)
- **ProteinGym** — Deep mutational scanning (DMS) fitness data → all 3 labels
- **Synthetic** — Programmatically generated LoF/WT/GoF variants

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training; CPU inference is supported)

### Local / WSL Install

```bash
# Clone the repository
git clone https://github.com/Stgeorge2002/PLMLoF.git
cd PLMLoF

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Dependencies

Core: `torch`, `transformers`, `peft`, `biopython`, `pysam`, `pandas`, `pyarrow`, `scikit-learn`, `captum`, `pyyaml`, `tqdm`

Dev: `pytest`, `wandb`

---

## Quick Start

### Smoke test (no GPU, no data, 2 min)

```bash
# Train a tiny model on synthetic data — tests the full pipeline on CPU
python scripts/train.py --tiny --max-epochs 2 --device cpu
```

### Predict with a trained model

```bash
# From paired FASTA files
python scripts/predict.py \
  --reference ref_genes.fasta \
  --variants var_genes.fasta \
  --model outputs/checkpoints/model_best.pt \
  --output results.tsv

# From VCF
python scripts/predict.py \
  --reference ref_genes.fasta \
  --vcf variants.vcf \
  --model outputs/checkpoints/model_best.pt \
  --output results.tsv
```

---

## Data Preparation

Download and curate training data from public databases. Scripts gracefully fall back to curated sequences if URLs are unreachable.

```bash
# 1. CARD — GoF/AMR SNP-mediated resistance mutations
python data/scripts/download_card.py

# 2. DEG — Essential genes → synthetic LoF variants
python data/scripts/download_deg.py

# 3. ProteinGym — DMS fitness data → LoF/WT/GoF
python data/scripts/download_proteingym.py

# 4. Synthetic — Programmatically generated variants
python data/scripts/generate_synthetic.py

# 5. Merge all sources + stratified 80/10/10 train/val/test split
python data/scripts/curate_dataset.py
```

Outputs written to `data/processed/`:
- `train.parquet` — Training set
- `val.parquet` — Validation set
- `test.parquet` — Test set (includes held-out species: Pseudomonas, Salmonella)
- `merged_all.parquet` — Full dataset before splitting

---

## Training

### Full production training (GPU required)

```bash
python scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --mixed-precision fp16 \
  --output-dir outputs/production/
```

### Quick local test (CPU, synthetic data)

```bash
python scripts/train.py \
  --tiny \
  --max-epochs 2 \
  --device cpu \
  --output-dir outputs/test_run/
```

### Custom hyperparameters via CLI

```bash
python scripts/train.py \
  --config configs/training.yaml \
  --model-config configs/model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --batch-size 16 \
  --lr 5e-4 \
  --max-epochs 30 \
  --mixed-precision fp16 \
  --num-workers 4 \
  --device cuda \
  --output-dir outputs/custom_run/
```

### Multi-GPU training (via accelerate)

```bash
accelerate launch \
  --num_processes 4 \
  --mixed_precision fp16 \
  scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --output-dir outputs/production/
```

**Checkpoints** are saved to `outputs/<output-dir>/checkpoints/model_best.pt` whenever validation macro-F1 improves.

---

## Evaluation

```bash
# Evaluate on held-out test set
python scripts/evaluate.py \
  --model outputs/production/checkpoints/model_best.pt \
  --test-data data/processed/test.parquet \
  --device cuda

# Evaluate tiny model on synthetic data (CPU)
python scripts/evaluate.py \
  --model outputs/test_run/checkpoints/model_best.pt \
  --tiny \
  --device cpu
```

Output includes: accuracy, macro-F1, per-class precision/recall/F1, AUROC, and a confusion matrix.

### Benchmark against SNPEff

```bash
python scripts/benchmark_vs_snpeff.py \
  --snpeff-vcf annotated.vcf \
  --reference ref.fasta \
  --model outputs/production/checkpoints/model_best.pt \
  --device cuda
```

---

## Prediction / Inference

```bash
# Paired FASTA → TSV output
python scripts/predict.py \
  --reference ref_genes.fasta \
  --variants var_genes.fasta \
  --model outputs/production/checkpoints/model_best.pt \
  --output predictions.tsv \
  --format tsv \
  --device cuda

# VCF input → JSON output
python scripts/predict.py \
  --reference ref_genes.fasta \
  --vcf variants.vcf \
  --model outputs/production/checkpoints/model_best.pt \
  --output predictions.json \
  --format json \
  --device cuda

# Skip attribution for faster inference
python scripts/predict.py \
  --reference ref_genes.fasta \
  --variants var_genes.fasta \
  --model outputs/production/checkpoints/model_best.pt \
  --no-attribution \
  --output predictions.tsv

# Test mode — tiny ESM2-8M model, no checkpoint needed
python scripts/predict.py \
  --reference ref_genes.fasta \
  --variants var_genes.fasta \
  --tiny
```

**Output columns (TSV):** `gene`, `prediction` (LoF/WT/GoF), `confidence`, `prob_LoF`, `prob_WT`, `prob_GoF`, `attribution_summary`

---

## RunPod GPU Cloud

### 1. Create a Pod

- **Template**: `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel` (or similar PyTorch image)
- **GPU**: A100 40GB recommended; A6000, RTX 4090, or H100 also work
- **Disk**: 50 GB+ container disk
- **Volume**: Mount `/workspace` as persistent storage (survives pod restarts)

**GPU VRAM guide:**

| GPU | VRAM | Stage 1 batch | Stage 2 batch | Precision |
|-----|------|---------------|---------------|-----------|
| RTX 4090 | 24 GB | 16 | 8 | fp16 |
| A6000 | 48 GB | 32 | 16 | fp16 |
| A100 40GB | 40 GB | 32 | 16 | fp16 |
| A100 80GB | 80 GB | 64 | 32 | bf16 |
| H100 | 80 GB | 64 | 32 | bf16 |

### 2. First-Time Pod Setup

SSH into the pod, then:

```bash
cd /workspace

# Clone the repo
git clone https://github.com/Stgeorge2002/PLMLoF.git
cd PLMLoF

# Install dependencies, pre-download ESM2 models, run smoke tests
bash runpod/setup.sh
```

`setup.sh` does:
- `pip install -r requirements.txt && pip install -e .`
- Pre-downloads ESM2-8M (31 MB) and ESM2-650M (2.6 GB) to `/workspace/.cache/`
- Runs a GPU smoke test
- Runs a model forward-pass test to verify the pipeline

### 3. Prepare Data

```bash
cd /workspace/PLMLoF

python data/scripts/download_card.py
python data/scripts/download_deg.py
python data/scripts/download_proteingym.py
python data/scripts/generate_synthetic.py
python data/scripts/curate_dataset.py
```

Or use the pipeline script:

```bash
bash runpod/run_all.sh --data-only
```

### 4. Train

```bash
# Production training (ESM2-650M, fp16, auto GPU detection)
python scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --mixed-precision fp16 \
  --output-dir outputs/production/
```

For **A100/H100**, switch to `bf16` for faster, more stable training:

```bash
python scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --mixed-precision bf16 \
  --output-dir outputs/production/
```

Or use the all-in-one pipeline:

```bash
# Full pipeline: data → train → evaluate
bash runpod/run_all.sh

# Quick test: synthetic data, ESM2-8M, ~2 min
bash runpod/run_all.sh --test

# Data preparation only
bash runpod/run_all.sh --data-only

# Training only (assumes data exists)
bash runpod/run_all.sh --train-only

# Evaluation only (assumes model checkpoint exists)
bash runpod/run_all.sh --eval-only
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
  --model outputs/production/checkpoints/model_best.pt \
  --test-data data/processed/test.parquet \
  --device cuda
```

### 6. Download Your Model

From your local machine:

```bash
# rsync from pod (replace with your pod IP/SSH alias)
rsync -avz root@<pod-ip>:/workspace/PLMLoF/outputs/production/checkpoints/ ./checkpoints/

# Or use RunPod's file manager in the web UI
```

### 7. Returning to an Existing Pod

After a pod restart, models are still cached in `/workspace/.cache/`:

```bash
cd /workspace/PLMLoF
git pull  # get any code updates

# Re-activate environment (the pip install persists in container disk)
# If needed: pip install -e . --quiet

# Continue training / run inference
python scripts/predict.py \
  --reference ref.fasta \
  --variants vars.fasta \
  --model outputs/production/checkpoints/model_best.pt \
  --device cuda
```

---

## All Commands Reference

### Setup

```bash
# Local install
pip install -e ".[dev]"

# RunPod one-time setup
bash runpod/setup.sh

# Pre-download ESM2 model weights only
bash runpod/download_models.sh          # ESM2-8M + 650M
bash runpod/download_models.sh --all    # All ESM2 sizes
```

### Data

```bash
python data/scripts/download_card.py          # CARD GoF variants
python data/scripts/download_deg.py           # DEG LoF variants
python data/scripts/download_proteingym.py    # ProteinGym DMS data
python data/scripts/generate_synthetic.py     # Synthetic variants
python data/scripts/curate_dataset.py         # Merge + split
```

### Training

```bash
# Tiny / CPU test
python scripts/train.py --tiny --max-epochs 2 --device cpu

# GPU production
python scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --mixed-precision fp16 \
  --output-dir outputs/production/

# bf16 (A100/H100)
python scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --mixed-precision bf16 \
  --output-dir outputs/production/

# Multi-GPU with accelerate
accelerate launch --num_processes 4 --mixed_precision fp16 \
  scripts/train.py \
  --config configs/runpod_training.yaml \
  --model-config configs/runpod_model.yaml \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --output-dir outputs/production/
```

### Evaluation

```bash
# Full test set evaluation
python scripts/evaluate.py \
  --model outputs/production/checkpoints/model_best.pt \
  --test-data data/processed/test.parquet \
  --device cuda

# CPU / tiny
python scripts/evaluate.py \
  --model outputs/test_run/checkpoints/model_best.pt \
  --tiny --device cpu

# SNPEff benchmark
python scripts/benchmark_vs_snpeff.py \
  --snpeff-vcf annotated.vcf \
  --reference ref.fasta \
  --model outputs/production/checkpoints/model_best.pt
```

### Prediction

```bash
# FASTA pairs → TSV
python scripts/predict.py \
  --reference ref.fasta \
  --variants vars.fasta \
  --model outputs/production/checkpoints/model_best.pt \
  --output results.tsv \
  --device cuda

# VCF → JSON
python scripts/predict.py \
  --reference ref.fasta \
  --vcf variants.vcf \
  --model outputs/production/checkpoints/model_best.pt \
  --output results.json \
  --format json \
  --device cuda

# Fast (no attribution)
python scripts/predict.py \
  --reference ref.fasta \
  --variants vars.fasta \
  --model outputs/production/checkpoints/model_best.pt \
  --no-attribution \
  --output results.tsv
```

### Pipeline (RunPod)

```bash
bash runpod/run_all.sh               # Full: data → train → eval
bash runpod/run_all.sh --test        # Quick test (~2 min, CPU or GPU)
bash runpod/run_all.sh --data-only   # Data preparation only
bash runpod/run_all.sh --train-only  # Training only
bash runpod/run_all.sh --eval-only   # Evaluation only
```

### Tests

```bash
pytest tests/ -v                      # All tests (CPU only)
pytest tests/test_model.py -v         # Model architecture tests
pytest tests/test_dataset.py -v       # Dataset / collator tests
pytest tests/test_features.py -v      # Nucleotide feature tests
pytest tests/test_predictor.py -v     # Inference pipeline tests
pytest tests/test_attribution.py -v   # Attribution tests
```

---

## Project Structure

```
PLMLoF/
├── configs/
│   ├── model.yaml              # Base model config (ESM2-650M, LoRA rank 16)
│   ├── training.yaml           # Base training config
│   ├── inference.yaml          # Inference config (batch size, caching)
│   ├── runpod_model.yaml       # RunPod model config (LoRA rank 32, larger head)
│   └── runpod_training.yaml    # RunPod training config (batch 32, fp16, 30 epochs)
├── plmlof/
│   ├── __init__.py             # LABEL_MAP, LABEL_TO_ID
│   ├── models/
│   │   ├── esm2_encoder.py     # ESM2 encoder with optional LoRA
│   │   ├── comparison.py       # Reference vs variant comparison module
│   │   ├── classifier.py       # MLP classifier head
│   │   └── plmlof_model.py     # Full model (encoder + comparison + classifier)
│   ├── data/
│   │   ├── dataset.py          # PLMLoFDataset, SyntheticPLMLoFDataset
│   │   ├── collator.py         # PLMLoFCollator (ESM2 tokenization + padding)
│   │   ├── features.py         # 12-dim nucleotide feature extraction
│   │   └── preprocessing.py    # DNA mutation application utilities
│   ├── training/
│   │   ├── trainer.py          # Two-stage trainer with AMP + gradient accumulation
│   │   └── metrics.py          # macro-F1, AUROC, confusion matrix
│   ├── inference/
│   │   ├── predictor.py        # High-level prediction interface
│   │   ├── vcf_handler.py      # VCF parsing, FASTA pair parsing
│   │   ├── reference_cache.py  # Pre-computed reference embedding cache
│   │   └── attribution.py      # Rule-based + integrated gradients attribution
│   └── utils/
│       └── sequence_utils.py   # translate_dna, find_mutations, is_frameshift, etc.
├── data/scripts/
│   ├── download_card.py        # CARD GoF AMR variants
│   ├── download_deg.py         # DEG essential genes → synthetic LoF
│   ├── download_proteingym.py  # ProteinGym DMS bacterial assays
│   ├── generate_synthetic.py   # Programmatic variant generation
│   └── curate_dataset.py       # Merge + stratified split
├── scripts/
│   ├── train.py                # Training CLI
│   ├── predict.py              # Inference CLI
│   ├── evaluate.py             # Evaluation CLI
│   └── benchmark_vs_snpeff.py  # SNPEff comparison
├── runpod/
│   ├── setup.sh                # One-time RunPod setup
│   ├── run_all.sh              # Full pipeline script
│   ├── download_models.sh      # ESM2 weight pre-caching
│   ├── env.sh                  # Cache path environment variables
│   └── README.md               # RunPod-specific guide
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_features.py
│   ├── test_predictor.py
│   └── test_attribution.py
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Configuration

### Model config (`configs/model.yaml` / `configs/runpod_model.yaml`)

```yaml
model:
  esm2_model_name: "facebook/esm2_t33_650M_UR50D"
  lora:
    enabled: true
    rank: 32
    alpha: 64
    dropout: 0.1
    target_modules: ["query", "value"]
  comparison:
    pool_strategy: "mean_max"   # mean or mean_max
  classifier:
    hidden_dims: [512, 128]
    dropout: 0.3
    label_smoothing: 0.05
```

### Training config (`configs/runpod_training.yaml`)

```yaml
training:
  stage1:
    max_epochs: 30
    learning_rate: 5.0e-4
    batch_size: 32
    gradient_accumulation_steps: 2
  stage2:
    max_epochs: 15
    learning_rate: 5.0e-5
    batch_size: 16
    gradient_accumulation_steps: 4
  early_stopping_patience: 7
  mixed_precision: "fp16"   # fp16 / bf16 / no
  seed: 42
```

For **base configs** (less aggressive), use `configs/model.yaml` + `configs/training.yaml`.

---

## Testing

All tests are CPU-compatible and use the tiny ESM2-8M model (~31 MB). No GPU or real data required.

```bash
# Run full test suite
pytest tests/ -v

# Run specific file
pytest tests/test_features.py -v
pytest tests/test_model.py -v
```

Expected: all tests pass in ~60 seconds on CPU.

---

## Troubleshooting

### `AttributeError: has no attribute 'total_mem'`
PyTorch uses `total_memory`. Fixed in current version.

### `BackendUnavailable: Cannot import 'setuptools.backends._legacy'`
Old setuptools on some RunPod images. Fixed in current `pyproject.toml` (uses `setuptools.build_meta`).

### Out of Memory (OOM)
- Reduce batch size in config or via `--batch-size 4`
- Ensure gradient accumulation is set: `gradient_accumulation_steps: 4`
- Lower precision: try fp16 if using bf16, or vice versa
- ESM2-650M Stage 2 requires ~18 GB VRAM minimum

### pip appears frozen during setup
Torch is ~2 GB. It will look frozen for 5–10 minutes. Wait — use `pip install -v ...` to see progress.

### CARD / DEG / ProteinGym download returns 0 records
External database URLs change periodically. The scripts fall back to curated sequences automatically. Run the scripts and check the log output for `CARD entries: N total, M with ref protein` to diagnose.

### CUDA errors
```bash
# Check CUDA/PyTorch compatibility
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Enable debug mode
CUDA_LAUNCH_BLOCKING=1 python scripts/train.py ...
```

### Slow training
- Check `num-workers`: GPU nodes should use `--num-workers 4` or higher
- Ensure data is on local SSD, not a network mount
- Check GPU utilisation: `watch -n 1 nvidia-smi`

---

## License

MIT
