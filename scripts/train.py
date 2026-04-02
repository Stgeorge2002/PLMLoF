"""Training entry point for PLMLoF.

Usage:
    # Full training with YAML config
    python scripts/train.py --config configs/training.yaml

    # Quick smoke test on CPU with synthetic data
    python scripts/train.py --tiny --max-epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from plmlof.models.plmlof_model import PLMLoFModel
from plmlof.data.dataset import PLMLoFDataset, SyntheticPLMLoFDataset, CachedEmbeddingDataset
from plmlof.data.collator import PLMLoFCollator
from plmlof.training.trainer import PLMLoFTrainer, CachedTrainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PLMLoF model")
    parser.add_argument("--config", type=str, default=None, help="Path to training YAML config")
    parser.add_argument("--model-config", type=str, default=None, help="Path to model YAML config")
    parser.add_argument("--train-data", type=str, default=None, help="Path to training data (parquet/csv)")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation data")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/auto). Default: auto-detect")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Output directory")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tiny", action="store_true", help="Use tiny ESM2 model + synthetic data for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["fp16", "bf16", "no"], help="Mixed precision mode")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: auto)")
    parser.add_argument("--precomputed", type=str, default=None,
                        help="Path to pre-computed embeddings dir (from precompute_embeddings.py). "
                             "Trains comparison+classifier only — no ESM2 forward passes.")
    return parser.parse_args()


def load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    # Load configs
    train_cfg = load_config(args.config).get("training", {})
    model_cfg = load_config(args.model_config or "configs/model.yaml").get("model", {})

    # Resolve device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")

    output_dir = args.output_dir
    seed = args.seed or train_cfg.get("seed", 42)

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    if args.tiny:
        esm2_name = "facebook/esm2_t6_8M_UR50D"
        max_epochs_s1 = args.max_epochs or 2
        max_epochs_s2 = 1
        batch_size_s1 = args.batch_size or 4
        batch_size_s2 = args.batch_size or 4
        lr_s1 = args.lr or 1e-3
        lr_s2 = 1e-4
        grad_accum_s1 = 1
        grad_accum_s2 = 1
    else:
        esm2_name = model_cfg.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D")
        s1_cfg = train_cfg.get("stage1", {})
        s2_cfg = train_cfg.get("stage2", {})
        max_epochs_s1 = args.max_epochs or s1_cfg.get("max_epochs", 20)
        max_epochs_s2 = s2_cfg.get("max_epochs", 10)
        batch_size_s1 = args.batch_size or s1_cfg.get("batch_size", 16)
        batch_size_s2 = args.batch_size or s2_cfg.get("batch_size", 8)
        lr_s1 = args.lr or s1_cfg.get("learning_rate", 1e-3)
        lr_s2 = s2_cfg.get("learning_rate", 1e-4)
        grad_accum_s1 = s1_cfg.get("gradient_accumulation_steps", 1)
        grad_accum_s2 = s2_cfg.get("gradient_accumulation_steps", 1)

    # LoRA config
    lora_cfg = model_cfg.get("lora", {})
    lora_config = {
        "rank": lora_cfg.get("rank", 16),
        "alpha": lora_cfg.get("alpha", 32),
        "dropout": lora_cfg.get("dropout", 0.1),
        "target_modules": lora_cfg.get("target_modules", ["query", "value"]),
    } if lora_cfg.get("enabled", True) else None

    # Build model — read pool_strategy from comparison section
    pool_strategy = model_cfg.get("comparison", {}).get("pool_strategy", "mean_max")
    classifier_hidden_dims = model_cfg.get("classifier", {}).get("hidden_dims", [256, 64])
    classifier_dropout = model_cfg.get("classifier", {}).get("dropout", 0.3)

    # Resolve mixed precision early (needed by both cached and standard paths)
    mixed_precision = args.mixed_precision or train_cfg.get("mixed_precision", "no")
    if mixed_precision != "no" and device != "cuda":
        logger.warning(f"Mixed precision '{mixed_precision}' requires CUDA. Falling back to 'no'.")
        mixed_precision = "no"

    logger.info(f"Building model with ESM2: {esm2_name}")
    model = PLMLoFModel(
        esm2_model_name=esm2_name,
        freeze_esm2=True,
        lora_config=lora_config,
        pool_strategy=pool_strategy,
        classifier_hidden_dims=classifier_hidden_dims,
        classifier_dropout=classifier_dropout,
    )

    # ── Pre-computed embedding mode (fast Stage 1 only) ──────────────────
    if args.precomputed:
        emb_dir = Path(args.precomputed)
        train_cache = emb_dir / "train_embeddings.pt"
        val_cache = emb_dir / "val_embeddings.pt"

        logger.info(f"Using pre-computed embeddings from {emb_dir}")
        train_dataset = CachedEmbeddingDataset(train_cache)
        val_dataset = CachedEmbeddingDataset(val_cache) if val_cache.exists() else train_dataset
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_s1 * 4, shuffle=True,
            num_workers=4, pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size_s1 * 4, shuffle=False,
            num_workers=4, pin_memory=(device == "cuda"),
        )

        # Build lightweight model (comparison + classifier only)
        hidden_size = train_dataset.ref_mean.shape[1]
        from plmlof.models.comparison import ComparisonModule
        from plmlof.models.classifier import ClassifierHead, RegressionHead
        from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES

        comparison = ComparisonModule(hidden_size=hidden_size, pool_strategy=pool_strategy)
        classifier_input = comparison.output_size + NUM_NUCLEOTIDE_FEATURES
        classifier = ClassifierHead(
            input_size=classifier_input,
            hidden_dims=classifier_hidden_dims,
            num_classes=3,
            dropout=classifier_dropout,
        )

        # Multi-task regression head for DMS z-score prediction
        regression_weight = train_cfg.get("regression_weight", 0.1)
        regressor = RegressionHead(input_size=classifier_input)
        logger.info(f"Multi-task: classification + regression (weight={regression_weight})")

        cached_trainer = CachedTrainer(
            comparison=comparison,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
            label_smoothing=model_cfg.get("classifier", {}).get("label_smoothing", 0.05),
            mixed_precision=mixed_precision,
            regressor=regressor,
            regression_weight=regression_weight,
            regression_warmup_epochs=train_cfg.get("regression_warmup_epochs", 3),
            esm2_model_name=esm2_name,
            pool_strategy=pool_strategy,
            classifier_hidden_dims=classifier_hidden_dims,
            classifier_dropout=classifier_dropout,
            lora_config=lora_config,
        )

        logger.info("=" * 60)
        logger.info("STAGE 1 (CACHED): Training comparison + classifier head")
        logger.info("=" * 60)
        cached_trainer.train(
            max_epochs=max_epochs_s1,
            learning_rate=lr_s1,
            patience=train_cfg.get("early_stopping_patience", 5),
            grad_accum_steps=grad_accum_s1,
        )
        logger.info("Training complete!")
        logger.info(f"Best model saved to {output_dir}/checkpoints/model_best.pt")
        return

    # ── Standard mode (ESM2 forward passes each batch) ────────────────────
    # Build datasets
    if args.tiny or (args.train_data is None):
        logger.info("Using synthetic dataset for testing")
        train_dataset = SyntheticPLMLoFDataset(num_samples=60, seed=seed)
        val_dataset = SyntheticPLMLoFDataset(num_samples=20, seed=seed + 1)
    else:
        train_dataset = PLMLoFDataset(args.train_data)
        val_dataset = PLMLoFDataset(args.val_data) if args.val_data else PLMLoFDataset(args.train_data)

    collator = PLMLoFCollator(tokenizer_name=esm2_name)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 4 if device == "cuda" else 0

    def _make_loaders(batch_size: int):
        tl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collator, num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        vl = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collator, num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        return tl, vl

    train_loader, val_loader = _make_loaders(batch_size_s1)

    # Build trainer
    trainer = PLMLoFTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        label_smoothing=model_cfg.get("classifier", {}).get("label_smoothing", 0.05),
        mixed_precision=mixed_precision,
    )

    # Save model config for later reconstruction during inference/eval
    trainer.model_config = {
        "esm2_model_name": esm2_name,
        "classifier_hidden_dims": classifier_hidden_dims,
        "classifier_dropout": classifier_dropout,
        "pool_strategy": pool_strategy,
        "lora_config": lora_config,
    }

    # Stage 1: Classification head only (ESM2 frozen)
    logger.info("=" * 60)
    logger.info("STAGE 1: Training classification head (ESM2 frozen)")
    logger.info("=" * 60)
    trainer.train(
        stage=1,
        max_epochs=max_epochs_s1,
        learning_rate=lr_s1,
        patience=train_cfg.get("early_stopping_patience", 5),
        grad_accum_steps=grad_accum_s1,
    )

    # Stage 2: LoRA fine-tuning (skip in tiny mode or if no LoRA)
    if lora_config and not args.tiny:
        # Rebuild loaders with stage2 batch size if different
        if batch_size_s2 != batch_size_s1:
            train_loader, val_loader = _make_loaders(batch_size_s2)
            trainer.train_loader = train_loader
            trainer.val_loader = val_loader
        logger.info("=" * 60)
        logger.info("STAGE 2: Fine-tuning with LoRA")
        logger.info("=" * 60)
        trainer.train(
            stage=2,
            max_epochs=max_epochs_s2,
            learning_rate=lr_s2,
            patience=train_cfg.get("early_stopping_patience", 5),
            grad_accum_steps=grad_accum_s2,
        )

    logger.info("Training complete!")
    logger.info(f"Best model saved to {output_dir}/checkpoints/model_best.pt")


if __name__ == "__main__":
    main()
