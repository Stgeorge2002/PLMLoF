"""Train one task seed on cached embeddings.

    python scripts/train.py --task lof --seed 0 \
        --precomputed $PLMLOF_EMB_DIR --output-dir outputs/lof/seed0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from plmlof.applicability import save_gallery
from plmlof.dataset import CachedDataset, lof_train_keep_indices
from plmlof.encoders import HIDDEN_SIZE, esm2_for_task, read_encoder_meta
from plmlof.model import TaskNet
from plmlof.trainer import Trainer

logger = logging.getLogger(__name__)


def load_yaml(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Train a LoF, MLoF, or GoF head")
    p.add_argument("--task", required=True, choices=["lof", "mlof", "growth_gof", "amr_gof"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precomputed", type=Path, required=True, help="Dir containing <task>/train_embeddings.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--mixed-precision", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=None)
    args = p.parse_args()

    train_cfg = load_yaml(args.config).get("training", {})
    model_cfg = load_yaml(args.model_config).get("model", {})
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    task_emb = Path(args.precomputed) / args.task
    esm_name = esm2_for_task(args.task, model_cfg)
    enc_meta = read_encoder_meta(task_emb)
    if enc_meta and enc_meta.get("esm2_model_name") != esm_name:
        raise SystemExit(
            f"{task_emb}/encoder.json is {enc_meta.get('esm2_model_name')}, "
            f"config wants {esm_name}. Re-run scripts/precompute.py"
        )
    train_full = CachedDataset(task_emb / "train_embeddings.pt")
    val_ds = CachedDataset(task_emb / "val_embeddings.pt")
    if args.task == "mlof":
        keep = lof_train_keep_indices(
            train_full.is_wreck, train_full.is_missense, train_full.targets,
            channels=list(train_full.channels), seed=args.seed,
        )
        train_ds = Subset(train_full, keep)
        logger.info(
            "MLoF train filter %s → %s (no sure wrecks, capped identity WT)",
            len(train_full), len(train_ds),
        )
    else:
        train_ds = train_full
    hidden = train_full.ref_mean.shape[1]
    expected = HIDDEN_SIZE.get(esm_name)
    if expected is not None and int(hidden) != int(expected):
        raise SystemExit(
            f"{args.task} embeddings are D={hidden} but {esm_name} is D={expected}. "
            "Delete that task's embedding dir and re-run precompute.py"
        )
    logger.info("%s seed=%s train=%s val=%s D=%s encoder=%s", args.task, args.seed, len(train_ds), len(val_ds), hidden, esm_name)

    batch = int(train_cfg.get("batch_size", 256))
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    net = TaskNet(
        hidden_size=hidden,
        task=args.task,
        pool_strategy=model_cfg.get("pool_strategy", "mean_max"),
        use_cross_attention=model_cfg.get("use_cross_attention", False),
        head_hidden=int(model_cfg.get("head_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    precision = args.mixed_precision or train_cfg.get("mixed_precision", "bf16")
    trainer = Trainer(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        mixed_precision=precision,
        gof_threshold=float(train_cfg.get("gof_threshold", 0.90)),
        esm2_model_name=esm_name,
        pool_strategy=model_cfg.get("pool_strategy", "mean_max"),
        seed=args.seed,
    )
    trainer.train(
        max_epochs=args.max_epochs or int(train_cfg.get("max_epochs", 20)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        patience=int(train_cfg.get("early_stopping_patience", 5)),
        grad_accum_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
    )

    if args.task == "lof":
        intact = ~train_full.is_wreck
        if not bool(intact.any()):
            intact = torch.ones(len(train_full), dtype=torch.bool)
        idx = intact.nonzero(as_tuple=False).view(-1).tolist()
        gal_mean = train_full.var_mean[intact]
    else:
        miss = train_full.is_missense
        if not bool(miss.any()):
            miss = torch.ones(len(train_full), dtype=torch.bool)
        idx = miss.nonzero(as_tuple=False).view(-1).tolist()
        gal_mean = train_full.ref_mean[miss]
    save_gallery(
        Path(args.output_dir) / "train_gallery.pt",
        gal_mean,
        [train_full.genes[i] for i in idx],
        [train_full.protein_ids[i] for i in idx],
    )
    meta = {
        "task": args.task,
        "seed": args.seed,
        "esm2_model_name": esm_name,
        "hidden_size": hidden,
    }
    (Path(args.output_dir) / "model_config.json").write_text(json.dumps(meta, indent=2))
    logger.info("Done → %s", args.output_dir)


if __name__ == "__main__":
    main()
