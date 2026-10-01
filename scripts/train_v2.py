"""Train one v2 task seed on cached embeddings.

    python scripts/train_v2.py --task lof --seed 0 \
        --precomputed $PLMLOF_EMB_DIR/v2 --output-dir outputs/v2/lof/seed0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from plmlof.v2.applicability import save_gallery
from plmlof.v2.dataset import V2CachedDataset
from plmlof.v2.model import V2TaskNet
from plmlof.v2.trainer import V2Trainer

logger = logging.getLogger(__name__)


def load_yaml(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Train a v2 LoF or GoF head")
    p.add_argument("--task", required=True, choices=["lof", "growth_gof", "amr_gof"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precomputed", type=Path, required=True, help="Dir containing <task>/train_embeddings.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/v2_training.yaml")
    p.add_argument("--model-config", default="configs/v2_model.yaml")
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
    train_ds = V2CachedDataset(task_emb / "train_embeddings.pt")
    val_ds = V2CachedDataset(task_emb / "val_embeddings.pt")
    hidden = train_ds.ref_mean.shape[1]
    logger.info("%s seed=%s train=%s val=%s D=%s", args.task, args.seed, len(train_ds), len(val_ds), hidden)

    batch = int(train_cfg.get("batch_size", 256))
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    net = V2TaskNet(
        hidden_size=hidden,
        task=args.task,
        pool_strategy=model_cfg.get("pool_strategy", "mean_max"),
        use_cross_attention=model_cfg.get("use_cross_attention", False),
        head_hidden=int(model_cfg.get("head_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    precision = args.mixed_precision or train_cfg.get("mixed_precision", "bf16")
    trainer = V2Trainer(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        mixed_precision=precision,
        gof_threshold=float(train_cfg.get("gof_threshold", 0.90)),
        esm2_model_name=model_cfg.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D"),
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

    miss = train_ds.is_missense
    if not bool(miss.any()):
        miss = torch.ones(len(train_ds), dtype=torch.bool)
    idx = miss.nonzero(as_tuple=False).view(-1).tolist()
    save_gallery(
        Path(args.output_dir) / "train_gallery.pt",
        train_ds.ref_mean[miss],
        [train_ds.genes[i] for i in idx],
        [train_ds.protein_ids[i] for i in idx],
    )
    meta = {
        "task": args.task,
        "seed": args.seed,
        "esm2_model_name": model_cfg.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D"),
        "hidden_size": hidden,
    }
    (Path(args.output_dir) / "model_config.json").write_text(json.dumps(meta, indent=2))
    logger.info("Done → %s", args.output_dir)


if __name__ == "__main__":
    main()
