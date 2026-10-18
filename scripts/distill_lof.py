"""Distill comparative LoF from ESM2-35M teacher → ESM2-8M student.

Predict-time cost is dominated by the frozen ESM2 forward. This trains a new
TaskNet on 8M pooled embeddings to match the teacher's scores on the same
parquet rows (soft-label distillation), optionally mixed with hard labels.

Prereqs
-------
1. Teacher ensemble, e.g. ``best-models/lof`` or ``outputs/lof``
2. Teacher (35M) embeddings under ``--teacher-emb/lof/``
3. Student (8M) embeddings under ``--student-emb/lof/``::

       python scripts/precompute.py \\
         --data-dir data/processed \\
         --output-dir $PLMLOF_EMB_DIR_8M \\
         --model-config configs/model_8m.yaml \\
         --device cuda --tasks lof

Usage
-----
    python scripts/distill_lof.py \\
      --teacher-dir best-models/lof \\
      --teacher-emb $PLMLOF_EMB_DIR \\
      --student-emb $PLMLOF_EMB_DIR_8M \\
      --output-dir outputs/lof_8m \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from plmlof.applicability import save_gallery
from plmlof.dataset import CachedDataset, SoftTargetDataset
from plmlof.encoders import ESM2_LOF_FAST, HIDDEN_SIZE, esm2_for_task, read_encoder_meta
from plmlof.model import TaskNet
from plmlof.predictor import _load_net, discover_ensemble
from plmlof.trainer import Trainer

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _require_encoder(emb_task_dir: Path, wanted: str, *, label: str) -> None:
    meta = read_encoder_meta(emb_task_dir)
    if not meta:
        raise SystemExit(f"{label}: missing {emb_task_dir}/encoder.json — run precompute.py")
    got = meta.get("esm2_model_name")
    if got != wanted:
        raise SystemExit(
            f"{label}: embeddings are {got}, expected {wanted}. "
            f"Re-run precompute into a separate emb root with the matching --model-config."
        )


def _assert_row_aligned(teacher: CachedDataset, student: CachedDataset, *, split: str) -> None:
    if len(teacher) != len(student):
        raise SystemExit(
            f"{split}: teacher rows={len(teacher)} student rows={len(student)}. "
            "Both must be built from the same parquet via precompute.py."
        )
    if not teacher.protein_ids or not student.protein_ids:
        return
    for i in (0, len(teacher) // 2, len(teacher) - 1):
        if teacher.protein_ids[i] != student.protein_ids[i] or teacher.genes[i] != student.genes[i]:
            raise SystemExit(
                f"{split}: row {i} misaligned "
                f"(teacher {teacher.genes[i]}/{teacher.protein_ids[i]} vs "
                f"student {student.genes[i]}/{student.protein_ids[i]})."
            )


@torch.no_grad()
def score_soft_targets(
    nets: list[TaskNet],
    ds: CachedDataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Mean ensemble score in [0, 1] for every row."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    parts: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="teacher soft labels", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        seed_scores = []
        for net in nets:
            raw = net.forward_from_cache(
                batch["ref_mean"], batch["ref_max"],
                batch["var_mean"], batch["var_max"],
                batch["nucleotide_features"],
            )
            seed_scores.append(net.probability(raw).float())
        parts.append(torch.stack(seed_scores, dim=0).mean(0).cpu())
    return torch.cat(parts, dim=0)


def _write_ensemble_meta(out_dir: Path, student_esm: str) -> None:
    members = [str(p.relative_to(out_dir)) for p in sorted(out_dir.glob("seed*/checkpoints/model_best.pt"))]
    (out_dir / "ensemble.json").write_text(
        json.dumps({"task": "lof", "members": members, "distilled_from": "esm2_35m"}, indent=2)
    )
    seed0 = out_dir / "seed0"
    gal = seed0 / "train_gallery.pt"
    if gal.exists():
        (out_dir / "train_gallery.pt").write_bytes(gal.read_bytes())
    cfg = seed0 / "model_config.json"
    if cfg.exists():
        (out_dir / "model_config.json").write_bytes(cfg.read_bytes())
        payload = json.loads(cfg.read_text())
        root = {
            "encoders": {"lof": student_esm},
            "esm2_model_name_lof": student_esm,
            "distilled": True,
        }
        root.update({k: payload.get(k) for k in ("pool_strategy", "head_hidden", "dropout") if k in payload})
        (out_dir / "model_config.json").write_text(json.dumps(root, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-dir", type=Path, required=True, help="35M LoF ensemble dir (…/lof)")
    p.add_argument("--teacher-emb", type=Path, required=True, help="Embedding root with lof/ (35M)")
    p.add_argument("--student-emb", type=Path, required=True, help="Embedding root with lof/ (8M)")
    p.add_argument("--output-dir", type=Path, required=True, help="e.g. outputs/lof_8m")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--model-config", default="configs/model_8m.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--mixed-precision", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument(
        "--distill-alpha", type=float, default=0.85,
        help="Weight on teacher soft labels; (1-alpha) on hard table targets (default 0.85)",
    )
    p.add_argument(
        "--teacher-mode", choices=("mean", "seed"), default="mean",
        help="mean: all student seeds match the teacher ensemble mean; "
             "seed: student seed i matches teacher seed i only",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Student seeds (default 0..4)")
    p.add_argument(
        "--skip-nulls", action="store_true",
        help="Do not score student null embeddings after training",
    )
    args = p.parse_args()

    train_cfg = load_yaml(args.config).get("training", {})
    model_cfg = load_yaml(args.model_config).get("model", {})
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.patience is not None:
        train_cfg["early_stopping_patience"] = args.patience
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    student_esm = esm2_for_task("lof", model_cfg)
    if student_esm != ESM2_LOF_FAST:
        logger.warning("Student encoder is %s (expected %s)", student_esm, ESM2_LOF_FAST)

    teacher_emb = args.teacher_emb / "lof"
    student_emb = args.student_emb / "lof"
    _require_encoder(student_emb, student_esm, label="student")
    teacher_meta = read_encoder_meta(teacher_emb)
    if teacher_meta:
        logger.info("Teacher embeddings: %s", teacher_meta.get("esm2_model_name"))

    teacher_members = discover_ensemble(args.teacher_dir)
    if not teacher_members:
        raise SystemExit(f"No teacher ensemble under {args.teacher_dir}")
    teacher_nets = [_load_net(m, device) for m in teacher_members]
    logger.info("Loaded %s teacher members from %s", len(teacher_nets), args.teacher_dir)

    train_t = CachedDataset(teacher_emb / "train_embeddings.pt")
    val_t = CachedDataset(teacher_emb / "val_embeddings.pt")
    train_s = CachedDataset(student_emb / "train_embeddings.pt")
    val_s = CachedDataset(student_emb / "val_embeddings.pt")
    _assert_row_aligned(train_t, train_s, split="train")
    _assert_row_aligned(val_t, val_s, split="val")

    expected_d = HIDDEN_SIZE.get(student_esm)
    if expected_d is not None and int(train_s.ref_mean.shape[1]) != int(expected_d):
        raise SystemExit(
            f"Student embeddings D={train_s.ref_mean.shape[1]} but {student_esm} is D={expected_d}"
        )

    batch = int(train_cfg.get("batch_size", 256))
    seeds = args.seeds or list(train_cfg.get("ensemble_seeds", [0, 1, 2, 3, 4]))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    soft_cache: dict[str, torch.Tensor] = {}
    if args.teacher_mode == "mean":
        soft_cache["train"] = score_soft_targets(teacher_nets, train_t, device, batch)
        soft_cache["val"] = score_soft_targets(teacher_nets, val_t, device, batch)
        logger.info(
            "Teacher mean soft labels  train mae-vs-hard=%.4f  val=%.4f",
            float(torch.mean(torch.abs(soft_cache["train"] - train_t.targets))),
            float(torch.mean(torch.abs(soft_cache["val"] - val_t.targets))),
        )

    precision = args.mixed_precision or train_cfg.get("mixed_precision", "bf16")
    pin = device.type == "cuda"

    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        if args.teacher_mode == "seed":
            if seed >= len(teacher_nets):
                raise SystemExit(f"--teacher-mode seed needs teacher seed{seed}, only {len(teacher_nets)} members")
            soft_train = score_soft_targets([teacher_nets[seed]], train_t, device, batch)
            soft_val = score_soft_targets([teacher_nets[seed]], val_t, device, batch)
        else:
            soft_train = soft_cache["train"]
            soft_val = soft_cache["val"]

        train_ds = SoftTargetDataset(train_s, soft_train)
        val_ds = SoftTargetDataset(val_s, soft_val)
        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin,
        )

        hidden = int(train_s.ref_mean.shape[1])
        net = TaskNet(
            hidden_size=hidden,
            task="lof",
            pool_strategy=model_cfg.get("pool_strategy", "mean_max"),
            use_cross_attention=model_cfg.get("use_cross_attention", False),
            head_hidden=int(model_cfg.get("head_hidden", 128)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        seed_out = out_root / f"seed{seed}"
        logger.info("── Distill student seed=%s → %s (D=%s, alpha=%.2f) ──", seed, seed_out, hidden, args.distill_alpha)
        trainer = Trainer(
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=seed_out,
            mixed_precision=precision,
            esm2_model_name=student_esm,
            pool_strategy=model_cfg.get("pool_strategy", "mean_max"),
            seed=seed,
            distill_alpha=args.distill_alpha,
        )
        best = trainer.train(
            max_epochs=args.max_epochs or int(train_cfg.get("max_epochs", 40)),
            learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            patience=int(train_cfg.get("early_stopping_patience", 8)),
            grad_accum_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.2)),
        )
        logger.info("seed%s best=%s", seed, {k: round(v, 4) for k, v in best.items() if isinstance(v, float)})

        intact = ~train_s.is_wreck
        if not bool(intact.any()):
            intact = torch.ones(len(train_s), dtype=torch.bool)
        idx = intact.nonzero(as_tuple=False).view(-1).tolist()
        save_gallery(
            seed_out / "train_gallery.pt",
            train_s.ref_mean[intact],
            [train_s.genes[i] for i in idx],
            [train_s.protein_ids[i] for i in idx],
        )
        meta = {
            "task": "lof",
            "seed": seed,
            "esm2_model_name": student_esm,
            "hidden_size": hidden,
            "head_hidden": int(model_cfg.get("head_hidden", 128)),
            "dropout": float(model_cfg.get("dropout", 0.2)),
            "distill_alpha": args.distill_alpha,
            "teacher_dir": str(args.teacher_dir),
            "teacher_mode": args.teacher_mode,
        }
        (seed_out / "model_config.json").write_text(json.dumps(meta, indent=2))

    _write_ensemble_meta(out_root, student_esm)

    null_emb = student_emb / "null_embeddings.pt"
    if not args.skip_nulls and null_emb.exists():
        import subprocess
        import sys

        logger.info("Scoring student nulls from %s", null_emb)
        rc = subprocess.call([
            sys.executable, str(Path(__file__).resolve().parent / "score_nulls.py"),
            "--task", "lof",
            "--ensemble-dir", str(out_root),
            "--embeddings", str(null_emb),
            "--device", str(device),
        ])
        if rc != 0:
            raise SystemExit(f"score_nulls.py failed with code {rc}")
    elif not args.skip_nulls:
        logger.warning("No %s — empirical p unavailable until nulls are embedded", null_emb)

    logger.info("Distilled LoF ensemble → %s", out_root)
    logger.info("Predict with: python scripts/predict.py --model %s --reference ref.fa --variants var.fa", out_root)


if __name__ == "__main__":
    main()
