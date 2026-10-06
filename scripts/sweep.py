"""Train LoF + MDG head variants on cached embeddings and write a scoreboard.

    python scripts/sweep.py \
        --precomputed $PLMLOF_EMB_DIR \
        --output-dir $PLMLOF_OUTPUT_DIR/sweep \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

FLAG_KEYS = {
    "learning_rate": "--learning-rate",
    "max_epochs": "--max-epochs",
    "patience": "--patience",
    "batch_size": "--batch-size",
    "rank_loss_weight": "--rank-loss-weight",
    "regression_loss_weight": "--regression-loss-weight",
    "mlof_proteins_per_batch": "--mlof-proteins-per-batch",
    "head_hidden": "--head-hidden",
    "dropout": "--dropout",
    "prokaryote_weight": "--prokaryote-weight",
}

BOOL_FLAGS = {
    "drop_sure_wrecks": "--drop-sure-wrecks",
    "ablate_logodds": "--ablate-logodds",
    "ablate_domain": "--ablate-domain",
    "drop_multi": "--drop-multi",
    "prokaryote_only": "--prokaryote-only",
}

SELECT = {
    "lof": "wreck_auroc",
    "mlof": "within_gene_spearman",
}


def _run(cmd: list[str]) -> None:
    logger.info("+ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _spec_flags(spec: dict) -> list[str]:
    flags: list[str] = []
    for key, flag in BOOL_FLAGS.items():
        if spec.get(key):
            flags.append(flag)
    for key, flag in FLAG_KEYS.items():
        if key in spec and spec[key] is not None:
            flags.extend([flag, str(spec[key])])
    return flags


def _write_ensemble(task_dir: Path, task: str) -> list[str]:
    members = [
        str(p.relative_to(task_dir))
        for p in sorted(task_dir.glob("seed*/checkpoints/model_best.pt"))
    ]
    (task_dir / "ensemble.json").write_text(
        json.dumps({"task": task, "members": members}, indent=2)
    )
    gal = task_dir / "seed0" / "train_gallery.pt"
    dest = task_dir / "train_gallery.pt"
    if gal.exists() and not dest.exists():
        dest.write_bytes(gal.read_bytes())
    cfg = task_dir / "seed0" / "model_config.json"
    if cfg.exists():
        (task_dir / "model_config.json").write_bytes(cfg.read_bytes())
    return members


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="LoF/MDG head sweep on frozen embeddings")
    p.add_argument("--precomputed", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sweep", type=Path, default=Path("configs/sweeps.yaml"))
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    payload = yaml.safe_load(args.sweep.read_text()) or {}
    seeds = [int(s) for s in payload.get("seeds", [0, 1])]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    py = args.python
    rows: list[dict] = []

    for task in ("lof", "mlof"):
        specs = payload.get(task) or []
        if not specs:
            continue
        emb = args.precomputed / task / "train_embeddings.pt"
        if not emb.exists():
            raise SystemExit(f"missing {emb} — embed LoF/MDG before sweeping")
        for spec in specs:
            name = str(spec["name"])
            exp_dir = root / task / name
            logger.info("════ %s / %s  %s", task, name, spec.get("note", ""))
            extra = _spec_flags(spec)
            for seed in seeds:
                out = exp_dir / f"seed{seed}"
                ckpt = out / "checkpoints" / "model_best.pt"
                if ckpt.exists():
                    logger.info("skip train %s (exists)", ckpt)
                    continue
                _run([
                    py, "scripts/train.py",
                    "--task", task,
                    "--seed", str(seed),
                    "--precomputed", str(args.precomputed),
                    "--output-dir", str(out),
                    "--config", args.config,
                    "--model-config", args.model_config,
                    "--device", args.device,
                    "--mixed-precision", args.mixed_precision,
                    "--num-workers", str(args.num_workers),
                    *extra,
                ])
            members = _write_ensemble(exp_dir, task)
            (exp_dir / "experiment.json").write_text(json.dumps(spec, indent=2))
            metrics: dict[str, dict] = {}
            for split in ("test", "val", "protein_test"):
                split_emb = args.precomputed / task / f"{split}_embeddings.pt"
                if not split_emb.exists():
                    continue
                json_out = exp_dir / f"metrics_{split}.json"
                if json_out.exists():
                    logger.info("skip eval %s (exists)", json_out)
                else:
                    _run([
                        py, "scripts/evaluate.py",
                        "--task", task,
                        "--ensemble-dir", str(exp_dir),
                        "--embeddings", str(split_emb),
                        "--device", args.device,
                        "--json-out", str(json_out),
                    ])
                metrics[split] = _load_metrics(json_out)
            key = SELECT[task]
            row = {
                "task": task,
                "name": name,
                "note": spec.get("note", ""),
                "n_members": len(members),
                "test": metrics.get("test", {}).get(key),
                "val": metrics.get("val", {}).get(key),
                "protein_test": metrics.get("protein_test", {}).get(key),
                "test_metrics": metrics.get("test", {}),
                "val_metrics": metrics.get("val", {}),
                "protein_test_metrics": metrics.get("protein_test", {}),
            }
            if task == "mlof":
                row["test_strong_vs_wt"] = metrics.get("test", {}).get("strong_vs_wt_auroc")
                row["protein_test_strong_vs_wt"] = metrics.get("protein_test", {}).get("strong_vs_wt_auroc")
                row["test_collapse"] = metrics.get("test", {}).get("collapse_fraction")
                pt = metrics.get("protein_test", {})
                row["protein_test_prokaryote"] = pt.get("taxon_prokaryote_within_gene_spearman")
                row["protein_test_eukaryote"] = pt.get("taxon_eukaryote_within_gene_spearman")
                row["protein_test_human"] = pt.get("taxon_human_within_gene_spearman")
            rows.append(row)

    scoreboard = {"selection": SELECT, "rows": rows}
    (root / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2))
    lines = ["task\tname\ttest\tval\tprotein_test\tprotein_test_prok\tnote"]
    for row in rows:
        lines.append(
            f"{row['task']}\t{row['name']}\t{row.get('test')}\t{row.get('val')}\t"
            f"{row.get('protein_test')}\t{row.get('protein_test_prokaryote')}\t"
            f"{row.get('note', '')}"
        )
    (root / "scoreboard.tsv").write_text("\n".join(lines) + "\n")
    logger.info("Scoreboard → %s", root / "scoreboard.tsv")
    print((root / "scoreboard.tsv").read_text())


if __name__ == "__main__":
    main()
