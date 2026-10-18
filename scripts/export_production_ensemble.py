"""Trim a 5-seed LoF ensemble to 1–2 members for production predict.

Keeps seed order from ensemble.json (or seed*/ glob). Copies checkpoints plus
null_scores.pt / train_gallery.pt / model_config.json into a slim directory.

    python scripts/export_production_ensemble.py \\
        --src outputs/lof_8m \\
        --dst best-models/lof_8m_prod \\
        --members 1

Teacher eval should keep the full 5-seed tree; batch predict uses the export.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from plmlof.predictor import discover_ensemble

logger = logging.getLogger(__name__)

SIDE_CARDS = (
    "null_scores.pt",
    "train_gallery.pt",
    "model_config.json",
    "metrics.json",
    "eval_metrics.json",
)


def _task_root(src: Path) -> Path:
    if (src / "lof" / "ensemble.json").exists() or list((src / "lof").glob("seed*/checkpoints/model_best.pt")):
        return src / "lof"
    return src


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=str, required=True, help="Teacher/student ensemble dir (…/lof_8m or …/lof)")
    p.add_argument("--dst", type=str, required=True, help="Slim production directory to create")
    p.add_argument("--members", type=int, default=1, choices=(1, 2), help="Seeds to ship (default 1)")
    args = p.parse_args()

    src_root = _task_root(Path(args.src))
    members = discover_ensemble(src_root)
    if not members:
        raise SystemExit(f"No ensemble members under {src_root}")
    keep = members[: args.members]
    dst = Path(args.dst)
    if dst.exists():
        raise SystemExit(f"Refusing to overwrite existing {dst}")
    dst.mkdir(parents=True)

    rel_members: list[str] = []
    for i, ckpt in enumerate(keep):
        seed_name = f"seed{i}"
        # Prefer original seed dirname if present.
        for part in ckpt.parts:
            if part.startswith("seed") and part[4:].isdigit():
                seed_name = part
                break
        out_ckpt = dst / seed_name / "checkpoints" / "model_best.pt"
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, out_ckpt)
        rel_members.append(str(out_ckpt.relative_to(dst)))
        logger.info("Copied %s -> %s", ckpt, out_ckpt)

    for name in SIDE_CARDS:
        src_file = src_root / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)
            logger.info("Copied %s", name)

    payload = {"task": "lof", "members": rel_members, "production": True, "source": str(src_root)}
    (dst / "ensemble.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote %s (%s members)", dst / "ensemble.json", len(rel_members))


if __name__ == "__main__":
    main()
