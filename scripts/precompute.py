"""Pre-compute ESM2 embeddings for task splits.

LoF uses ESM2-35M. MLoF and both GoF heads share ESM2-650M.
One unique-sequence pass per encoder.

    python scripts/precompute.py \
        --data-dir data/processed \
        --output-dir $PLMLOF_EMB_DIR \
        --device cuda --batch-size 128
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from plmlof.constants import TASKS
from plmlof.dataset import PairDataset
from plmlof.embed import embed_unique_sequences
from plmlof.encoders import ESM2_PAIR, esm2_for_task, read_encoder_meta, write_encoder_meta

logger = logging.getLogger(__name__)


def scatter(dataset: PairDataset, seq_to_idx: dict[str, int], mean_t: torch.Tensor, max_t: torch.Tensor, out: Path) -> None:
    n = len(dataset)
    hidden = mean_t.shape[1]
    ref_idx = torch.tensor([seq_to_idx[s] for s in dataset._ref], dtype=torch.long)
    var_idx = torch.tensor([seq_to_idx[s] for s in dataset._var], dtype=torch.long)
    data = {
        "ref_mean": mean_t[ref_idx].contiguous(),
        "ref_max": max_t[ref_idx].contiguous(),
        "var_mean": mean_t[var_idx].contiguous(),
        "var_max": max_t[var_idx].contiguous(),
        "nucleotide_features": dataset._nuc,
        "targets": torch.tensor(dataset._target, dtype=torch.float32),
        "weights": torch.tensor(dataset._weight, dtype=torch.float32),
        "dms_zscores": torch.tensor(dataset._z, dtype=torch.float32),
        "is_wreck": torch.tensor(dataset._is_wreck, dtype=torch.bool),
        "is_missense": torch.tensor(dataset._is_missense, dtype=torch.bool),
        "genes": dataset._genes,
        "protein_ids": dataset._protein_id,
        "channels": dataset._channel,
        "hidden_size": hidden,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    logger.info("Wrote %s  rows=%s  D=%s", out, n, hidden)


def _load_model_cfg(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("model", {}) or {}


def _fresh(parquet: Path, dest: Path, wanted: str, task: str) -> bool:
    """True if dest can be reused (exists, newer than parquet, matching encoder)."""
    if not dest.exists() or parquet.stat().st_mtime > dest.stat().st_mtime:
        return False
    meta = read_encoder_meta(dest.parent)
    if meta:
        return meta.get("esm2_model_name") == wanted
    return task != "lof" and wanted == ESM2_PAIR


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--esm2-model", default=None, help="Override pair-head encoder (MLoF/GoF)")
    p.add_argument("--esm2-model-lof", default=None, help="Override LoF encoder")
    p.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = p.parse_args()

    model_cfg = _load_model_cfg(args.model_config)
    if args.esm2_model:
        model_cfg["esm2_model_name"] = args.esm2_model
    if args.esm2_model_lof:
        model_cfg["esm2_model_name_lof"] = args.esm2_model_lof

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    by_encoder: dict[str, list[tuple[str, str, PairDataset, Path]]] = defaultdict(list)
    for task in args.tasks:
        wanted = esm2_for_task(task, model_cfg)
        tdir = args.data_dir / task
        for split in ("train", "val", "test", "null"):
            parquet = tdir / f"{split}.parquet"
            if not parquet.exists() or parquet.stat().st_size < 100:
                continue
            dest = args.output_dir / task / f"{split}_embeddings.pt"
            if _fresh(parquet, dest, wanted, task):
                logger.info("Skip %s/%s (embeddings match %s)", task, split, wanted)
                continue
            ds = PairDataset(parquet, max_seq_length=args.max_seq_length)
            if len(ds) == 0:
                continue
            by_encoder[wanted].append((task, split, ds, dest))
            logger.info("%s/%s  %s rows  encoder=%s", task, split, len(ds), wanted)

    if not by_encoder:
        logger.info("All embeddings up to date under %s", args.output_dir)
        return

    for esm_name, jobs in by_encoder.items():
        unique: set[str] = set()
        for _, _, ds, _ in jobs:
            unique.update(ds._ref)
            unique.update(ds._var)
        logger.info("Embedding %s unique sequences with %s", len(unique), esm_name)
        tokenizer = AutoTokenizer.from_pretrained(esm_name)
        model = AutoModel.from_pretrained(esm_name).to(device)
        model.eval()
        for p_ in model.parameters():
            p_.requires_grad = False
        if not args.no_compile and hasattr(torch, "compile") and device.type == "cuda":
            try:
                model = torch.compile(model)
            except RuntimeError as exc:
                logger.warning("torch.compile skipped: %s", exc)

        ordered, mean_t, max_t = embed_unique_sequences(
            list(unique), model, tokenizer, device,
            args.batch_size, args.max_seq_length, num_workers=args.num_workers,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seq_to_idx = {s: i for i, s in enumerate(ordered)}
        hidden = int(mean_t.shape[1])
        stamped: set[str] = set()
        for task, split, ds, dest in jobs:
            logger.info("Scatter %s/%s", task, split)
            scatter(ds, seq_to_idx, mean_t, max_t, dest)
            if task not in stamped:
                write_encoder_meta(dest.parent, esm_name, hidden)
                stamped.add(task)

    logger.info("Embeddings complete → %s", args.output_dir)


if __name__ == "__main__":
    main()
