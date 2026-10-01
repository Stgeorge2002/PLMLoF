"""Pre-compute ESM2 embeddings for all v2 task splits (one unique-seq pass).

    python scripts/precompute_v2.py \
        --data-dir data/processed/v2 \
        --output-dir $PLMLOF_EMB_DIR/v2 \
        --device cuda --batch-size 128
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from plmlof.v2 import TASKS
from plmlof.v2.dataset import V2PairDataset
from plmlof.v2.embed import embed_unique_sequences

logger = logging.getLogger(__name__)


def scatter(dataset: V2PairDataset, seq_to_idx: dict[str, int], mean_t: torch.Tensor, max_t: torch.Tensor, out: Path) -> None:
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
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    logger.info("  %s  %s samples  %.1f MB", out.name, n, out.stat().st_size / 1e6)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/v2"))
    p.add_argument("--output-dir", type=Path, default=Path("data/embeddings/v2"))
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--esm2-model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    jobs: list[tuple[str, str, V2PairDataset, Path]] = []
    unique: set[str] = set()
    for task in args.tasks:
        tdir = args.data_dir / task
        for split in ("train", "val", "test", "null"):
            parquet = tdir / f"{split}.parquet"
            if not parquet.exists() or parquet.stat().st_size < 100:
                continue
            ds = V2PairDataset(parquet, max_seq_length=args.max_seq_length)
            if len(ds) == 0:
                continue
            dest = args.output_dir / task / f"{split}_embeddings.pt"
            jobs.append((task, split, ds, dest))
            unique.update(ds._ref)
            unique.update(ds._var)
            logger.info("%s/%s  %s rows  %s unique so far", task, split, len(ds), len(unique))

    if not jobs:
        raise SystemExit(f"No v2 parquets under {args.data_dir}. Run scripts/prepare_v2_local.sh on a laptop first.")

    logger.info("Embedding %s unique sequences with %s", len(unique), args.esm2_model)
    tokenizer = AutoTokenizer.from_pretrained(args.esm2_model)
    model = AutoModel.from_pretrained(args.esm2_model).to(device)
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

    for task, split, ds, dest in jobs:
        logger.info("Scatter %s/%s", task, split)
        scatter(ds, seq_to_idx, mean_t, max_t, dest)
    logger.info("v2 embeddings complete → %s", args.output_dir)


if __name__ == "__main__":
    main()
