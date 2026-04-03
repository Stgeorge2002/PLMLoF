"""Pre-compute ESM2 embeddings for all training/validation samples.

This eliminates ESM2 forward passes during training, reducing training
time from hours to minutes. Each sample's ref and var protein are
encoded once and saved to disk as tensors.

Optimisations:
  - Cross-split deduplication: each unique protein across train+val
    is embedded exactly once (shared refs get a single forward pass)
  - Length-sorted batching on unique sequences minimises padding waste
  - fp16 autocast, pinned memory, prefetched DataLoader
  - Vectorised scatter via tensor indexing (no per-sample Python loop)
  - Embedding cache (.embedding_cache.pt) for crash resume
  - Optional torch.compile for ESM2 forward pass

Usage:
    python scripts/precompute_embeddings.py \
        --train-data data/processed/train.parquet \
        --val-data data/processed/val.parquet \
        --output-dir data/embeddings/ \
        --device cuda \
        --batch-size 256
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from plmlof.data.dataset import PLMLoFDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight dataset over unique sequences (for DataLoader compatibility)
# ---------------------------------------------------------------------------
class _UniqueSeqDataset(Dataset):
    """Thin wrapper: stores unique sequences sorted by length for batching."""

    def __init__(self, sequences: list[str]):
        # Sort by length so consecutive batches have similar-length seqs
        self.sequences = sorted(sequences, key=len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute ESM2 embeddings")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/embeddings/")
    parser.add_argument("--esm2-model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on ESM2 (requires PyTorch 2.0+, ~30%% faster)")
    return parser.parse_args()


def _collate_strings(batch: list[str], tokenizer, max_length: int) -> dict:
    """Tokenize a batch of raw protein strings."""
    enc = tokenizer(batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "sequences": batch}


def _pool(emb: torch.Tensor, mask: torch.Tensor):
    """Mean + max pool, matching ComparisonModule._pool."""
    m_f = mask.unsqueeze(-1).float()
    mean_p = (emb * m_f).sum(1) / m_f.sum(1).clamp(min=1)
    emb_masked = emb.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
    max_p = emb_masked.max(dim=1).values
    max_p = max_p.masked_fill(max_p == float("-inf"), 0.0)
    return mean_p.float(), max_p.float()


@torch.no_grad()
def _embed_unique_sequences(
    sequences: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_seq_length: int,
    desc: str = "Embedding",
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Embed unique sequences and return indexed tensors.

    Returns:
        (ordered_sequences, mean_tensor [N, D], max_tensor [N, D]) on CPU.
    """
    ds = _UniqueSeqDataset(sequences)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_strings(b, tokenizer, max_seq_length),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    ordered_seqs: list[str] = []
    mean_chunks: list[torch.Tensor] = []
    max_chunks: list[torch.Tensor] = []
    model.eval()

    for batch in tqdm(loader, desc=desc):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(ids, attention_mask=mask).last_hidden_state

        mean_p, max_p = _pool(out, mask)
        mean_chunks.append(mean_p.cpu())
        max_chunks.append(max_p.cpu())
        ordered_seqs.extend(batch["sequences"])

    return ordered_seqs, torch.cat(mean_chunks), torch.cat(max_chunks)


def _scatter_embeddings(
    dataset: PLMLoFDataset,
    seq_to_idx: dict[str, int],
    mean_tensor: torch.Tensor,
    max_tensor: torch.Tensor,
    output_path: Path,
) -> None:
    """Scatter pre-computed embeddings to per-sample tensors using vectorised indexing."""
    n = len(dataset)

    # Access pre-extracted protein lists directly (skip __getitem__ overhead)
    ref_proteins = [s.replace("*", "") for s in dataset._ref_proteins_raw]
    var_proteins = [s.replace("*", "") for s in dataset._var_proteins_raw]

    ref_idx = torch.tensor([seq_to_idx[s] for s in ref_proteins], dtype=torch.long)
    var_idx = torch.tensor([seq_to_idx[s] for s in var_proteins], dtype=torch.long)

    # Vectorised gather — single C-level indexing op per tensor
    data = {
        "ref_mean": mean_tensor[ref_idx],
        "ref_max": max_tensor[ref_idx],
        "var_mean": mean_tensor[var_idx],
        "var_max": max_tensor[var_idx],
        "nucleotide_features": dataset._nuc_features,
        "labels": torch.tensor(dataset._labels, dtype=torch.long),
        "dms_scores": torch.tensor(dataset._dms_scores, dtype=torch.float),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"  Saved {n} samples → {output_path} ({size_mb:.1f} MB)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load datasets ──
    logger.info(f"Loading train data: {args.train_data}")
    train_ds = PLMLoFDataset(args.train_data, max_seq_length=args.max_seq_length)
    splits: list[tuple[str, PLMLoFDataset, Path]] = [
        ("train", train_ds, output_dir / "train_embeddings.pt"),
    ]

    if args.val_data:
        logger.info(f"Loading val data: {args.val_data}")
        val_ds = PLMLoFDataset(args.val_data, max_seq_length=args.max_seq_length)
        splits.append(("val", val_ds, output_dir / "val_embeddings.pt"))

    # ── Collect unique sequences across ALL splits ──
    all_unique: set[str] = set()
    for name, ds, _ in splits:
        ref_seqs = {s.replace("*", "") for s in ds._ref_proteins_raw}
        var_seqs = {s.replace("*", "") for s in ds._var_proteins_raw}
        split_unique = ref_seqs | var_seqs
        logger.info(f"  {name}: {len(ds)} samples, {len(split_unique)} unique sequences")
        all_unique |= split_unique

    total_seqs = sum(len(ds) * 2 for _, ds, _ in splits)
    logger.info(
        f"Cross-split dedup: {total_seqs} total → {len(all_unique)} unique "
        f"({(1 - len(all_unique) / max(total_seqs, 1)) * 100:.0f}% reduction)"
    )

    # ── Embed unique sequences (with cache for crash resume) ──
    cache_path = output_dir / ".embedding_cache.pt"
    need_embed = True

    if cache_path.exists():
        logger.info(f"Found embedding cache: {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
        cached_set = set(cache["sequences"])
        if all_unique <= cached_set:
            logger.info(f"  Cache complete ({len(cached_set)} sequences), skipping ESM2")
            ordered_seqs = cache["sequences"]
            mean_tensor = cache["means"]
            max_tensor = cache["maxes"]
            need_embed = False
        else:
            logger.info(
                f"  Cache stale ({len(cached_set)} cached, "
                f"{len(all_unique - cached_set)} missing), re-embedding all"
            )

    if need_embed:
        logger.info(f"Loading ESM2: {args.esm2_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.esm2_model)
        model = AutoModel.from_pretrained(args.esm2_model).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        if args.compile and hasattr(torch, "compile"):
            logger.info("Compiling ESM2 with torch.compile (first batch will be slow)")
            model = torch.compile(model)

        ordered_seqs, mean_tensor, max_tensor = _embed_unique_sequences(
            list(all_unique), model, tokenizer, device,
            args.batch_size, args.max_seq_length,
            desc="Encoding all splits",
        )

        # Save cache for crash resume
        logger.info(f"Saving embedding cache → {cache_path}")
        torch.save(
            {"sequences": ordered_seqs, "means": mean_tensor, "maxes": max_tensor},
            cache_path,
        )

        # Free GPU memory before scatter
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Build sequence → index lookup ──
    seq_to_idx = {seq: i for i, seq in enumerate(ordered_seqs)}

    # ── Scatter to each split ──
    for name, ds, out_path in splits:
        logger.info(f"Scattering {name} ({len(ds)} samples)...")
        _scatter_embeddings(ds, seq_to_idx, mean_tensor, max_tensor, out_path)

    # Clean up cache (output files are the source of truth now)
    if cache_path.exists():
        cache_path.unlink()
        logger.info("Cleaned up embedding cache")

    logger.info("Embedding pre-computation complete!")


if __name__ == "__main__":
    main()
