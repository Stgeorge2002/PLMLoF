"""Pre-compute ESM2 embeddings for all training/validation samples.

This eliminates ESM2 forward passes during training, reducing training
time from hours to minutes. Each sample's ref and var protein are
encoded once and saved to disk as tensors.

Optimisations:
  - Cross-split deduplication: each unique protein across train+val
    is embedded exactly once (shared refs get a single forward pass)
  - Length-sorted batching on unique sequences minimises padding waste
  - Optional bucketed batch sampling (--use-bucketing) for tighter
    per-batch length variance and less padding
  - Optional adaptive batch sizing (--adaptive-batch-size) targeting a
    fixed token budget per batch instead of a fixed sample count
  - fp16/bf16 autocast, pinned memory, prefetched DataLoader
  - Optional SDPA/Flash-Attention backend (--attn-implementation sdpa);
    defaults to eager since EsmModel has no SDPA support upstream yet
  - torch.compile enabled by default in default mode (--no-compile to disable;
    reduce-overhead/CUDA-graphs mode is avoided as it recompiles per batch
    shape, which is counterproductive with our variable sequence lengths)
  - Vectorised, chunked scatter via tensor indexing (bounded peak memory)
  - Embedding cache (.embedding_cache.pt) for crash resume

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
from torch.utils.data import DataLoader, Dataset, Sampler
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


class _BucketedSeqDataset(Dataset):
    """Sequences sorted by length and split into contiguous length buckets.

    Used with _BucketBatchSampler so each batch is drawn from a single
    bucket, keeping per-batch length variance (and padding waste) low.
    """

    def __init__(self, sequences: list[str], num_buckets: int = 8):
        self.sequences = sorted(sequences, key=len)
        n = len(self.sequences)
        bucket_size = max(1, -(-n // max(num_buckets, 1)))  # ceil division
        self.buckets: list[tuple[int, int]] = [
            (start, min(start + bucket_size, n)) for start in range(0, n, bucket_size)
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


class _BucketBatchSampler(Sampler):
    """Yields index batches one length-bucket at a time.

    Bucket order and within-bucket order are reshuffled each epoch; the
    last (possibly short) batch of each bucket is kept intact so
    length-mismatched samples never share a batch.
    """

    def __init__(self, buckets: list[tuple[int, int]], batch_size: int, seed: int = 0):
        self.buckets = buckets
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        batches: list[list[int]] = []
        for start, end in self.buckets:
            perm = (torch.randperm(end - start, generator=g) + start).tolist()
            batches.extend(perm[i:i + self.batch_size] for i in range(0, len(perm), self.batch_size))

        order = torch.randperm(len(batches), generator=g).tolist()
        for b in order:
            yield batches[b]

    def __len__(self):
        return sum(-(-(end - start) // self.batch_size) for start, end in self.buckets)


def _compute_adaptive_batch_size(sequences: list[str], target_tokens: int, cap: int = 512) -> int:
    """Pick a batch size so batch_size * avg_length ≈ target_tokens."""
    avg_len = sum(len(s) for s in sequences) / max(len(sequences), 1)
    size = int(target_tokens / max(avg_len, 1))
    return max(16, min(cap, size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute ESM2 embeddings")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--holdout-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/embeddings/")
    parser.add_argument("--esm2-model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (enabled by default on PyTorch 2.0+, ~20-30%% faster)")
    parser.add_argument("--attn-implementation", type=str, default="eager", choices=["sdpa", "eager"],
                        help="Attention backend. 'sdpa' requests PyTorch's fused/Flash-Attention kernels, but "
                             "EsmModel does not implement it as of transformers<4.55 (falls back automatically).")
    parser.add_argument("--use-bucketing", action="store_true",
                        help="Sample batches from a single length-bucket at a time to minimise padding waste")
    parser.add_argument("--num-buckets", type=int, default=8,
                        help="Number of length buckets when --use-bucketing is set")
    parser.add_argument("--adaptive-batch-size", action="store_true",
                        help="Derive batch size from average sequence length (targets --target-tokens per batch)")
    parser.add_argument("--target-tokens", type=int, default=256_000,
                        help="Target tokens per batch when --adaptive-batch-size is set")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes used for tokenization")
    parser.add_argument("--scatter-chunk-size", type=int, default=100_000,
                        help="Rows processed per chunk during scatter, to bound peak memory")
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
    num_workers: int = 4,
    use_bucketing: bool = False,
    num_buckets: int = 8,
    desc: str = "Embedding",
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Embed unique sequences and return indexed tensors.

    Returns:
        (ordered_sequences, mean_tensor [N, D], max_tensor [N, D]) on CPU.
    """
    prefetch_factor = 2 if num_workers > 0 else None
    collate = lambda b: _collate_strings(b, tokenizer, max_seq_length)

    if use_bucketing:
        ds = _BucketedSeqDataset(sequences, num_buckets=num_buckets)
        batch_sampler = _BucketBatchSampler(ds.buckets, batch_size=batch_size)
        loader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
    else:
        ds = _UniqueSeqDataset(sequences)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )

    ordered_seqs: list[str] = []
    mean_chunks: list[torch.Tensor] = []
    max_chunks: list[torch.Tensor] = []
    model.eval()

    for batch in tqdm(loader, desc=desc):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if _is_ampere(device) else torch.float16, enabled=device.type == "cuda"):
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
    chunk_size: int = 100_000,
) -> None:
    """Scatter pre-computed embeddings to per-sample tensors, chunked to bound peak memory."""
    n = len(dataset)
    logger.info(f"  Building index mappings for {n} samples...")

    # Access pre-extracted protein lists directly (skip __getitem__ overhead)
    ref_proteins = [s.replace("*", "") for s in dataset._ref_proteins_raw]
    var_proteins = [s.replace("*", "") for s in dataset._var_proteins_raw]

    logger.info(f"  Creating index tensors...")
    ref_idx = torch.tensor([seq_to_idx[s] for s in ref_proteins], dtype=torch.long)
    var_idx = torch.tensor([seq_to_idx[s] for s in var_proteins], dtype=torch.long)
    del ref_proteins, var_proteins

    hidden_size = mean_tensor.shape[1]
    ref_mean = torch.empty((n, hidden_size), dtype=torch.float32)
    ref_max = torch.empty((n, hidden_size), dtype=torch.float32)
    var_mean = torch.empty((n, hidden_size), dtype=torch.float32)
    var_max = torch.empty((n, hidden_size), dtype=torch.float32)

    logger.info(f"  Gathering embeddings in chunks of {chunk_size}...")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        r_idx = ref_idx[start:end]
        v_idx = var_idx[start:end]
        ref_mean[start:end] = mean_tensor[r_idx]
        ref_max[start:end] = max_tensor[r_idx]
        var_mean[start:end] = mean_tensor[v_idx]
        var_max[start:end] = max_tensor[v_idx]

    data = {
        "ref_mean": ref_mean,
        "ref_max": ref_max,
        "var_mean": var_mean,
        "var_max": var_max,
        "nucleotide_features": dataset._nuc_features,
        "labels": torch.tensor(dataset._labels, dtype=torch.long),
        "dms_scores": torch.tensor(dataset._dms_scores, dtype=torch.float),
    }

    # Estimate output size
    estimated_mb = sum(
        v.numel() * v.element_size() for v in data.values() if isinstance(v, torch.Tensor)
    ) / 1e6
    logger.info(f"  Writing {estimated_mb:.1f} MB to disk (this may take 10-30 seconds)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"  ✓ Saved {n} samples → {output_path} ({size_mb:.1f} MB)")


def _is_ampere(device: torch.device) -> bool:
    """Return True if the device is Ampere or newer (compute capability >= 8.0)."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


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
    if args.test_data:
        logger.info(f"Loading test data: {args.test_data}")
        test_ds = PLMLoFDataset(args.test_data, max_seq_length=args.max_seq_length)
        splits.append(("test", test_ds, output_dir / "test_embeddings.pt"))
    if args.holdout_data:
        logger.info(f"Loading holdout data: {args.holdout_data}")
        holdout_ds = PLMLoFDataset(args.holdout_data, max_seq_length=args.max_seq_length)
        splits.append(("holdout", holdout_ds, output_dir / "holdout_embeddings.pt"))

    # Auto-detect per-species test files (test_ecoli.parquet, test_myctu.parquet, …)
    # produced by curate_dataset.py and add them to the embedding job.
    _data_dir = Path(args.test_data).parent if args.test_data else Path("data/processed")
    for _species_parquet in sorted(_data_dir.glob("test_*.parquet")):
        if _species_parquet.name == "test.parquet":
            continue  # main test already added above
        _tag = _species_parquet.stem  # e.g. "test_ecoli"
        _emb_path = output_dir / f"{_tag}_embeddings.pt"
        logger.info(f"Auto-detected per-species test file: {_species_parquet.name}")
        _ds = PLMLoFDataset(str(_species_parquet), max_seq_length=args.max_seq_length)
        if len(_ds) > 0:
            splits.append((_tag, _ds, _emb_path))

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
        try:
            model = AutoModel.from_pretrained(
                args.esm2_model, attn_implementation=args.attn_implementation
            ).to(device)
            logger.info(f"  Attention backend: {args.attn_implementation}")
        except Exception as e:
            logger.warning(
                f"  '{args.attn_implementation}' attention unavailable ({e}), "
                "falling back to the model's default backend"
            )
            model = AutoModel.from_pretrained(args.esm2_model).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        if not args.no_compile and hasattr(torch, "compile"):
            try:
                # Default mode, not reduce-overhead: batches here have highly variable
                # sequence lengths, and reduce-overhead's CUDA graphs re-capture on every
                # new shape instead of generalising, recompiling on nearly every batch.
                # Default mode settles into fast dynamic-shape execution after ~2 recompiles.
                logger.info("Compiling ESM2 with torch.compile (first couple of batches will be slow)")
                model = torch.compile(model)
            except RuntimeError as e:
                logger.warning(f"  torch.compile failed ({e}), continuing uncompiled")

        batch_size = args.batch_size
        if args.adaptive_batch_size:
            batch_size = _compute_adaptive_batch_size(list(all_unique), args.target_tokens)
            logger.info(f"  Adaptive batch size: {batch_size} (target_tokens={args.target_tokens})")

        ordered_seqs, mean_tensor, max_tensor = _embed_unique_sequences(
            list(all_unique), model, tokenizer, device,
            batch_size, args.max_seq_length,
            num_workers=args.num_workers,
            use_bucketing=args.use_bucketing,
            num_buckets=args.num_buckets,
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
    import gc
    import sys
    
    for split_idx, (name, ds, out_path) in enumerate(splits, 1):
        logger.info(f"Scattering {name} ({len(ds)} samples) [{split_idx}/{len(splits)}]...")
        sys.stdout.flush()  # Ensure logs appear immediately
        
        _scatter_embeddings(ds, seq_to_idx, mean_tensor, max_tensor, out_path, chunk_size=args.scatter_chunk_size)
        
        # Force garbage collection after each scatter to free memory
        gc.collect()
        sys.stdout.flush()

    # Clean up cache only after ALL splits have been successfully scattered
    if cache_path.exists():
        cache_path.unlink()
        logger.info("Cleaned up embedding cache")

    logger.info("Embedding pre-computation complete!")


if __name__ == "__main__":
    main()
