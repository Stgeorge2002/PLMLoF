"""Pre-compute ESM2 embeddings for all training/validation samples.

This eliminates ESM2 forward passes during training, reducing training
time from hours to minutes. Each sample's ref and var protein are
encoded once and saved to disk as tensors.

Optimisations:
  - Sequence-level deduplication: each unique protein is embedded once
  - Length-sorted batching on unique sequences minimises padding waste
  - fp16 autocast, pinned memory, prefetched DataLoader
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
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Embed a list of unique sequences and return a lookup dict.

    Returns:
        Dict mapping sequence → (mean_pooled [D], max_pooled [D]) on CPU.
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

    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    model.eval()

    for batch in tqdm(loader, desc=desc):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        seqs = batch["sequences"]

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(ids, attention_mask=mask).last_hidden_state

        mean_p, max_p = _pool(out, mask)
        mean_cpu = mean_p.cpu()
        max_cpu = max_p.cpu()

        for j, seq in enumerate(seqs):
            lookup[seq] = (mean_cpu[j], max_cpu[j])

    return lookup


@torch.no_grad()
def precompute_split(
    dataset: PLMLoFDataset,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    output_path: Path,
    device: torch.device,
    batch_size: int = 256,
    max_seq_length: int = 1024,
) -> None:
    """Pre-compute and save pooled embeddings for a dataset split.

    Deduplicates all protein sequences (ref ∪ var), embeds each unique
    sequence exactly once, then scatters results back per sample.
    """
    n_samples = len(dataset)

    # Collect all ref/var protein strings and deduplicate
    ref_seqs = [dataset[i]["ref_protein"] for i in range(n_samples)]
    var_seqs = [dataset[i]["var_protein"] for i in range(n_samples)]
    unique_seqs = list(set(ref_seqs) | set(var_seqs))

    logger.info(
        f"  {n_samples} samples → {n_samples * 2} total sequences, "
        f"{len(unique_seqs)} unique ({(1 - len(unique_seqs) / (n_samples * 2)) * 100:.0f}% dedup)"
    )

    # Embed unique sequences
    lookup = _embed_unique_sequences(
        unique_seqs, model, tokenizer, device, batch_size, max_seq_length,
        desc=f"Encoding {output_path.stem}",
    )

    # Scatter back to per-sample tensors
    hidden = next(iter(lookup.values()))[0].shape[0]
    all_ref_mean = torch.empty(n_samples, hidden)
    all_ref_max = torch.empty(n_samples, hidden)
    all_var_mean = torch.empty(n_samples, hidden)
    all_var_max = torch.empty(n_samples, hidden)
    all_nuc = torch.empty(n_samples, dataset[0]["nucleotide_features"].shape[0])
    all_labels = torch.empty(n_samples, dtype=torch.long)
    all_dms = torch.empty(n_samples)

    for i in range(n_samples):
        sample = dataset[i]
        ref_m, ref_x = lookup[sample["ref_protein"]]
        var_m, var_x = lookup[sample["var_protein"]]
        all_ref_mean[i] = ref_m
        all_ref_max[i] = ref_x
        all_var_mean[i] = var_m
        all_var_max[i] = var_x
        all_nuc[i] = sample["nucleotide_features"]
        all_labels[i] = sample["label"]
        all_dms[i] = sample.get("dms_score", 0.0)

    data = {
        "ref_mean": all_ref_mean,
        "ref_max": all_ref_max,
        "var_mean": all_var_mean,
        "var_max": all_var_max,
        "nucleotide_features": all_nuc,
        "labels": all_labels,
        "dms_scores": all_dms,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved {n_samples} embeddings (hidden={hidden}) to {output_path} ({size_mb:.1f} MB)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading ESM2: {args.esm2_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.esm2_model)
    model = AutoModel.from_pretrained(args.esm2_model).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling ESM2 with torch.compile (first batch will be slow)")
        model = torch.compile(model)

    # Train split
    logger.info(f"Loading train data: {args.train_data}")
    train_ds = PLMLoFDataset(args.train_data, max_seq_length=args.max_seq_length)
    logger.info(f"Train samples: {len(train_ds)}")
    precompute_split(train_ds, model, tokenizer, output_dir / "train_embeddings.pt",
                     device, args.batch_size, args.max_seq_length)

    # Val split
    if args.val_data:
        logger.info(f"Loading val data: {args.val_data}")
        val_ds = PLMLoFDataset(args.val_data, max_seq_length=args.max_seq_length)
        logger.info(f"Val samples: {len(val_ds)}")
        precompute_split(val_ds, model, tokenizer, output_dir / "val_embeddings.pt",
                         device, args.batch_size, args.max_seq_length)

    logger.info("Embedding pre-computation complete!")


if __name__ == "__main__":
    main()
