"""Pre-compute ESM2 embeddings for all training/validation samples.

This eliminates ESM2 forward passes during Stage 1 training, reducing
training time from hours to minutes. Each sample's ref and var protein
are encoded once and saved to disk as tensors.

Usage:
    python scripts/precompute_embeddings.py \
        --train-data data/processed/train.parquet \
        --val-data data/processed/val.parquet \
        --output-dir data/embeddings/ \
        --device cuda \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from plmlof.data.dataset import PLMLoFDataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute ESM2 embeddings")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/embeddings/")
    parser.add_argument("--esm2-model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()


def _collate_for_embedding(batch: list[dict], tokenizer, max_length: int) -> dict:
    """Collate samples for embedding extraction."""
    ref_seqs = [s["ref_protein"] for s in batch]
    var_seqs = [s["var_protein"] for s in batch]

    ref_enc = tokenizer(ref_seqs, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    var_enc = tokenizer(var_seqs, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")

    nuc_features = torch.stack([s["nucleotide_features"] for s in batch])
    labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)

    return {
        "ref_input_ids": ref_enc["input_ids"],
        "ref_attention_mask": ref_enc["attention_mask"],
        "var_input_ids": var_enc["input_ids"],
        "var_attention_mask": var_enc["attention_mask"],
        "nucleotide_features": nuc_features,
        "labels": labels,
    }


@torch.no_grad()
def precompute_split(
    dataset: PLMLoFDataset,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    output_path: Path,
    device: torch.device,
    batch_size: int = 32,
    max_seq_length: int = 1024,
) -> None:
    """Pre-compute and save pooled embeddings for a dataset split."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_for_embedding(b, tokenizer, max_seq_length),
        num_workers=4,
        pin_memory=True,
    )

    all_ref_mean = []
    all_ref_max = []
    all_var_mean = []
    all_var_max = []
    all_nuc = []
    all_labels = []

    model.eval()
    for batch in tqdm(loader, desc=f"Encoding {output_path.stem}"):
        # Move to device
        ref_ids = batch["ref_input_ids"].to(device)
        ref_mask = batch["ref_attention_mask"].to(device)
        var_ids = batch["var_input_ids"].to(device)
        var_mask = batch["var_attention_mask"].to(device)

        # Forward through ESM2 with fp16
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            ref_out = model(ref_ids, attention_mask=ref_mask).last_hidden_state
            var_out = model(var_ids, attention_mask=var_mask).last_hidden_state

        # Pool: mean and max (matching ComparisonModule._pool logic)
        def _pool(emb, mask):
            mask_f = mask.unsqueeze(-1).float()
            mean_p = (emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            emb_masked = emb.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            max_p = emb_masked.max(dim=1).values
            max_p = max_p.masked_fill(max_p == float("-inf"), 0.0)
            return mean_p.float().cpu(), max_p.float().cpu()

        ref_mean, ref_max = _pool(ref_out, ref_mask)
        var_mean, var_max = _pool(var_out, var_mask)

        all_ref_mean.append(ref_mean)
        all_ref_max.append(ref_max)
        all_var_mean.append(var_mean)
        all_var_max.append(var_max)
        all_nuc.append(batch["nucleotide_features"])
        all_labels.append(batch["labels"])

    # Save as single tensor file
    data = {
        "ref_mean": torch.cat(all_ref_mean),
        "ref_max": torch.cat(all_ref_max),
        "var_mean": torch.cat(all_var_mean),
        "var_max": torch.cat(all_var_max),
        "nucleotide_features": torch.cat(all_nuc),
        "labels": torch.cat(all_labels),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    n = len(data["labels"])
    hidden = data["ref_mean"].shape[1]
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved {n} embeddings (hidden={hidden}) to {output_path} ({size_mb:.1f} MB)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading ESM2: {args.esm2_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.esm2_model)
    model = AutoModel.from_pretrained(args.esm2_model).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

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
