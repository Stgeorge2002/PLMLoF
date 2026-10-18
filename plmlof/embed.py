"""ESM2 unique-sequence embedding used by precompute and predict."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class _LenSortedSeqs(Dataset):
    """Sort by length so padded ESM2 batches waste less compute."""

    def __init__(self, sequences: list[str]):
        self.sequences = sorted(sequences, key=len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.unsqueeze(-1).float()
    summed = (hidden * mask_f).sum(dim=1)
    lengths = mask_f.sum(dim=1).clamp(min=1.0)
    mean_p = summed / lengths
    max_p = hidden.masked_fill(mask.unsqueeze(-1) == 0, float("-inf")).max(dim=1).values
    max_p = torch.where(torch.isfinite(max_p), max_p, torch.zeros_like(max_p))
    return mean_p, max_p


def resolve_amp_dtype(device: torch.device, prefer: str | None = None) -> torch.dtype | None:
    """Return autocast dtype for CUDA, or None to disable autocast."""
    if device.type != "cuda":
        return None
    if prefer == "fp32":
        return None
    if prefer == "fp8":
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is not None:
            return dtype
        logger.warning("FP8 not available in this torch build; falling back to bf16/fp16")
    if prefer == "fp16":
        return torch.float16
    if prefer in (None, "bf16", "auto", "fp8"):
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    raise ValueError(f"unknown amp prefer={prefer!r}")


def forward_hidden_and_logits(model, ids: torch.Tensor, mask: torch.Tensor):
    """Return last_hidden_state (and logits if the model is a MaskedLM)."""
    if hasattr(model, "esm"):
        hidden = model.esm(ids, attention_mask=mask).last_hidden_state
        logits = model.lm_head(hidden) if hasattr(model, "lm_head") else None
        return hidden, logits
    out = model(ids, attention_mask=mask)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise AttributeError(
            f"{type(out).__name__} has no last_hidden_state; expected EsmModel or EsmForMaskedLM"
        )
    return hidden, getattr(out, "logits", None)


@torch.no_grad()
def embed_unique_sequences(
    sequences: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_workers: int = 4,
    amp_dtype: torch.dtype | None | str = "auto",
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Pool unique sequences with ESM2 (length-sorted batches)."""
    ds = _LenSortedSeqs(sequences)

    def collate(batch: list[str]) -> dict:
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        enc["sequences"] = batch
        return enc

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    means, maxes, ordered = [], [], []
    if isinstance(amp_dtype, str) or amp_dtype == "auto":
        resolved = resolve_amp_dtype(device, prefer=None if amp_dtype == "auto" else str(amp_dtype))
    else:
        resolved = amp_dtype
    use_amp = resolved is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="ESM2 unique sequences"):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=resolved, enabled=use_amp):
            hidden, _logits = forward_hidden_and_logits(model, ids, mask)
        mean_p, max_p = _pool(hidden, mask)
        means.append(mean_p.float().cpu())
        maxes.append(max_p.float().cpu())
        ordered.extend(batch["sequences"])
        del hidden

    return ordered, torch.cat(means), torch.cat(maxes)
