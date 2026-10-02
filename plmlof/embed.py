"""ESM2 unique-sequence embedding used by the precompute script."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from plmlof.sites import token_index

logger = logging.getLogger(__name__)


class _LenSortedSeqs(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sorted(sequences, key=len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.unsqueeze(-1).float()
    mean_p = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
    masked = hidden.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
    max_p = masked.max(dim=1).values
    max_p = max_p.masked_fill(max_p == float("-inf"), 0.0)
    return mean_p, max_p


@torch.no_grad()
def embed_unique_sequences(
    sequences: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_workers: int = 4,
    residue_requests: dict[str, set[int]] | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, dict[tuple[str, int], torch.Tensor]]:
    """Pool unique sequences; optionally bank residue tokens at requested indices."""
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
    site_bank: dict[tuple[str, int], torch.Tensor] = {}
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    if use_amp:
        major, _ = torch.cuda.get_device_capability(device)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16

    for batch in tqdm(loader, desc="ESM2 unique sequences"):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            hidden = model(ids, attention_mask=mask).last_hidden_state
        mean_p, max_p = _pool(hidden, mask)
        means.append(mean_p.float().cpu())
        maxes.append(max_p.float().cpu())
        seqs = batch["sequences"]
        ordered.extend(seqs)
        if residue_requests:
            hidden_cpu = hidden.float().cpu()
            n_tok = hidden_cpu.size(1)
            for b, seq in enumerate(seqs):
                for residue in residue_requests.get(seq, ()):
                    tok = token_index(int(residue))
                    if 0 <= tok < n_tok:
                        site_bank[(seq, int(residue))] = hidden_cpu[b, tok].contiguous()
    return ordered, torch.cat(means), torch.cat(maxes), site_bank
