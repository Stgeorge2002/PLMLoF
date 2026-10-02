"""Nearest-training-protein applicability (in-family flag)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def cosine_max(query: torch.Tensor, gallery: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """query [B, D], gallery [N, D] → (max_sim [B], argmax [B])."""
    q = torch.nn.functional.normalize(query.float(), dim=-1)
    g = torch.nn.functional.normalize(gallery.float(), dim=-1)
    sim = q @ g.T
    vals, idx = sim.max(dim=-1)
    return vals, idx


def save_gallery(
    path: Path,
    ref_means: torch.Tensor,
    genes: list[str],
    protein_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    keep = []
    for i, pid in enumerate(protein_ids):
        if pid not in seen:
            seen[pid] = i
            keep.append(i)
    idx = torch.tensor(keep, dtype=torch.long)
    torch.save(
        {
            "ref_mean": ref_means[idx].contiguous(),
            "genes": [genes[i] for i in keep],
            "protein_ids": [protein_ids[i] for i in keep],
        },
        path,
    )


def load_gallery(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def in_family_flags(
    query_mean: torch.Tensor,
    gallery: dict,
    threshold: float,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    g = gallery["ref_mean"]
    sims, idx = cosine_max(query_mean.cpu(), g)
    genes = gallery["genes"]
    nearest = [genes[int(i)] for i in idx.tolist()]
    return (sims.numpy() >= threshold), nearest, sims.numpy()
