"""Within-protein RankNet loss for MLoF."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ranknet_loss(
    scores: torch.Tensor,
    z: torch.Tensor,
    protein_ids: list[str],
    is_missense: torch.Tensor,
) -> torch.Tensor:
    """Pairwise ranking: higher score should match lower fitness z.

    Only pairs from the same protein, both missense, with unequal finite z.
    Returns a scalar (0 when the batch has no valid pairs).
    """
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    n = int(scores.shape[0])
    if n < 2 or not protein_ids or len(protein_ids) != n:
        return scores.new_zeros(())

    miss = is_missense.reshape(-1).bool()
    uniq = {p: i for i, p in enumerate(dict.fromkeys(protein_ids))}
    pid = torch.tensor([uniq[p] for p in protein_ids], device=scores.device, dtype=torch.long)

    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=scores.device)
    same = pid[i_idx] == pid[j_idx]
    both = miss[i_idx] & miss[j_idx]
    zi = z[i_idx]
    zj = z[j_idx]
    comparable = same & both & torch.isfinite(zi) & torch.isfinite(zj) & (zi != zj)
    i_idx = i_idx[comparable]
    j_idx = j_idx[comparable]
    if i_idx.numel() == 0:
        return scores.new_zeros(())

    labels = (z[i_idx] < z[j_idx]).float()
    logits = scores[i_idx] - scores[j_idx]
    return F.binary_cross_entropy_with_logits(logits, labels)
