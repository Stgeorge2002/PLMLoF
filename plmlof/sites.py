"""Aligned missense residue index and ESM2 token windows.

ESM2 (HuggingFace) is ``[CLS] + residues + [EOS]``. Residue ``i`` (0-based) sits
at token ``i + 1``. Training keeps Hamming ≤ 2 so the ranker sees the same max-of-sites
rule used at predict. Identity is Hamming 0.
"""

from __future__ import annotations

import torch

from plmlof.constants import MLOF_MAX_HAMMING, SITE_RADIUS

ESM_CLS_OFFSET = 1


def aligned_site_index(ref: str, var: str) -> int:
    """0-based index of the first substitution, or 0 for identity, or -1 if unaligned."""
    sites = missense_sites(ref, var)
    if sites:
        return sites[0]
    if not ref or not var:
        return -1
    if len(ref) != len(var):
        return -1
    return 0


def missense_sites(ref: str, var: str) -> list[int]:
    """0-based indices of same-length substitutions. Empty if unaligned or identity."""
    if not ref or not var or len(ref) != len(var):
        return []
    return [i for i, (a, b) in enumerate(zip(ref, var)) if a != b]


def aa_hamming(ref: str, var: str) -> int:
    """Number of substitutions on same-length proteins; 0 if unaligned or identity."""
    return len(missense_sites(ref, var))


def is_mlof_missense(ref: str, var: str, max_hamming: int = MLOF_MAX_HAMMING) -> bool:
    n = aa_hamming(ref, var)
    return 1 <= n <= int(max_hamming)


def window_indices(center: int, length: int, radius: int = SITE_RADIUS) -> tuple[int, ...]:
    """``2*radius+1`` residue indices; ``-1`` for out-of-range slots."""
    width = 2 * int(radius) + 1
    if center < 0 or length <= 0:
        return tuple(-1 for _ in range(width))
    return tuple(
        center + off if 0 <= center + off < length else -1
        for off in range(-int(radius), int(radius) + 1)
    )


def neighbor_indices(center: int, length: int) -> tuple[int, ...]:
    return window_indices(center, length, SITE_RADIUS)


def token_index(residue: int) -> int:
    return residue + ESM_CLS_OFFSET


def gather_site_windows(
    hidden: torch.Tensor,
    sequences: list[str],
    centers: list[int],
    radius: int = SITE_RADIUS,
) -> torch.Tensor:
    """Pull a ``2*radius+1`` residue window. ``hidden`` is ``[B, T, D]``.

    Out-of-range neighbours (and unaligned centres) stay zero.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T, D], got {tuple(hidden.shape)}")
    if len(sequences) != hidden.size(0) or len(centers) != hidden.size(0):
        raise ValueError("sequences, centers, and hidden batch dim must match")
    width = 2 * int(radius) + 1
    bsz, n_tok, dim = hidden.shape
    out = hidden.new_zeros(bsz, width, dim)
    for b, (seq, center) in enumerate(zip(sequences, centers)):
        if center < 0 or not seq:
            continue
        for slot, residue in enumerate(window_indices(center, len(seq), radius)):
            if residue < 0:
                continue
            tok = token_index(residue)
            if 0 <= tok < n_tok:
                out[b, slot] = hidden[b, tok]
    return out
