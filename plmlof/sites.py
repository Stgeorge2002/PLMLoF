"""Aligned missense residue index and ESM2 token windows.

ESM2 (HuggingFace) is ``[CLS] + residues + [EOS]``. Residue ``i`` (0-based) sits
at token ``i + 1``. MLoF only trains Hamming-1 pairs, so the substituted
residue is a well-defined column in both hidden states.
"""

from __future__ import annotations

import torch

ESM_CLS_OFFSET = 1


def aligned_site_index(ref: str, var: str) -> int:
    """0-based index of the substitution, or 0 for identity, or -1 if unaligned.

    Same-length Hamming-1 → the mutated residue. Identity → 0 (delta is zero).
    Length change or empty → -1 (no site window).
    """
    if not ref or not var:
        return -1
    if len(ref) != len(var):
        return -1
    if ref == var:
        return 0
    hits = [i for i, (a, b) in enumerate(zip(ref, var)) if a != b]
    return hits[0] if hits else 0


def neighbor_indices(center: int, length: int) -> tuple[int, int, int]:
    """``(left, center, right)`` with ``-1`` for an out-of-range neighbour."""
    if center < 0 or length <= 0:
        return -1, -1, -1
    left = center - 1 if center > 0 else -1
    right = center + 1 if center + 1 < length else -1
    return left, center, right


def token_index(residue: int) -> int:
    return residue + ESM_CLS_OFFSET


def gather_site_windows(
    hidden: torch.Tensor,
    sequences: list[str],
    centers: list[int],
) -> torch.Tensor:
    """Pull ``(i-1, i, i+1)`` residue tokens. ``hidden`` is ``[B, T, D]``.

    Out-of-range neighbours (and unaligned centres) stay zero.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T, D], got {tuple(hidden.shape)}")
    if len(sequences) != hidden.size(0) or len(centers) != hidden.size(0):
        raise ValueError("sequences, centers, and hidden batch dim must match")
    bsz, n_tok, dim = hidden.shape
    out = hidden.new_zeros(bsz, 3, dim)
    for b, (seq, center) in enumerate(zip(sequences, centers)):
        if center < 0 or not seq:
            continue
        left, mid, right = neighbor_indices(center, len(seq))
        for slot, residue in enumerate((left, mid, right)):
            if residue < 0:
                continue
            tok = token_index(residue)
            if 0 <= tok < n_tok:
                out[b, slot] = hidden[b, tok]
    return out
