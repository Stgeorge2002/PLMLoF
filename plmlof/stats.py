"""Empirical p-values and Benjamini–Hochberg q-values.

A neural net does not output a p-value. Rank the observed score in a
held-out null (higher = more extreme).
"""

from __future__ import annotations

import numpy as np


def empirical_p(observed: np.ndarray | float, null_scores: np.ndarray) -> np.ndarray:
    """P(null >= observed) with a +1 continuity correction.

    null_scores must be 1-D. For a LoF score, larger is more extreme.
    For a GoF probability, larger is more extreme.
    """
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    null = np.asarray(null_scores, dtype=np.float64).reshape(-1)
    if null.size == 0:
        raise ValueError("null_scores is empty")
    n = null.size
    null_asc = np.sort(null)
    n_less = np.searchsorted(null_asc, obs, side="left")
    n_ge = n - n_less
    p = (1.0 + n_ge) / (1.0 + n)
    return p


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH q-values (positive false-discovery rate control)."""
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out
