"""Validation metrics for v2 LoF (graded) and GoF (precision-first)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    from scipy.stats import spearmanr

    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if rho == rho else 0.0


def lof_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    z: np.ndarray,
    is_wreck: np.ndarray,
    is_missense: np.ndarray,
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    is_wreck = np.asarray(is_wreck, dtype=bool)
    is_missense = np.asarray(is_missense, dtype=bool)
    out: dict[str, float] = {
        "mae": float(np.mean(np.abs(pred - target))),
        "n": float(pred.size),
    }

    if is_wreck.any() and (~is_wreck).any():
        y = is_wreck.astype(np.int32)
        try:
            out["wreck_auroc"] = float(roc_auc_score(y, pred))
        except ValueError:
            out["wreck_auroc"] = 0.0
    else:
        out["wreck_auroc"] = 0.0

    miss = is_missense & np.isfinite(z)
    out["missense_spearman_vs_z"] = _spearman(pred[miss], z[miss]) if miss.any() else 0.0
    # Higher LoF score should track more negative z.
    out["missense_spearman"] = _spearman(pred[miss], -z[miss]) if miss.any() else 0.0

    strong = miss & (z <= -2.0)
    wt = miss & (np.abs(z) <= 1.0)
    if strong.any() and wt.any():
        y = np.concatenate([np.ones(int(strong.sum())), np.zeros(int(wt.sum()))])
        s = np.concatenate([pred[strong], pred[wt]])
        try:
            out["strong_vs_wt_auroc"] = float(roc_auc_score(y, s))
        except ValueError:
            out["strong_vs_wt_auroc"] = 0.0
    else:
        out["strong_vs_wt_auroc"] = 0.0

    weak = miss & (z < -1.0) & (z > -2.0)
    if weak.any() and wt.any():
        y = np.concatenate([np.ones(int(weak.sum())), np.zeros(int(wt.sum()))])
        s = np.concatenate([pred[weak], pred[wt]])
        try:
            out["weak_vs_wt_auroc"] = float(roc_auc_score(y, s))
        except ValueError:
            out["weak_vs_wt_auroc"] = 0.0
    else:
        out["weak_vs_wt_auroc"] = 0.0

    # Gene-prior collapse: if wreck AUROC is high but missense Spearman is ~0,
    # the head is cheating on length. Surface that as a combined score.
    spearman01 = (out["missense_spearman"] + 1.0) / 2.0
    if miss.any() and is_wreck.any():
        out["selection"] = 0.5 * out["wreck_auroc"] + 0.5 * spearman01
    elif miss.any():
        out["selection"] = spearman01
    else:
        out["selection"] = out["wreck_auroc"]
    return out


def gof_metrics(
    prob: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.90,
) -> dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(target, dtype=np.int32)
    out: dict[str, float] = {"n": float(prob.size), "n_pos": float(y.sum())}
    if len(np.unique(y)) < 2:
        out["auroc"] = 0.0
        out["auprc"] = 0.0
    else:
        out["auroc"] = float(roc_auc_score(y, prob))
        out["auprc"] = float(average_precision_score(y, prob))

    called = prob >= threshold
    n_call = int(called.sum())
    out["n_calls"] = float(n_call)
    if n_call == 0:
        out["precision_at_thr"] = 0.0
        out["recall_at_thr"] = 0.0
    else:
        tp = int((called & (y == 1)).sum())
        out["precision_at_thr"] = tp / n_call
        out["recall_at_thr"] = tp / max(int(y.sum()), 1)

    out["selection"] = out["auroc"]
    return out


def gene_prior_collapse(pred: np.ndarray, gene: list[str], z: np.ndarray | None = None) -> dict[str, float]:
    """Flag heads that emit an almost-constant score per gene."""
    pred = np.asarray(pred, dtype=np.float64)
    genes = np.asarray(gene)
    collapsed = 0
    n_genes = 0
    for g in np.unique(genes):
        mask = genes == g
        if mask.sum() < 8:
            continue
        n_genes += 1
        if np.std(pred[mask]) < 0.05:
            if z is None or (np.isfinite(z[mask]).sum() >= 8 and np.nanstd(z[mask]) > 0.5):
                collapsed += 1
    return {
        "n_genes_scored": float(n_genes),
        "n_genes_collapsed": float(collapsed),
        "collapse_fraction": float(collapsed / n_genes) if n_genes else 0.0,
    }
