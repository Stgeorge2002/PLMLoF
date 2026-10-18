"""Validation metrics: wreck AUROC for LoF."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _strong_weak_auroc(pred: np.ndarray, z: np.ndarray, miss: np.ndarray) -> dict[str, float]:
    out = {"strong_vs_wt_auroc": 0.0, "weak_vs_wt_auroc": 0.0}
    strong = miss & (z <= -2.0)
    wt = miss & (np.abs(z) <= 1.0)
    weak = miss & (z < -1.0) & (z > -2.0)
    if strong.any() and wt.any():
        y = np.concatenate([np.ones(int(strong.sum())), np.zeros(int(wt.sum()))])
        s = np.concatenate([pred[strong], pred[wt]])
        try:
            out["strong_vs_wt_auroc"] = float(roc_auc_score(y, s))
        except ValueError:
            pass
    if weak.any() and wt.any():
        y = np.concatenate([np.ones(int(weak.sum())), np.zeros(int(wt.sum()))])
        s = np.concatenate([pred[weak], pred[wt]])
        try:
            out["weak_vs_wt_auroc"] = float(roc_auc_score(y, s))
        except ValueError:
            pass
    return out


def lof_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    z: np.ndarray,
    is_wreck: np.ndarray,
    is_missense: np.ndarray,
) -> dict[str, float]:
    """LoF is a wreck scanner. Checkpoint on wreck AUROC, not missense rank."""
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
    out.update(_strong_weak_auroc(pred, z, miss))
    out["selection"] = out["wreck_auroc"]
    return out
