"""Validation metrics: wreck AUROC for LoF, within-gene ranking for MLoF."""

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


def _groups(gene: list[str] | None, protein_id: list[str] | None, n: int) -> np.ndarray:
    if protein_id is not None and len(protein_id) == n and any(protein_id):
        return np.asarray(protein_id, dtype=object)
    if gene is not None and len(gene) == n:
        return np.asarray(gene, dtype=object)
    return np.asarray([""] * n, dtype=object)


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
    out["missense_spearman_vs_z"] = _spearman(pred[miss], z[miss]) if miss.any() else 0.0
    out["missense_spearman"] = _spearman(pred[miss], -z[miss]) if miss.any() else 0.0
    out.update(_strong_weak_auroc(pred, z, miss))
    out["selection"] = out["wreck_auroc"]
    return out


def within_gene_spearman(
    pred: np.ndarray,
    z: np.ndarray,
    groups: np.ndarray,
    is_missense: np.ndarray,
    min_n: int = 8,
    min_z_std: float = 0.5,
) -> dict[str, float]:
    """Mean Spearman of damage vs -z inside proteins that actually vary."""
    pred = np.asarray(pred, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    miss = np.asarray(is_missense, dtype=bool)
    groups = np.asarray(groups)
    rhos: list[float] = []
    for g in np.unique(groups):
        mask = (groups == g) & miss & np.isfinite(z) & np.isfinite(pred)
        if int(mask.sum()) < min_n:
            continue
        if float(np.nanstd(z[mask])) < min_z_std:
            continue
        rho = _spearman(pred[mask], -z[mask])
        rhos.append(rho)
    return {
        "within_gene_spearman": float(np.mean(rhos)) if rhos else 0.0,
        "median_within_gene_spearman": float(np.median(rhos)) if rhos else 0.0,
        "n_genes_ranked": float(len(rhos)),
    }


def centered_spearman(
    pred: np.ndarray,
    z: np.ndarray,
    groups: np.ndarray,
    is_missense: np.ndarray,
) -> float:
    """Global Spearman after subtracting each protein's mean (kills gene priors)."""
    pred = np.asarray(pred, dtype=np.float64).copy()
    z = np.asarray(z, dtype=np.float64).copy()
    miss = np.asarray(is_missense, dtype=bool)
    groups = np.asarray(groups)
    keep = miss & np.isfinite(z) & np.isfinite(pred)
    if not keep.any():
        return 0.0
    pred_c = pred.copy()
    z_c = z.copy()
    for g in np.unique(groups):
        mask = keep & (groups == g)
        if int(mask.sum()) < 3:
            continue
        pred_c[mask] -= np.mean(pred[mask])
        z_c[mask] -= np.mean(z[mask])
    return _spearman(pred_c[keep], -z_c[keep])


def mlof_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    z: np.ndarray,
    is_missense: np.ndarray,
    *,
    gene: list[str] | None = None,
    protein_id: list[str] | None = None,
) -> dict[str, float]:
    """MLoF is a within-gene missense ranker. Checkpoint on within-gene Spearman."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    is_missense = np.asarray(is_missense, dtype=bool)
    groups = _groups(gene, protein_id, pred.size)
    miss = is_missense & np.isfinite(z)
    out: dict[str, float] = {
        "mae": float(np.mean(np.abs(pred - target))) if pred.size else 0.0,
        "n": float(pred.size),
        "missense_spearman": _spearman(pred[miss], -z[miss]) if miss.any() else 0.0,
        "missense_spearman_vs_z": _spearman(pred[miss], z[miss]) if miss.any() else 0.0,
        "wreck_auroc": 0.0,
    }
    out.update(_strong_weak_auroc(pred, z, miss))
    out.update(within_gene_spearman(pred, z, groups, is_missense))
    out["centered_spearman"] = centered_spearman(pred, z, groups, is_missense)
    collapse = gene_prior_collapse(pred, list(groups), z)
    out.update(collapse)
    # Collapse is a hard fail: a gene-prior model must not win the scoreboard.
    if collapse_is_fail(collapse) or out["n_genes_ranked"] < 1:
        out["selection"] = 0.0
    else:
        out["selection"] = out["within_gene_spearman"]
    return out


def gof_metrics(
    prob: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.90,
    gene: list[str] | None = None,
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

    if gene is not None:
        wg = within_gene_auroc(prob, y, gene)
        out.update(wg)
        if wg["n_genes_with_auroc"] >= 1:
            out["selection"] = wg["within_gene_auroc"]
        else:
            out["selection"] = out["auroc"]
        out["caller_ready"] = float(
            wg["n_genes_with_auroc"] >= 3 and wg["within_gene_auroc"] >= 0.60
        )
    else:
        out["selection"] = out["auroc"]
        out["caller_ready"] = 0.0
    return out


def within_gene_auroc(pred: np.ndarray, target: np.ndarray, gene: list[str]) -> dict[str, float]:
    """Mean AUROC inside each gene that has both classes. The honest GoF metric."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target)
    genes = np.asarray(gene)
    aucs: list[float] = []
    for g in np.unique(genes):
        mask = genes == g
        if int(mask.sum()) < 8:
            continue
        labels = y[mask]
        if len(np.unique(labels)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(labels, pred[mask])))
        except ValueError:
            continue
    return {
        "within_gene_auroc": float(np.mean(aucs)) if aucs else 0.0,
        "n_genes_with_auroc": float(len(aucs)),
    }


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


def collapse_is_fail(collapse: dict, min_genes: int = 3, fraction: float = 0.5) -> bool:
    """One held-out gene is not a collapse verdict."""
    return float(collapse.get("n_genes_scored", 0)) >= min_genes and float(collapse.get("collapse_fraction", 0)) > fraction
