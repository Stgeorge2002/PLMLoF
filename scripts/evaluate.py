"""Evaluate an ensemble on cached test embeddings.

    python scripts/evaluate.py --task lof \
        --ensemble-dir outputs/lof \
        --embeddings $PLMLOF_EMB_DIR/lof/test_embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from plmlof.dataset import CachedDataset
from plmlof.metrics import (
    collapse_is_fail,
    gene_prior_collapse,
    gof_metrics,
    lof_metrics,
    mlof_metrics,
    within_gene_auroc,
)
from plmlof.predictor import _load_net, discover_ensemble
from plmlof.stats import empirical_p

logger = logging.getLogger(__name__)


def _str_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(v) for v in list(value)]


def _slice_mlof(
    pred: np.ndarray,
    packed: dict,
    genes: list[str],
    pids: list[str],
    labels: list[str],
    prefix: str,
    min_n: int = 32,
) -> dict[str, float]:
    """Within-gene Spearman inside a taxon or assay slice (diagnostic, not selection)."""
    if not labels or len(labels) != len(pred):
        return {}
    out: dict[str, float] = {}
    tags = np.array([str(t).strip().lower() for t in labels])
    for name in sorted(set(tags.tolist()) - {""}):
        mask = tags == name
        if int(mask.sum()) < min_n:
            continue
        idx = np.flatnonzero(mask)
        sub = mlof_metrics(
            pred[idx], packed["target"][idx], packed["z"][idx], packed["missense"][idx],
            gene=[genes[i] for i in idx],
            protein_id=[pids[i] for i in idx],
        )
        key = name.replace(" ", "_")
        out[f"{prefix}_{key}_within_gene_spearman"] = sub["within_gene_spearman"]
        out[f"{prefix}_{key}_n"] = sub["n"]
        out[f"{prefix}_{key}_n_genes_ranked"] = sub["n_genes_ranked"]
    return out


def ensemble_predict(nets, loader, device) -> tuple[np.ndarray, np.ndarray, dict]:
    means, sds = [], []
    extra: dict[str, list] = {
        "target": [], "z": [], "wreck": [], "missense": [], "gene": [], "protein_id": [],
        "taxon": [], "assay": [],
    }
    with torch.no_grad():
        for batch in loader:
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            parts = []
            for net in nets:
                raw = net.forward_from_cache(
                    tensors["ref_mean"], tensors["ref_max"],
                    tensors["var_mean"], tensors["var_max"],
                    tensors["nucleotide_features"],
                    site_ref=tensors.get("site_ref"),
                    site_var=tensors.get("site_var"),
                    site_chem=tensors.get("site_chem"),
                    site_ref2=tensors.get("site_ref2"),
                    site_var2=tensors.get("site_var2"),
                    site_chem2=tensors.get("site_chem2"),
                    n_sites=tensors.get("n_sites"),
                )
                parts.append(net.probability(raw).float().cpu())
            stacked = torch.stack(parts, dim=0)
            means.append(stacked.mean(0).numpy())
            sds.append(stacked.std(0).numpy())
            extra["target"].append(batch["target"].numpy())
            extra["z"].append(batch["dms_zscore"].numpy())
            extra["wreck"].append(batch["is_wreck"].numpy())
            extra["missense"].append(batch["is_missense"].numpy())
            extra["gene"].extend(_str_list(batch.get("gene", [])))
            extra["protein_id"].extend(_str_list(batch.get("protein_id", [])))
            extra["taxon"].extend(_str_list(batch.get("taxon", [])))
            extra["assay"].extend(_str_list(batch.get("assay", [])))
    pred = np.concatenate(means)
    sd = np.concatenate(sds)
    packed = {
        k: np.concatenate(v)
        for k, v in extra.items()
        if v and k not in {"gene", "protein_id", "taxon", "assay"}
    }
    packed["gene"] = extra["gene"]
    packed["protein_id"] = extra["protein_id"]
    packed["taxon"] = extra["taxon"]
    packed["assay"] = extra["assay"]
    return pred, sd, packed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["lof", "mlof", "growth_gof", "amr_gof"])
    p.add_argument("--ensemble-dir", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gof-threshold", type=float, default=0.90)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    members = discover_ensemble(args.ensemble_dir)
    if not members:
        raise SystemExit(f"No ensemble in {args.ensemble_dir}")
    nets = [_load_net(m, device) for m in members]
    ds = CachedDataset(args.embeddings)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    pred, sd, packed = ensemble_predict(nets, loader, device)
    genes = packed["gene"] if packed["gene"] else list(ds.genes)
    pids = packed["protein_id"] if packed["protein_id"] else list(ds.protein_ids)
    groups = pids if any(pids) else genes

    print("=" * 60)
    print(f"{args.task}  n={len(pred)}  members={len(nets)}  mean_sd={float(sd.mean()):.4f}")
    if args.task == "mlof":
        metrics = mlof_metrics(
            pred, packed["target"], packed["z"], packed["missense"],
            gene=genes, protein_id=pids,
        )
        taxa = packed.get("taxon") or list(ds.taxon)
        assays = packed.get("assay") or list(ds.assay)
        metrics.update(_slice_mlof(pred, packed, genes, pids, taxa, "taxon"))
        metrics.update(_slice_mlof(pred, packed, genes, pids, assays, "assay"))
        if hasattr(ds, "n_sites"):
            metrics["n_multi"] = float((ds.n_sites >= 2).sum().item())
    elif args.task == "lof":
        metrics = lof_metrics(pred, packed["target"], packed["z"], packed["wreck"], packed["missense"])
        metrics.update(gene_prior_collapse(pred, groups, packed["target"]))
    else:
        metrics = gof_metrics(pred, packed["target"], threshold=args.gof_threshold, gene=genes)
        if "within_gene_auroc" not in metrics:
            metrics.update(within_gene_auroc(pred, packed["target"], genes))
        metrics.update(gene_prior_collapse(pred, genes, packed["target"]))
    for k, v in metrics.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if collapse_is_fail(metrics):
        print("FAIL: gene-prior collapse (most genes have near-constant scores).")
    if args.task in {"growth_gof", "amr_gof"} and float(metrics.get("caller_ready", 0)) < 1:
        print("NOTE: GoF is not caller-ready (need ≥3 genes with within-gene AUROC ≥ 0.60).")

    null_path = args.ensemble_dir / "null_scores.pt"
    if null_path.exists():
        null = torch.load(null_path, map_location="cpu", weights_only=False)["scores"].numpy()
        pval = empirical_p(pred, null)
        print(f"  empirical p  median={np.nanmedian(pval):.4f}  p<0.05={float(np.mean(pval < 0.05)):.3f}")
        metrics["p_median"] = float(np.nanmedian(pval))
        metrics["frac_p_lt_0.05"] = float(np.mean(pval < 0.05))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))
        logger.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    main()
