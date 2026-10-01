"""Evaluate a v2 ensemble on cached test embeddings.

    python scripts/evaluate_v2.py --task lof \
        --ensemble-dir outputs/v2/lof \
        --embeddings $PLMLOF_EMB_DIR/v2/lof/test_embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from plmlof.v2.dataset import V2CachedDataset
from plmlof.v2.metrics import gene_prior_collapse, gof_metrics, lof_metrics
from plmlof.v2.predictor import _load_net, discover_ensemble
from plmlof.v2.stats import empirical_p

logger = logging.getLogger(__name__)


def ensemble_predict(nets, loader, device) -> tuple[np.ndarray, np.ndarray, dict]:
    means, sds = [], []
    extra = {"target": [], "z": [], "wreck": [], "missense": [], "gene": []}
    with torch.no_grad():
        for batch in loader:
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            parts = []
            for net in nets:
                raw = net.forward_from_pooled(
                    tensors["ref_mean"], tensors["ref_max"],
                    tensors["var_mean"], tensors["var_max"],
                    tensors["nucleotide_features"],
                )
                parts.append(net.probability(raw).float().cpu())
            stacked = torch.stack(parts, dim=0)
            means.append(stacked.mean(0).numpy())
            sds.append(stacked.std(0).numpy())
            extra["target"].append(batch["target"].numpy())
            extra["z"].append(batch["dms_zscore"].numpy())
            extra["wreck"].append(batch["is_wreck"].numpy())
            extra["missense"].append(batch["is_missense"].numpy())
            if hasattr(loader.dataset, "genes"):
                # genes not in batch — index from dataset sequentially is wrong with shuffle=False + default sampler
                pass
    pred = np.concatenate(means)
    sd = np.concatenate(sds)
    packed = {k: np.concatenate(v) for k, v in extra.items() if v and k != "gene"}
    return pred, sd, packed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["lof", "growth_gof", "amr_gof"])
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
    ds = V2CachedDataset(args.embeddings)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    pred, sd, packed = ensemble_predict(nets, loader, device)
    genes = list(ds.genes)

    print("=" * 60)
    print(f"v2 {args.task}  n={len(pred)}  members={len(nets)}  mean_sd={float(sd.mean()):.4f}")
    if args.task == "lof":
        metrics = lof_metrics(pred, packed["target"], packed["z"], packed["wreck"], packed["missense"])
    else:
        metrics = gof_metrics(pred, packed["target"], threshold=args.gof_threshold)
    collapse = gene_prior_collapse(pred, genes, packed["z"])
    metrics.update(collapse)
    for k, v in metrics.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if collapse["collapse_fraction"] > 0.5:
        print("FAIL: gene-prior collapse (most genes have near-constant scores).")

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
