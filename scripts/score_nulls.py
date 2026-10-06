"""Score held-out null pairs with a trained ensemble; write null_scores.pt."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plmlof.dataset import CachedDataset
from plmlof.predictor import discover_ensemble, _load_net

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["lof", "mlof", "growth_gof", "amr_gof"])
    p.add_argument("--ensemble-dir", type=Path, required=True, help="outputs/<task>")
    p.add_argument("--embeddings", type=Path, required=True, help="null_embeddings.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    members = discover_ensemble(args.ensemble_dir)
    if not members:
        raise SystemExit(f"No ensemble members in {args.ensemble_dir}")
    nets = [_load_net(m, device) for m in members]
    ds = CachedDataset(args.embeddings)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    acc = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            parts = []
            for net in nets:
                raw = net.forward_from_cache(
                    batch["ref_mean"], batch["ref_max"],
                    batch["var_mean"], batch["var_max"],
                    batch["nucleotide_features"],
                    site_ref=batch.get("site_ref"),
                    site_var=batch.get("site_var"),
                    site_chem=batch.get("site_chem"),
                )
                parts.append(net.probability(raw).float().cpu())
            acc.append(torch.stack(parts, dim=0).mean(0))
    scores = torch.cat(acc)
    out = args.ensemble_dir / "null_scores.pt"
    torch.save({"scores": scores, "n": int(scores.numel())}, out)
    logger.info("Wrote %s null scores → %s  mean=%.4f", scores.numel(), out, float(scores.mean()))

    # Copy a gallery from seed0 if the task dir does not have one
    gal = args.ensemble_dir / "train_gallery.pt"
    if not gal.exists():
        for m in members:
            cand = m.parent.parent / "train_gallery.pt"
            if cand.exists():
                gal.write_bytes(cand.read_bytes())
                logger.info("Copied gallery %s → %s", cand, gal)
                break


if __name__ == "__main__":
    main()
