"""Held-out Dewachter fabZ/lpxC/murA exam (never used in training).

Expects paired FASTA plus a TSV with at least gene (or id) and a competition
coefficient column.

    python scripts/evaluate_dewachter.py \
        --model-dir outputs/v2 \
        --reference dewachter/fasta/ref.fasta \
        --variants  dewachter/fasta/var.fasta \
        --scores    dewachter/labels.tsv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from plmlof.v2.metrics import gene_prior_collapse
from plmlof.v2.predictor import V2Predictor

logger = logging.getLogger(__name__)


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if rho == rho else float("nan")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--scores", type=Path, default=None, help="TSV with gene/id + CompetitionCoefficient")
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = V2Predictor(args.model_dir, device=device)
    rows = pred.predict_fasta(args.reference, args.variants)
    df = pd.DataFrame(rows)

    if args.scores and args.scores.exists():
        lab = pd.read_csv(args.scores, sep="\t" if args.scores.suffix != ".csv" else ",")
        key = "gene" if "gene" in lab.columns else lab.columns[0]
        cc_col = next((c for c in lab.columns if "competition" in c.lower() or c.lower() in {"cc", "score", "fitness"}), None)
        if cc_col is None:
            raise SystemExit(f"No competition/score column in {args.scores}: {list(lab.columns)}")
        lab = lab.rename(columns={key: "gene", cc_col: "cc"})
        df = df.merge(lab[["gene", "cc"]], on="gene", how="left")
        rho = _spearman(df["lof_score"].to_numpy(), -df["cc"].to_numpy())
        rho_raw = _spearman(df["lof_score"].to_numpy(), df["cc"].to_numpy())
        print(f"Dewachter Spearman(lof_score, -CC) = {rho:.4f}")
        print(f"Dewachter Spearman(lof_score,  CC) = {rho_raw:.4f}")
        collapse = gene_prior_collapse(
            df["lof_score"].to_numpy(),
            df["gene"].astype(str).str.replace(r"_.*", "", regex=True).tolist(),
            df["cc"].to_numpy() if "cc" in df.columns else None,
        )
        print(f"collapse_fraction={collapse['collapse_fraction']:.3f}  (gene-prior fail if high)")
        if "growth_gof_call" in df.columns:
            print(f"growth GoF calls: {int(df['growth_gof_call'].sum())}  (should be ~0)")
        if "amr_gof_call" in df.columns:
            print(f"AMR GoF calls: {int(df['amr_gof_call'].sum())}")

    if args.output:
        df.to_csv(args.output, sep="\t", index=False)
        logger.info("Wrote %s", args.output)
    else:
        print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
