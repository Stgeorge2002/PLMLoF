"""Compare FP16 8M student scores vs INT8/FP8 weight-only quantized encoder.

Scores the same FASTA pairs (or a random subset) twice and reports Spearman
correlation of lof_score plus wreck AUROC if a binary wreck column is present
in an optional labels TSV (gene\\twreck).

    python scripts/validate_quantize.py \\
        --model outputs/lof_8m \\
        --reference ref.fasta --variants var.fasta \\
        --quantize int8 --max-rows 400 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(y_score) & np.isfinite(y_true)
    yt, ys = y_true[mask], y_score[mask]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--variants", required=True)
    p.add_argument("--quantize", choices=["int8", "fp8"], required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-rows", type=int, default=400)
    p.add_argument("--labels", type=str, default=None, help="Optional TSV: gene\\twreck (0/1)")
    p.add_argument("--max-ensemble-members", type=int, default=1)
    args = p.parse_args()

    import torch

    from plmlof.inference.vcf_handler import parse_fasta_pairs
    from plmlof.predict_args import resolve_model_dir
    from plmlof.predictor import Predictor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = parse_fasta_pairs(args.reference, args.variants)[: args.max_rows]
    if not records:
        logger.error("No records")
        sys.exit(1)

    common = dict(
        model_dir=resolve_model_dir(args.model),
        device=device,
        batch_size=args.batch_size,
        max_ensemble_members=args.max_ensemble_members,
        compile_encoder=False,
        rule_short_circuit=False,
        lof_only=True,
    )
    logger.info("Scoring FP16 baseline (%s pairs)…", len(records))
    base = Predictor(**common, quantize=None)
    r_base = base.predict_records(records, pair=True)
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info("Scoring quantize=%s…", args.quantize)
    quant = Predictor(**common, quantize=args.quantize)
    r_q = quant.predict_records(records, pair=True)

    s0 = np.array([r["lof_score"] for r in r_base], dtype=np.float64)
    s1 = np.array([r["lof_score"] for r in r_q], dtype=np.float64)
    rho = _spearman(s0, s1)
    mae = float(np.nanmean(np.abs(s0 - s1)))
    logger.info("lof_score Spearman=%.4f  MAE=%.5f  n=%s", rho, mae, len(s0))

    wreck = np.array([float(bool(r["wreck"])) for r in r_base], dtype=np.float64)
    if args.labels:
        lab = {}
        for line in Path(args.labels).read_text().splitlines():
            if not line.strip() or line.startswith("gene"):
                continue
            g, w = line.split("\t")[:2]
            lab[g] = float(w)
        wreck = np.array([lab.get(r["gene"], np.nan) for r in r_base], dtype=np.float64)

    auc0 = _auroc(wreck, s0)
    auc1 = _auroc(wreck, s1)
    logger.info("wreck AUROC  fp16=%.4f  %s=%.4f  delta=%+.4f", auc0, args.quantize, auc1, auc1 - auc0)

    # Soft gate for ops: warn if correlation collapses.
    if np.isfinite(rho) and rho < 0.95:
        logger.warning("Spearman < 0.95 — inspect before shipping quantized encoder")
        sys.exit(2)
    if np.isfinite(auc0) and np.isfinite(auc1) and abs(auc1 - auc0) > 0.02:
        logger.warning("|Δ AUROC| > 0.02 — inspect before shipping quantized encoder")
        sys.exit(2)
    print(f"OK spearman={rho:.4f} mae={mae:.5f} auroc_fp16={auc0:.4f} auroc_q={auc1:.4f}")


if __name__ == "__main__":
    main()
