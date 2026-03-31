"""Inference entry point for PLMLoF predictions.

Usage:
    # From paired FASTA files
    python scripts/predict.py --reference ref.fasta --variants var.fasta --model outputs/checkpoints/model_best.pt

    # From VCF + reference
    python scripts/predict.py --reference ref.fasta --vcf variants.vcf --model outputs/checkpoints/model_best.pt

    # Quick test with tiny model (no trained checkpoint needed)
    python scripts/predict.py --reference tests/fixtures/ref.fasta --variants tests/fixtures/var.fasta --tiny
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from plmlof.models.plmlof_model import PLMLoFModel
from plmlof.inference.predictor import PLMLoFPredictor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLMLoF variant effect prediction")
    parser.add_argument("--reference", type=str, required=True, help="Reference FASTA (gene CDS)")
    parser.add_argument("--variants", type=str, default=None, help="Variant FASTA file")
    parser.add_argument("--vcf", type=str, default=None, help="VCF file")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="Output file (JSON or TSV)")
    parser.add_argument("--format", type=str, choices=["json", "tsv"], default="tsv")
    parser.add_argument("--no-attribution", action="store_true", help="Skip attribution")
    parser.add_argument("--tiny", action="store_true", help="Use tiny ESM2 model (no checkpoint)")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.variants is None and args.vcf is None:
        logger.error("Provide either --variants (FASTA) or --vcf (VCF file)")
        sys.exit(1)

    # Auto-detect device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Build or load model
    if args.tiny:
        esm2_name = "facebook/esm2_t6_8M_UR50D"
        model = PLMLoFModel(esm2_model_name=esm2_name, freeze_esm2=True)
        predictor = PLMLoFPredictor(
            model=model,
            device=device,
            batch_size=args.batch_size,
        )
    elif args.model:
        predictor = PLMLoFPredictor(
            model_path=args.model,
            device=device,
            batch_size=args.batch_size,
        )
    else:
        logger.error("Provide --model or use --tiny for testing")
        sys.exit(1)

    # Load reference
    predictor.load_reference(args.reference)

    # Run prediction
    compute_attr = not args.no_attribution
    if args.vcf:
        results = predictor.predict_vcf(args.vcf, compute_attribution=compute_attr)
    else:
        results = predictor.predict_fasta(
            args.variants,
            reference_fasta=args.reference,
            compute_attribution=compute_attr,
        )

    # Output results
    if args.output:
        output_path = Path(args.output)
        if args.format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        else:
            _write_tsv(results, output_path)
        logger.info(f"Results written to {output_path}")
    else:
        # Print to stdout
        _print_results(results)


def _print_results(results: list[dict]) -> None:
    """Pretty-print results to stdout."""
    print(f"\n{'Gene':<25} {'Prediction':<12} {'Confidence':<12} {'Summary'}")
    print("-" * 90)
    for r in results:
        summary = ""
        if "attribution" in r:
            summary = r["attribution"].get("summary", "")
        print(f"{r['gene']:<25} {r['prediction']:<12} {r['confidence']:<12.4f} {summary}")
    print(f"\nTotal: {len(results)} genes analyzed")


def _write_tsv(results: list[dict], path: Path) -> None:
    """Write results as TSV."""
    with open(path, "w") as f:
        f.write("gene\tprediction\tconfidence\tLoF_prob\tWT_prob\tGoF_prob\tsummary\n")
        for r in results:
            probs = r.get("probabilities", {})
            summary = r.get("attribution", {}).get("summary", "") if "attribution" in r else ""
            f.write(
                f"{r['gene']}\t{r['prediction']}\t{r['confidence']:.4f}\t"
                f"{probs.get('LoF', 0):.4f}\t{probs.get('WT', 0):.4f}\t"
                f"{probs.get('GoF', 0):.4f}\t{summary}\n"
            )


if __name__ == "__main__":
    main()
