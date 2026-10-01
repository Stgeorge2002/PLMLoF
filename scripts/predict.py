"""Inference entry point for PLMLoF.

v2 (graded LoF + GoF flags):
    python scripts/predict.py --model outputs/v2 --reference ref.fasta --variants var.fasta

Legacy 3-class checkpoint:
    python scripts/predict.py --model outputs/production/checkpoints/model_best.pt \\
        --reference ref.fasta --variants var.fasta
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLMLoF variant effect prediction")
    parser.add_argument("--reference", type=str, required=True, help="Reference FASTA (CDS or protein)")
    parser.add_argument("--variants", type=str, default=None, help="Variant FASTA file")
    parser.add_argument("--vcf", type=str, default=None, help="VCF file")
    parser.add_argument("--model", type=str, default=None, help="v2 output dir or legacy .pt checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="Output file (JSON or TSV)")
    parser.add_argument("--format", type=str, choices=["json", "tsv"], default="tsv")
    parser.add_argument("--no-attribution", action="store_true", help="Skip attribution (legacy 3-class only)")
    parser.add_argument("--tiny", action="store_true", help="Use tiny ESM2 model (legacy 3-class, no checkpoint)")
    return parser.parse_args()


def _is_v2(path: Path) -> bool:
    if path.is_dir():
        return (path / "lof").is_dir() or (path / "ensemble.json").exists() or (path / "lof" / "ensemble.json").exists()
    if path.is_file() and path.suffix == ".pt":
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return bool(ckpt.get("v2"))
        except Exception:
            return False
    return False


def _write_v2_tsv(results: list[dict], path: Path) -> None:
    keys = [
        "gene", "wreck", "wreck_kind",
        "lof_score", "lof_sd", "lof_p", "lof_q", "in_family", "nearest_train_gene", "ref_cosine",
        "growth_gof_p", "growth_gof_sd", "growth_gof_p_emp", "growth_gof_q", "growth_gof_call",
        "growth_gof_in_family", "growth_gof_nearest",
        "amr_gof_p", "amr_gof_sd", "amr_gof_p_emp", "amr_gof_q", "amr_gof_call",
        "amr_gof_in_family", "amr_gof_nearest",
    ]
    with open(path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in results:
            f.write("\t".join(_fmt(r.get(k, "")) for k in keys) + "\n")


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if v != v:  # nan
            return ""
        return f"{v:.6g}"
    return str(v)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.variants is None and args.vcf is None:
        logger.error("Provide either --variants (FASTA) or --vcf")
        sys.exit(1)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))

    if args.tiny:
        from plmlof.models.plmlof_model import PLMLoFModel
        from plmlof.inference.predictor import PLMLoFPredictor
        model = PLMLoFModel(esm2_model_name="facebook/esm2_t6_8M_UR50D", freeze_esm2=True)
        predictor = PLMLoFPredictor(model=model, device=device, batch_size=args.batch_size)
        _run_legacy(predictor, args)
        return

    if not args.model:
        logger.error("Provide --model (outputs/v2) or --tiny")
        sys.exit(1)

    model_path = Path(args.model)
    if _is_v2(model_path):
        from plmlof.v2.predictor import V2Predictor
        from plmlof.inference.vcf_handler import load_reference_dna, parse_vcf_variants

        v2_dir = model_path if model_path.is_dir() else model_path.parent.parent.parent
        pred = V2Predictor(v2_dir, device=device, batch_size=args.batch_size)
        if args.vcf:
            ref_dna = load_reference_dna(args.reference)
            records = parse_vcf_variants(args.vcf, ref_dna)
            results = pred.predict_records(records)
        else:
            results = pred.predict_fasta(args.reference, args.variants)
        if args.output:
            out = Path(args.output)
            if args.format == "json":
                out.write_text(json.dumps(results, indent=2, default=str))
            else:
                _write_v2_tsv(results, out)
            logger.info("Wrote %s (%s rows)", out, len(results))
        else:
            print(f"{'gene':<24} {'lof':>7} {'±':>6} {'wreck':<12} {'gGof':>6} {'amr':>6}")
            for r in results[:50]:
                print(
                    f"{r['gene']:<24} {r['lof_score']:7.3f} {r['lof_sd']:6.3f} "
                    f"{r['wreck_kind']:<12} {r.get('growth_gof_p', float('nan')):6.3f} "
                    f"{r.get('amr_gof_p', float('nan')):6.3f}"
                )
            print(f"Total {len(results)} pairs")
        return

    from plmlof.inference.predictor import PLMLoFPredictor
    predictor = PLMLoFPredictor(model_path=args.model, device=device, batch_size=args.batch_size)
    _run_legacy(predictor, args)


def _run_legacy(predictor, args) -> None:
    predictor.load_reference(args.reference)
    compute_attr = not args.no_attribution
    if args.vcf:
        results = predictor.predict_vcf(args.vcf, compute_attribution=compute_attr)
    else:
        results = predictor.predict_fasta(
            args.variants, reference_fasta=args.reference, compute_attribution=compute_attr,
        )
    if args.output:
        output_path = Path(args.output)
        if args.format == "json":
            output_path.write_text(json.dumps(results, indent=2, default=str))
        else:
            with open(output_path, "w") as f:
                f.write("gene\tprediction\tconfidence\tLoF_prob\tWT_prob\tGoF_prob\tsummary\n")
                for r in results:
                    probs = r.get("probabilities", {})
                    summary = r.get("attribution", {}).get("summary", "") if "attribution" in r else ""
                    f.write(
                        f"{r['gene']}\t{r['prediction']}\t{r['confidence']:.4f}\t"
                        f"{probs.get('LoF', 0):.4f}\t{probs.get('WT', 0):.4f}\t"
                        f"{probs.get('GoF', 0):.4f}\t{summary}\n"
                    )
        logger.info("Results written to %s", args.output)
    else:
        print(f"\n{'Gene':<25} {'Prediction':<12} {'Confidence':<12}")
        print("-" * 60)
        for r in results:
            print(f"{r['gene']:<25} {r['prediction']:<12} {r['confidence']:<12.4f}")
        print(f"\nTotal: {len(results)} genes")


if __name__ == "__main__":
    main()
