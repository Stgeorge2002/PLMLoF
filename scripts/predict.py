"""Inference entry point for PLMLoF.

    python scripts/predict.py --model outputs --reference ref.fasta --variants var.fasta
    python scripts/predict.py --model outputs --proteins alleles.faa
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
    parser.add_argument("--reference", type=str, default=None, help="Reference FASTA (CDS or protein)")
    parser.add_argument("--variants", type=str, default=None, help="Variant FASTA file")
    parser.add_argument("--proteins", type=str, default=None, help="Alignment-free LoF: protein FASTA, no reference")
    parser.add_argument("--vcf", type=str, default=None, help="VCF file")
    parser.add_argument("--model", type=str, required=True, help="Output dir with lof/ mlof/ growth_gof/ amr_gof/")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="Output file (JSON or TSV)")
    parser.add_argument("--format", type=str, choices=["json", "tsv"], default="tsv")
    parser.add_argument(
        "--domains", type=str, default=None,
        help="pfam_domains.parquet (protein_id, start, end). Grades wrecks; last-10% tail if omitted.",
    )
    parser.add_argument("--hmm", type=str, default=None, help="Pfam-A.hmm for refs missing from --domains")
    parser.add_argument("--hmm-cpus", type=int, default=1)
    return parser.parse_args()


def _write_tsv(results: list[dict], path: Path) -> None:
    keys = [
        "gene", "wreck", "wreck_kind", "wreck_prior",
        "lof_score", "lof_sd", "lof_p", "lof_q",
        "mlof_score", "mlof_bin", "mlof_mean", "mlof_n_sites", "mlof_sd", "mlof_p", "mlof_q",
        "in_family", "nearest_train_gene", "ref_cosine",
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
        if v != v:
            return ""
        return f"{v:.6g}"
    return str(v)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.proteins is None and args.variants is None and args.vcf is None:
        logger.error("Provide --proteins, --variants, or --vcf")
        sys.exit(1)
    if args.proteins is None and args.reference is None:
        logger.error("Pair mode needs --reference (or use --proteins for alignment-free LoF)")
        sys.exit(1)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))

    from plmlof.inference.vcf_handler import load_reference_dna, parse_vcf_variants
    from plmlof.predictor import Predictor

    model_path = Path(args.model)
    pred_dir = model_path if model_path.is_dir() else model_path.parent.parent.parent
    pred = Predictor(
        pred_dir, device=device, batch_size=args.batch_size,
        domains=args.domains, hmm=args.hmm, hmm_cpus=args.hmm_cpus,
    )
    if args.proteins:
        results = pred.predict_proteins(args.proteins)
    elif args.vcf:
        ref_dna = load_reference_dna(args.reference)
        records = parse_vcf_variants(args.vcf, ref_dna)
        results = pred.predict_records(records, pair=True)
    else:
        results = pred.predict_fasta(args.reference, args.variants)

    if args.output:
        out = Path(args.output)
        if args.format == "json":
            out.write_text(json.dumps(results, indent=2, default=str))
        else:
            _write_tsv(results, out)
        logger.info("Wrote %s (%s rows)", out, len(results))
        return

    print(f"{'gene':<24} {'lof':>7} {'mlof':>7} {'wreck':<12} {'gGof':>6} {'amr':>6}")
    for r in results[:50]:
        print(
            f"{r['gene']:<24} {r['lof_score']:7.3f} {r.get('mlof_score', float('nan')):7.3f} "
            f"{r['wreck_kind']:<12} {r.get('growth_gof_p', float('nan')):6.3f} "
            f"{r.get('amr_gof_p', float('nan')):6.3f}"
        )
    print(f"Total {len(results)} pairs")


if __name__ == "__main__":
    main()
