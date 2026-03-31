"""Benchmark PLMLoF predictions against SNPEff annotations.

Usage:
    python scripts/benchmark_vs_snpeff.py \
        --plmlof-results results.json \
        --snpeff-vcf annotated.vcf \
        --output benchmark_report.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# SNPEff impact → PLMLoF label mapping
SNPEFF_TO_LABEL = {
    # HIGH impact → likely LoF
    "stop_gained": "LoF",
    "frameshift_variant": "LoF",
    "stop_lost": "GoF",
    "start_lost": "LoF",
    "splice_acceptor_variant": "LoF",
    "splice_donor_variant": "LoF",
    # MODERATE impact → context dependent
    "missense_variant": "WT",  # Default to WT; could be LoF or GoF
    "inframe_deletion": "WT",
    "inframe_insertion": "WT",
    # LOW impact → WT
    "synonymous_variant": "WT",
    "stop_retained_variant": "WT",
    # MODIFIER → WT
    "upstream_gene_variant": "WT",
    "downstream_gene_variant": "WT",
    "intergenic_region": "WT",
}


def parse_snpeff_vcf(vcf_path: str | Path) -> pd.DataFrame:
    """Parse SNPEff-annotated VCF and extract per-gene effect predictions.

    Returns DataFrame with columns: gene, snpeff_effect, snpeff_impact, snpeff_label.
    """
    records = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            info = parts[7]
            # Parse ANN field from SNPEff
            ann_entries = []
            for field in info.split(";"):
                if field.startswith("ANN="):
                    ann_entries = field[4:].split(",")
                    break

            for ann in ann_entries:
                ann_parts = ann.split("|")
                if len(ann_parts) < 11:
                    continue

                effect = ann_parts[1]
                impact = ann_parts[2]
                gene = ann_parts[3]

                label = SNPEFF_TO_LABEL.get(effect, "WT")

                records.append({
                    "gene": gene,
                    "snpeff_effect": effect,
                    "snpeff_impact": impact,
                    "snpeff_label": label,
                })

    df = pd.DataFrame(records)
    # Deduplicate per gene — take highest impact
    impact_order = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0}
    if not df.empty:
        df["impact_score"] = df["snpeff_impact"].map(impact_order).fillna(0)
        df = df.sort_values("impact_score", ascending=False).drop_duplicates("gene", keep="first")
        df = df.drop("impact_score", axis=1)

    return df


def compare_predictions(
    plmlof_results: list[dict],
    snpeff_df: pd.DataFrame,
) -> dict:
    """Compare PLMLoF predictions against SNPEff annotations.

    Returns concordance report.
    """
    plmlof_df = pd.DataFrame(plmlof_results)[["gene", "prediction"]]
    plmlof_df = plmlof_df.rename(columns={"prediction": "plmlof_label"})

    merged = pd.merge(plmlof_df, snpeff_df, on="gene", how="inner")

    if merged.empty:
        return {"error": "No overlapping genes found"}

    # Compute agreement
    agree = (merged["plmlof_label"] == merged["snpeff_label"]).sum()
    total = len(merged)
    agreement_rate = agree / total if total > 0 else 0

    # Per-class concordance
    report = {
        "total_genes": total,
        "agreement_rate": float(agreement_rate),
        "agree_count": int(agree),
        "disagree_count": int(total - agree),
    }

    for label in ["LoF", "WT", "GoF"]:
        mask = merged["snpeff_label"] == label
        if mask.sum() > 0:
            label_agree = ((merged["plmlof_label"] == label) & mask).sum()
            report[f"{label}_concordance"] = float(label_agree / mask.sum())
        else:
            report[f"{label}_concordance"] = None

    # Disagreements
    disagreements = merged[merged["plmlof_label"] != merged["snpeff_label"]]
    report["disagreements"] = disagreements[["gene", "plmlof_label", "snpeff_label", "snpeff_effect"]].to_dict("records")

    return report


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Benchmark PLMLoF vs SNPEff")
    parser.add_argument("--plmlof-results", type=str, required=True, help="PLMLoF results JSON")
    parser.add_argument("--snpeff-vcf", type=str, required=True, help="SNPEff annotated VCF")
    parser.add_argument("--output", type=str, default=None, help="Output report file")
    args = parser.parse_args()

    with open(args.plmlof_results) as f:
        plmlof_results = json.load(f)

    snpeff_df = parse_snpeff_vcf(args.snpeff_vcf)
    report = compare_predictions(plmlof_results, snpeff_df)

    print("\n" + "=" * 60)
    print("BENCHMARK: PLMLoF vs SNPEff")
    print("=" * 60)
    print(f"Total overlapping genes: {report.get('total_genes', 0)}")
    print(f"Agreement rate:          {report.get('agreement_rate', 0):.1%}")
    print(f"LoF concordance:         {report.get('LoF_concordance', 'N/A')}")
    print(f"WT concordance:          {report.get('WT_concordance', 'N/A')}")
    print(f"GoF concordance:         {report.get('GoF_concordance', 'N/A')}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
