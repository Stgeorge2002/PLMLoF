#!/usr/bin/env python3
"""Main PLMLoF import: LoF matrix + Panaroo → per-gene ref/var CDS FASTA pairs.

Reads a MutCode-backed LoF matrix (see ``N44_ST258/lof_matrix/lof_matrix.tsv``)
and the matching Panaroo ``pan_genome_reference.fa`` for each replicon, then
losslessly rebuilds every sample allele with MutCode's called-consensus rule.

Output layout matches ``scripts/run_predict_batch.py``:

    <out-dir>/chromosome/<gene>/{ref.fasta,var.fasta,manifest.tsv}
    <out-dir>/plasmid/<mge_N>/<gene>/{...}
    <out-dir>/manifest_all.tsv
    <out-dir>/build_stats.tsv

Usage:

    python scripts/build_from_lof_matrix.py \\
        --lof-matrix N44_ST258/lof_matrix/lof_matrix.tsv \\
        --results-dir N44_ST258 \\
        --out-dir N44_ST258/plmlof_input
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from plmlof.lof_import import LofImportError, build_from_lof_matrix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--lof-matrix",
        type=Path,
        required=True,
        help="LoF matrix TSV/TSV.GZ with Mutation_Codes (MutCode format)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Outbreak results root containing panaroo/{chromosome,plasmids}/...",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <results-dir>/plmlof_input",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="skip gene folders present in fewer samples than this",
    )
    p.add_argument(
        "--strict-sample-length",
        action="store_true",
        help="fail when Sample_Length ≠ MutCode reconstructed DNA length",
    )
    p.add_argument(
        "--skip-missing-genes",
        action="store_true",
        help="skip rows whose Gene is absent from pan_genome_reference.fa",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    out_dir = args.out_dir if args.out_dir else args.results_dir / "plmlof_input"

    if not args.lof_matrix.is_file():
        raise SystemExit(f"LoF matrix not found: {args.lof_matrix}")
    if not args.results_dir.is_dir():
        raise SystemExit(f"results dir not found: {args.results_dir}")

    try:
        build_from_lof_matrix(
            lof_matrix=args.lof_matrix,
            results_dir=args.results_dir,
            out_dir=out_dir,
            strict_sample_length=args.strict_sample_length,
            skip_missing_genes=args.skip_missing_genes,
            min_samples=args.min_samples,
        )
    except (LofImportError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main(sys.argv[1:])
