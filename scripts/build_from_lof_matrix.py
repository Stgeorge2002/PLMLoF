#!/usr/bin/env python3
"""Main PLMLoF import: MutCode codes + Panaroo → per-gene ref/var CDS FASTA pairs.

Decode follows MutCode mutcode-v1 against ``pan_genome_reference.fa``:

    []              → allele is exactly the FASTA reference
    [[start,d,ins]] → apply edits on original reference coordinates

Outbreak LoF-matrix ``Reference_Length`` is annotation metadata only and is
not checked against the FASTA (it can omit a terminal stop). Native MutCode
``mutation_codes.tsv.gz`` tables verify SHA-256 hashes instead.

Usage (outbreak LoF matrix):

    python scripts/build_from_lof_matrix.py \\
        --lof-matrix N44_ST258/lof_matrix/lof_matrix.tsv \\
        --results-dir N44_ST258 \\
        --out-dir N44_ST258/plmlof_input

Usage (native MutCode table + one reference FASTA):

    python scripts/build_from_lof_matrix.py \\
        --mutcode-table mutation_codes.tsv.gz \\
        --reference pan_genome_reference.fa \\
        --out-dir plmlof_input
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from plmlof.lof_import import (
    LofImportError,
    build_from_lof_matrix,
    build_from_mutcode_table,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--lof-matrix",
        type=Path,
        help="Outbreak LoF matrix TSV/TSV.GZ with Mutation_Codes",
    )
    src.add_argument(
        "--mutcode-table",
        type=Path,
        help="Native MutCode mutation_codes.tsv[.gz]",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Outbreak root with panaroo/{chromosome,plasmids}/... (LoF matrix mode)",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Exact pan_genome_reference.fa used at MutCode encode (native mode)",
    )
    p.add_argument(
        "--replicon",
        type=str,
        default="chromosome",
        help="Output category for native mode (default: chromosome)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <results-dir>/plmlof_input or ./plmlof_input",
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
        help="fail when LoF-matrix Sample_Length ≠ MutCode DNA length",
    )
    p.add_argument(
        "--skip-missing-genes",
        action="store_true",
        help="skip rows whose Gene is absent from pan_genome_reference.fa",
    )
    p.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help="native mode: skip reference/target SHA-256 checks",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)

    try:
        if args.lof_matrix is not None:
            if args.results_dir is None:
                raise SystemExit("--results-dir is required with --lof-matrix")
            if not args.lof_matrix.is_file():
                raise SystemExit(f"LoF matrix not found: {args.lof_matrix}")
            if not args.results_dir.is_dir():
                raise SystemExit(f"results dir not found: {args.results_dir}")
            out_dir = args.out_dir if args.out_dir else args.results_dir / "plmlof_input"
            build_from_lof_matrix(
                lof_matrix=args.lof_matrix,
                results_dir=args.results_dir,
                out_dir=out_dir,
                strict_sample_length=args.strict_sample_length,
                skip_missing_genes=args.skip_missing_genes,
                min_samples=args.min_samples,
            )
        else:
            if args.reference is None:
                raise SystemExit("--reference is required with --mutcode-table")
            if not args.mutcode_table.is_file():
                raise SystemExit(f"MutCode table not found: {args.mutcode_table}")
            out_dir = args.out_dir if args.out_dir else Path("plmlof_input")
            build_from_mutcode_table(
                codes=args.mutcode_table,
                reference_fa=args.reference,
                out_dir=out_dir,
                replicon=args.replicon,
                min_samples=args.min_samples,
                verify_hashes=not args.no_verify_hashes,
            )
    except (LofImportError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main(sys.argv[1:])
