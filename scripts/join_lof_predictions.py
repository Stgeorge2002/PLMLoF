#!/usr/bin/env python3
"""Left-join PLMLoF predictions onto the original LoF matrix.

Keeps every matrix column and row (including rows skipped at import). After
``Mutation_Codes`` appends only:

    LoF_score  continuous LoF head
    lof_p      empirical p vs WT-like null
    lof_q      BH-adjusted FDR
    mlof_score continuous MLoF head
    mlof_p     empirical p for MLoF
    mlof_q     BH-adjusted FDR for MLoF

Pass ``--extras-output`` to also write wreck / bin / diagnostics (and join
keys) to a sidecar TSV.

Match key is Gene + Sample + genomic coordinates when present (paralog-safe);
otherwise Gene + Sample.

Usage:

    python scripts/join_lof_predictions.py \\
        --lof-matrix N44_ST258/lof_matrix/lof_matrix.tsv \\
        --predictions N44_ST258/plmlof_input/predictions_mutcode.tsv \\
        --manifest N44_ST258/plmlof_input/manifest_all.tsv \\
        --output N44_ST258/lof_matrix/lof_matrix_plmlof.tsv \\
        --extras-output N44_ST258/lof_matrix/lof_matrix_plmlof_extras.tsv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)

# Matrix column name → predictor field name.
MAIN_SCORE_COLS: list[tuple[str, str]] = [
    ("LoF_score", "lof_score"),
    ("lof_p", "lof_p"),
    ("lof_q", "lof_q"),
    ("mlof_score", "mlof_score"),
    ("mlof_p", "mlof_p"),
    ("mlof_q", "mlof_q"),
]

MAIN_OUT_NAMES = [name for name, _ in MAIN_SCORE_COLS]

# Join keys shared with the matrix so the extras file can be re-merged.
JOIN_KEY_COLS = [
    "Replicon",
    "Gene",
    "Sample",
    "Genome_Contig",
    "Genome_Start",
    "Genome_End",
    "Genome_Strand",
]

# Predictor fields written only to --extras-output (order preserved).
EXTRAS_PRED_COLS = [
    "mlof_bin",
    "wreck",
    "wreck_kind",
    "wreck_prior",
    "lof_sd",
    "mlof_sd",
    "mlof_mean",
    "mlof_n_sites",
    "in_family",
    "nearest_train_gene",
    "ref_cosine",
    "category",
    "mge_group",
    "record_id",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lof-matrix", type=Path, required=True)
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="TSV from scripts/run_predict_batch.py (gene column = record_id)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="manifest_all.tsv from build_from_lof_matrix.py",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--extras-output",
        type=Path,
        default=None,
        help="Optional sidecar with wreck/bin/diagnostics + join keys",
    )
    return p.parse_args(argv)


def _norm(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    # Matrix often stores floats as "11021.0"; manifest may store "11021".
    if text.endswith(".0"):
        try:
            float(text)
            return text[:-2]
        except ValueError:
            pass
    return text


def _coord_key(
    gene: str,
    sample: str,
    contig: str,
    start: str,
    end: str,
    strand: str,
) -> tuple[str, ...]:
    return (
        gene,
        sample,
        _norm(contig),
        _norm(start),
        _norm(end),
        _norm(strand),
    )


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    """record_id → prediction row (``gene`` column is the FASTA record_id)."""
    by_id: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rid = (row.get("gene") or "").strip()
            if not rid:
                continue
            by_id[rid] = row
    return by_id


def load_manifest_index(
    path: Path,
) -> tuple[dict[tuple[str, ...], str], dict[tuple[str, str], str]]:
    """Build (coord_key → record_id) and fallback (gene, sample) → record_id."""
    by_coord: dict[tuple[str, ...], str] = {}
    by_gene_sample: dict[tuple[str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rid = (row.get("record_id") or "").strip()
            gene = (row.get("gene_id") or "").strip()
            sample = (row.get("sample") or "").strip()
            if not rid or not gene or not sample:
                continue
            key = _coord_key(
                gene,
                sample,
                row.get("genome_contig", ""),
                row.get("genome_start", ""),
                row.get("genome_end", ""),
                row.get("genome_strand", ""),
            )
            by_coord.setdefault(key, rid)
            by_gene_sample.setdefault((gene, sample), rid)
    return by_coord, by_gene_sample


def resolve_record_id(
    row: dict[str, str],
    by_coord: dict[tuple[str, ...], str],
    by_gene_sample: dict[tuple[str, str], str],
) -> str | None:
    gene = (row.get("Gene") or "").strip()
    sample = (row.get("Sample") or "").strip()
    if not gene or not sample:
        return None
    key = _coord_key(
        gene,
        sample,
        row.get("Genome_Contig", ""),
        row.get("Genome_Start", ""),
        row.get("Genome_End", ""),
        row.get("Genome_Strand", ""),
    )
    rid = by_coord.get(key)
    if rid is not None:
        return rid
    return by_gene_sample.get((gene, sample))


def _matrix_base_fields(fieldnames: list[str]) -> list[str]:
    """Original matrix columns, stripping any prior score / extras joins."""
    drop = set(MAIN_OUT_NAMES) | set(EXTRAS_PRED_COLS) | {"plmlof_record_id"}
    return [c for c in fieldnames if c not in drop]


def _insert_scores_after_mutation_codes(base_fields: list[str]) -> list[str]:
    """Place the six score columns immediately after Mutation_Codes."""
    if "Mutation_Codes" in base_fields:
        i = base_fields.index("Mutation_Codes") + 1
        return base_fields[:i] + MAIN_OUT_NAMES + base_fields[i:]
    return base_fields + MAIN_OUT_NAMES


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    logger.info("Loading predictions from %s", args.predictions)
    preds = load_predictions(args.predictions)
    logger.info("Loaded %d prediction rows", len(preds))

    logger.info("Loading manifest index from %s", args.manifest)
    by_coord, by_gene_sample = load_manifest_index(args.manifest)
    logger.info(
        "Indexed %d coord keys, %d gene+sample keys",
        len(by_coord),
        len(by_gene_sample),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.extras_output is not None:
        args.extras_output.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_matched = 0
    n_unmatched = 0

    extras_fields = JOIN_KEY_COLS + EXTRAS_PRED_COLS

    with open(args.lof_matrix, newline="", encoding="utf-8") as fin, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"empty or headerless matrix: {args.lof_matrix}")

        base_fields = _matrix_base_fields(list(reader.fieldnames))
        out_fields = _insert_scores_after_mutation_codes(base_fields)
        writer = csv.DictWriter(
            fout,
            fieldnames=out_fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()

        extras_fh = None
        extras_writer = None
        if args.extras_output is not None:
            extras_fh = open(args.extras_output, "w", newline="", encoding="utf-8")
            extras_writer = csv.DictWriter(
                extras_fh,
                fieldnames=extras_fields,
                delimiter="\t",
                extrasaction="ignore",
                lineterminator="\n",
            )
            extras_writer.writeheader()

        try:
            for row in reader:
                n_rows += 1
                rid = resolve_record_id(row, by_coord, by_gene_sample)
                pred = preds.get(rid) if rid else None

                out = {c: row.get(c, "") for c in base_fields}
                if pred is None:
                    n_unmatched += 1
                    for out_name, _src in MAIN_SCORE_COLS:
                        out[out_name] = ""
                else:
                    n_matched += 1
                    for out_name, src in MAIN_SCORE_COLS:
                        out[out_name] = pred.get(src, "")
                writer.writerow(out)

                if extras_writer is not None:
                    extra = {c: row.get(c, "") for c in JOIN_KEY_COLS}
                    if pred is None:
                        for c in EXTRAS_PRED_COLS:
                            extra[c] = ""
                        extra["record_id"] = rid or ""
                    else:
                        for c in EXTRAS_PRED_COLS:
                            if c == "record_id":
                                extra[c] = rid or pred.get("gene", "")
                            else:
                                extra[c] = pred.get(c, "")
                    extras_writer.writerow(extra)
        finally:
            if extras_fh is not None:
                extras_fh.close()

    logger.info(
        "Wrote %s  (%d rows: %d scored, %d without prediction)",
        args.output,
        n_rows,
        n_matched,
        n_unmatched,
    )
    if args.extras_output is not None:
        logger.info("Wrote extras sidecar %s", args.extras_output)
    if n_unmatched:
        logger.info(
            "Unscored rows are expected for skipped import (missing Panaroo gene "
            "or mge_unclustered with no reference)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
