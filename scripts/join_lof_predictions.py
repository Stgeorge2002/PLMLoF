#!/usr/bin/env python3
"""Annotate a MutCode LoF matrix with PLMLoF scores.

One command: matrix in → same rows/columns out, plus LoF_score / lof_p / lof_q
immediately after Mutation_Codes.

    python scripts/join_lof_predictions.py N44_ST258/lof_matrix/lof_matrix.tsv

Finds ``../plmlof_input/predictions_8m.tsv`` (or ``predictions.tsv``) next to the
matrix tree, writes ``lof_matrix_plmlof.tsv`` beside the input, and uses
``manifest_all.tsv`` when present (paralog-safe). Override any of that with flags.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)

LOF_SCORE_COLS: list[tuple[str, str]] = [
    ("LoF_score", "lof_score"),
    ("lof_p", "lof_p"),
    ("lof_q", "lof_q"),
]
MAIN_OUT_NAMES = [name for name, _ in LOF_SCORE_COLS]

JOIN_KEY_COLS = [
    "Replicon", "Gene", "Sample",
    "Genome_Contig", "Genome_Start", "Genome_End", "Genome_Strand",
]
EXTRAS_PRED_COLS = [
    "wreck", "wreck_kind", "wreck_prior", "lof_sd",
    "in_family", "nearest_train_gene", "ref_cosine", "route",
    "category", "mge_group", "record_id",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "lof_matrix",
        type=Path,
        nargs="?",
        default=None,
        help="Original LoF matrix TSV",
    )
    p.add_argument(
        "--lof-matrix",
        dest="lof_matrix_opt",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Predict TSV (default: auto-discover under ../plmlof_input/)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest_all.tsv (default: beside predictions if present)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <matrix_dir>/<matrix_stem>_plmlof.tsv",
    )
    p.add_argument(
        "--extras-output",
        type=Path,
        default=None,
        help="Optional wreck/diagnostics sidecar",
    )
    return p.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    matrix = args.lof_matrix_opt or args.lof_matrix
    if matrix is None:
        raise SystemExit("Pass the LoF matrix path, e.g. N44_ST258/lof_matrix/lof_matrix.tsv")
    matrix = matrix.resolve()
    if not matrix.is_file():
        raise SystemExit(f"LoF matrix not found: {matrix}")

    if args.predictions is not None:
        preds = args.predictions.resolve()
    else:
        preds = _discover_predictions(matrix)
    if not preds.is_file():
        raise SystemExit(f"Predictions not found: {preds}")

    if args.manifest is not None:
        manifest = args.manifest.resolve()
    else:
        cand = preds.parent / "manifest_all.tsv"
        manifest = cand if cand.is_file() else None

    if args.output is not None:
        out = args.output
    else:
        out = matrix.with_name(f"{matrix.stem}_plmlof{matrix.suffix}")
    return matrix, preds, manifest, out


def _discover_predictions(matrix: Path) -> Path:
    """Prefer predictions_8m.tsv, then predictions.tsv, near the outbreak tree."""
    # …/N44_ST258/lof_matrix/lof_matrix.tsv → …/N44_ST258/plmlof_input/
    outbreak = matrix.parent.parent
    candidates = [
        outbreak / "plmlof_input" / "predictions_8m.tsv",
        outbreak / "plmlof_input" / "predictions.tsv",
        matrix.parent / "predictions_8m.tsv",
        matrix.parent / "predictions.tsv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise SystemExit(
        "Could not auto-find predictions. Pass --predictions PATH.\n"
        "Looked for:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def _norm(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            float(text)
            return text[:-2]
        except ValueError:
            pass
    return text


def _coord_key(
    gene: str, sample: str, contig: str, start: str, end: str, strand: str,
) -> tuple[str, ...]:
    return (gene, sample, _norm(contig), _norm(start), _norm(end), _norm(strand))


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rid = (row.get("gene") or "").strip()
            if rid:
                by_id[rid] = row
    return by_id


def load_manifest_index(
    path: Path,
) -> tuple[dict[tuple[str, ...], str], dict[tuple[str, str], str]]:
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
                gene, sample,
                row.get("genome_contig", ""),
                row.get("genome_start", ""),
                row.get("genome_end", ""),
                row.get("genome_strand", ""),
            )
            by_coord.setdefault(key, rid)
            by_gene_sample.setdefault((gene, sample), rid)
    return by_coord, by_gene_sample


def index_from_predictions(
    preds: dict[str, dict[str, str]],
) -> dict[tuple[str, str], str]:
    by_gene_sample: dict[tuple[str, str], str] = {}
    n_dup = 0
    for rid, row in preds.items():
        gene = (row.get("gene_id") or "").strip()
        sample = (row.get("sample") or "").strip()
        if not gene or not sample:
            continue
        key = (gene, sample)
        if key in by_gene_sample and by_gene_sample[key] != rid:
            n_dup += 1
            continue
        by_gene_sample.setdefault(key, rid)
    if n_dup:
        logger.warning(
            "%d Gene+Sample collisions in predictions (paralogs); "
            "prefer a manifest_all.tsv beside the predictions file",
            n_dup,
        )
    return by_gene_sample


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
        gene, sample,
        row.get("Genome_Contig", ""),
        row.get("Genome_Start", ""),
        row.get("Genome_End", ""),
        row.get("Genome_Strand", ""),
    )
    return by_coord.get(key) or by_gene_sample.get((gene, sample))


def _matrix_base_fields(fieldnames: list[str]) -> list[str]:
    drop = set(MAIN_OUT_NAMES) | set(EXTRAS_PRED_COLS) | {"plmlof_record_id"}
    return [c for c in fieldnames if c not in drop]


def _insert_scores_after_mutation_codes(base_fields: list[str]) -> list[str]:
    if "Mutation_Codes" in base_fields:
        i = base_fields.index("Mutation_Codes") + 1
        return base_fields[:i] + MAIN_OUT_NAMES + base_fields[i:]
    return base_fields + MAIN_OUT_NAMES


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    matrix, preds_path, manifest, out = _resolve_paths(args)

    logger.info("Matrix:      %s", matrix)
    logger.info("Predictions: %s", preds_path)
    logger.info("Output:      %s", out)

    preds = load_predictions(preds_path)
    logger.info("Loaded %d prediction rows", len(preds))

    if manifest is not None:
        logger.info("Manifest:    %s", manifest)
        by_coord, by_gene_sample = load_manifest_index(manifest)
    else:
        by_coord = {}
        by_gene_sample = index_from_predictions(preds)
    logger.info(
        "Indexed %d coord keys, %d gene+sample keys",
        len(by_coord), len(by_gene_sample),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    extras_path = args.extras_output
    if extras_path is not None:
        extras_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = n_matched = n_unmatched = 0
    extras_fields = JOIN_KEY_COLS + EXTRAS_PRED_COLS

    with open(matrix, newline="", encoding="utf-8") as fin, open(
        out, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"empty or headerless matrix: {matrix}")

        base_fields = _matrix_base_fields(list(reader.fieldnames))
        writer = csv.DictWriter(
            fout,
            fieldnames=_insert_scores_after_mutation_codes(base_fields),
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()

        extras_fh = extras_writer = None
        if extras_path is not None:
            extras_fh = open(extras_path, "w", newline="", encoding="utf-8")
            extras_writer = csv.DictWriter(
                extras_fh, fieldnames=extras_fields, delimiter="\t",
                extrasaction="ignore", lineterminator="\n",
            )
            extras_writer.writeheader()

        try:
            for row in reader:
                n_rows += 1
                rid = resolve_record_id(row, by_coord, by_gene_sample)
                pred = preds.get(rid) if rid else None
                out_row = {c: row.get(c, "") for c in base_fields}
                if pred is None:
                    n_unmatched += 1
                    for name, _ in LOF_SCORE_COLS:
                        out_row[name] = ""
                else:
                    n_matched += 1
                    for name, src in LOF_SCORE_COLS:
                        out_row[name] = pred.get(src, "")
                writer.writerow(out_row)

                if extras_writer is not None:
                    extra = {c: row.get(c, "") for c in JOIN_KEY_COLS}
                    if pred is None:
                        for c in EXTRAS_PRED_COLS:
                            extra[c] = ""
                        extra["record_id"] = rid or ""
                    else:
                        for c in EXTRAS_PRED_COLS:
                            extra[c] = (
                                rid or pred.get("gene", "")
                                if c == "record_id"
                                else pred.get(c, "")
                            )
                    extras_writer.writerow(extra)
        finally:
            if extras_fh is not None:
                extras_fh.close()

    logger.info(
        "Wrote %s  (%d rows: %d scored, %d without prediction)",
        out, n_rows, n_matched, n_unmatched,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
