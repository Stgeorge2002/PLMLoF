"""Build PLMLoF FASTA pairs from MutCode codes plus a pan-genome reference.

Decode policy matches MutCode mutcode-v1:

    start from the reference FASTA entry for the gene
    apply each [start, delete_length, inserted] on original coordinates
    unlisted bases stay as the FASTA (``[]`` ⇒ allele == reference)

Panaroo stores that FASTA as ``pan_genome_reference.fa``. Rustaroo stores it
as ``genes.fa``. When both trees sit in one results directory, the matrix
gene ids select which tree to decode against.

The outbreak LoF matrix's ``Reference_Length`` / ``Sample_Length`` columns are
annotation metadata and are **not** used as the MutCode reference length.
Those can disagree with the FASTA (e.g. CDS length without a terminal stop).
"""

from __future__ import annotations

import csv
import gzip
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from plmlof.mutcode import (
    MUTCODE_NATIVE_COLUMNS,
    REQUIRED_LOF_COLUMNS,
    MutCodeError,
    apply_edits,
    category_from_replicon,
    expected_target_length,
    fasta_header_from_description,
    iter_reference_fastas,
    mge_group_from_replicon,
    parse_codes,
    read_fasta,
    read_fasta_ids,
    reconstruct,
    reference_style_hint,
    resolve_reference_fasta,
    sha256_seq,
    write_fasta,
)

logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)


class LofImportError(RuntimeError):
    """Fatal LoF-matrix / Panaroo import failure."""


@dataclass
class GeneBucket:
    """Accumulated FASTA records for one (category, mge, gene) folder."""

    gene_id: str
    category: str
    mge_group: str
    description: str = ""
    ref_seq: str = ""
    ref_records: list[tuple[str, str]] = field(default_factory=list)
    var_records: list[tuple[str, str]] = field(default_factory=list)
    manifest: list[dict] = field(default_factory=list)
    sample_counts: dict[str, int] = field(default_factory=dict)


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, newline="", encoding="utf-8")


def match_reference(
    seqs: dict[str, str], gene: str, description: str
) -> str | None:
    """Sequence for a matrix gene, including Rustaroo header aliases.

    Order:

    1. Exact FASTA id (Panaroo ``group_N`` and Rustaroo ids that already match).
    2. The product string rendered as a Rustaroo header, when that header is
       an extension of the matrix gene (``ATP`` + ``ATP-NAD kinase`` →
       ``ATP-NAD_kinase``). This also picks the full product id when a
       truncated gene id prefixes both that header and a ``_2`` paralog.
    3. The single FASTA id that extends the gene with ``_`` or ``-``
       (``fabY`` → ``fabY_2``). Several extensions and no product hit is
       ambiguous, so the row is left unmatched.
    """
    if gene in seqs:
        return seqs[gene]
    if not gene:
        return None
    slug = fasta_header_from_description(description)
    hits = [
        header
        for header in seqs
        if len(header) > len(gene)
        and header.startswith(gene)
        and header[len(gene)] in "_-"
    ]
    if slug in hits:
        return seqs[slug]
    if len(hits) == 1:
        return seqs[hits[0]]
    return None


class ReferenceIndex:
    """One replicon's reference FASTA and its gene-id lookup cache."""

    def __init__(self, seqs: dict[str, str], *, style: str, path: Path):
        self.seqs = seqs
        self.style = style
        self.path = path
        self._cache: dict[tuple[str, str], str | None] = {}

    def get(self, gene: str, description: str) -> str | None:
        key = (gene, description)
        if key not in self._cache:
            self._cache[key] = match_reference(self.seqs, gene, description)
        return self._cache[key]


def reference_for_row(
    cache: dict[str, ReferenceIndex],
    replicon: str,
    gene: str,
    description: str,
) -> str | None:
    key = "chromosome" if replicon == "chromosome" else mge_group_from_replicon(replicon)
    index = cache.get(key)
    if index is None:
        return None
    return index.get(gene, description)


def record_id(gene_id: str, sample: str, paralog_index: int) -> str:
    if paralog_index == 0:
        return f"{gene_id}__{sample}"
    return f"{gene_id}__{sample}__p{paralog_index}"


def _ensure_ref_cache(
    ref_cache: dict[str, ReferenceIndex],
    results_dir: Path,
    replicon: str,
    *,
    style: str,
    missing_replicons: set[str],
) -> bool:
    """Load this replicon's reference FASTA. Return False when it is absent.

    ``mge_unclustered__...`` singleton plasmids have no reference tree.
    """
    key = "chromosome" if replicon == "chromosome" else mge_group_from_replicon(replicon)
    if key in ref_cache:
        return True
    if key in missing_replicons:
        return False
    try:
        fasta = resolve_reference_fasta(results_dir, replicon, style=style)
    except FileNotFoundError:
        missing_replicons.add(key)
        logger.info(
            "No %s reference FASTA for %s — skipping those matrix rows",
            style,
            key,
        )
        return False
    seqs = read_fasta(fasta)
    ref_cache[key] = ReferenceIndex(seqs, style=style, path=fasta)
    logger.info("Loaded %d genes from %s (%s)", len(seqs), fasta, key)
    return True


def _matrix_gene_ids(lof_matrix: Path) -> set[str]:
    """Unique ``Gene`` values, used to choose between reference trees."""
    with open_text(lof_matrix) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = list(reader.fieldnames or [])
        missing = [column for column in REQUIRED_LOF_COLUMNS if column not in fields]
        if missing:
            raise LofImportError(
                f"LoF matrix missing columns {missing}; have {fields}"
            )
        return {
            (row.get("Gene") or "").strip()
            for row in reader
            if (row.get("Gene") or "").strip()
        }


def choose_reference_style(results_dir: Path, lof_matrix: Path) -> str:
    """Pick ``panaroo`` or ``rustaroo`` for this matrix.

    A results directory that contains only one tree uses that tree. When both
    are present, the tree with more exact ``Gene`` id hits wins. A tie uses
    the matrix filename (``rustaroo_lof_matrix.tsv`` / ``panaroo_lof_matrix.tsv``).
    """
    present = {
        style: iter_reference_fastas(results_dir, style)
        for style in ("panaroo", "rustaroo")
    }
    present = {style: paths for style, paths in present.items() if paths}
    if not present:
        raise LofImportError(
            f"no Panaroo or Rustaroo reference FASTA under {results_dir}"
        )
    if len(present) == 1:
        style = next(iter(present))
        logger.info("Reference style %s (only reference tree present)", style)
        return style

    gene_ids = _matrix_gene_ids(lof_matrix)
    scores: dict[str, int] = {}
    for style, paths in present.items():
        headers: set[str] = set()
        for path in paths:
            headers.update(read_fasta_ids(path))
        scores[style] = sum(gene in headers for gene in gene_ids)
    ranked = sorted(present, key=lambda style: scores[style], reverse=True)
    best, second = ranked[0], ranked[1]
    summary = ", ".join(f"{style}={scores[style]}" for style in ranked)
    if scores[best] > scores[second]:
        logger.info("Reference style %s (exact gene-id hits %s)", best, summary)
        return best
    hint = reference_style_hint(lof_matrix.name)
    if hint in present:
        logger.info(
            "Reference style %s (exact gene-id hits tied %s; filename)",
            hint,
            summary,
        )
        return hint
    raise LofImportError(
        "both Panaroo and Rustaroo references are present and match the "
        f"matrix gene ids equally ({summary}). Pass --reference-style "
        "panaroo or --reference-style rustaroo."
    )


def _write_buckets(
    buckets: dict[tuple[str, str, str], GeneBucket],
    out_dir: Path,
    min_samples: int,
) -> tuple[list[dict], list[dict], int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_manifest: list[dict] = []
    index_rows: list[dict] = []
    n_dirs = 0

    for category, mge_group, gene_id in sorted(buckets):
        bucket = buckets[(category, mge_group, gene_id)]
        n_samples = len(bucket.sample_counts)
        if n_samples < min_samples:
            continue

        if category == "chromosome":
            gene_dir = out_dir / "chromosome" / gene_id
        else:
            gene_dir = out_dir / "plasmid" / mge_group / gene_id

        gene_dir.mkdir(parents=True, exist_ok=True)
        write_fasta(gene_dir / "ref.fasta", bucket.ref_records)
        write_fasta(gene_dir / "var.fasta", bucket.var_records)
        with open(gene_dir / "manifest.tsv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(bucket.manifest[0].keys()), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(bucket.manifest)

        all_manifest.extend(bucket.manifest)
        index_rows.append({
            "gene_id": gene_id,
            "category": category,
            "mge_group": mge_group,
            "description": bucket.description,
            "n_records": len(bucket.var_records),
            "n_samples": n_samples,
            "ref_length_nt": len(bucket.ref_seq),
            "dir": str(gene_dir),
        })
        n_dirs += 1

    by_cat: dict[Path, list[dict]] = defaultdict(list)
    for row in index_rows:
        if row["category"] == "chromosome":
            by_cat[out_dir / "chromosome"].append(row)
        else:
            by_cat[out_dir / "plasmid" / row["mge_group"]].append(row)
    for cat_path, rows in by_cat.items():
        with open(cat_path / "index.tsv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    with open(out_dir / "manifest_all.tsv", "w", newline="") as fh:
        if all_manifest:
            writer = csv.DictWriter(
                fh, fieldnames=list(all_manifest[0].keys()), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(all_manifest)

    return all_manifest, index_rows, n_dirs


def detect_table_format(fieldnames: list[str] | None, first_line: str | None = None) -> str:
    """Return ``lof_matrix`` or ``mutcode_native``."""
    if first_line and first_line.startswith("#mutcode"):
        return "mutcode_native"
    if fieldnames and all(c in fieldnames for c in MUTCODE_NATIVE_COLUMNS):
        return "mutcode_native"
    if fieldnames and all(c in fieldnames for c in REQUIRED_LOF_COLUMNS):
        return "lof_matrix"
    raise LofImportError(
        f"unrecognised code table columns: {fieldnames!r}; "
        f"need LoF-matrix {REQUIRED_LOF_COLUMNS} or MutCode {MUTCODE_NATIVE_COLUMNS}"
    )


def build_from_lof_matrix(
    *,
    lof_matrix: Path,
    results_dir: Path,
    out_dir: Path,
    strict_sample_length: bool = False,
    skip_missing_genes: bool = True,
    min_samples: int = 1,
    reference_style: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Reconstruct alleles from an outbreak LoF matrix (Mutation_Codes column).

    The MutCode reference is Panaroo ``pan_genome_reference.fa`` or Rustaroo
    ``genes.fa``. ``reference_style`` ``None`` selects the tree automatically.
    Matrix ``Reference_Length`` is stop-trimmed annotation metadata and is
    ignored for decode (counted in the final summary only). Missing gene IDs
    are skipped by default.
    """
    if reference_style is None:
        style = choose_reference_style(results_dir, lof_matrix)
    elif reference_style in ("panaroo", "rustaroo"):
        if not iter_reference_fastas(results_dir, reference_style):
            raise LofImportError(
                f"no {reference_style} reference FASTA under {results_dir}"
            )
        style = reference_style
        logger.info("Reference style %s (--reference-style)", style)
    else:
        raise LofImportError(
            f"reference_style must be panaroo, rustaroo, or omitted; got {reference_style!r}"
        )

    buckets: dict[tuple[str, str, str], GeneBucket] = {}
    ref_cache: dict[str, ReferenceIndex] = {}
    seen_replicons: set[str] = set()
    warned_ref_len: set[str] = set()

    n_rows = 0
    n_mutated = 0
    n_identical = 0
    n_skipped_missing = 0
    n_skipped_no_replicon_ref = 0
    n_sample_len_mismatch = 0
    n_matrix_ref_len_mismatch = 0
    n_matrix_ref_len_mismatch_mutated = 0
    missing_genes: set[str] = set()
    missing_replicons: set[str] = set()
    errors: list[str] = []

    with open_text(lof_matrix) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise LofImportError(f"empty LoF matrix: {lof_matrix}")
        fmt = detect_table_format(list(reader.fieldnames))
        if fmt != "lof_matrix":
            raise LofImportError(
                "this path expects an outbreak LoF matrix; "
                "for native mutation_codes.tsv.gz use build_from_mutcode_table()"
            )
        missing_cols = [c for c in REQUIRED_LOF_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise LofImportError(
                f"LoF matrix missing columns {missing_cols}; "
                f"have {list(reader.fieldnames)}"
            )

        for row in reader:
            n_rows += 1
            replicon = row["Replicon"].strip()
            gene_id = row["Gene"].strip()
            sample = row["Sample"].strip()
            description = (row.get("Description") or "").strip()
            codes_raw = row["Mutation_Codes"]

            matrix_ref_len: int | None = None
            sample_len: int | None = None
            if row.get("Reference_Length") not in (None, ""):
                try:
                    matrix_ref_len = int(float(row["Reference_Length"]))
                except (TypeError, ValueError) as exc:
                    raise LofImportError(
                        f"row {n_rows}: bad Reference_Length for {gene_id}/{sample}: {exc}"
                    ) from exc
            if row.get("Sample_Length") not in (None, ""):
                try:
                    sample_len = int(float(row["Sample_Length"]))
                except (TypeError, ValueError) as exc:
                    raise LofImportError(
                        f"row {n_rows}: bad Sample_Length for {gene_id}/{sample}: {exc}"
                    ) from exc

            if replicon not in seen_replicons:
                seen_replicons.add(replicon)
            if not _ensure_ref_cache(
                ref_cache,
                results_dir,
                replicon,
                style=style,
                missing_replicons=missing_replicons,
            ):
                n_skipped_no_replicon_ref += 1
                continue

            ref_seq = reference_for_row(ref_cache, replicon, gene_id, description)
            if ref_seq is None:
                missing_genes.add(f"{replicon}:{gene_id}")
                n_skipped_missing += 1
                if not skip_missing_genes:
                    raise LofImportError(
                        f"gene {gene_id!r} absent from the {style} reference for {replicon}. "
                        f"Re-run with --skip-missing-genes to continue past absent IDs."
                    )
                continue

            try:
                edits = parse_codes(codes_raw)
            except MutCodeError as exc:
                errors.append(f"{gene_id}/{sample}: {exc}")
                if len(errors) >= 50:
                    break
                continue

            # Matrix Reference_Length is stop-trimmed annotation metadata in N44;
            # MutCode coordinates are on the Panaroo FASTA. Count, do not spam.
            if matrix_ref_len is not None and len(ref_seq) != matrix_ref_len:
                n_matrix_ref_len_mismatch += 1
                if gene_id not in warned_ref_len:
                    warned_ref_len.add(gene_id)
                if edits:
                    n_matrix_ref_len_mismatch_mutated += 1

            try:
                # [] → identity copy of the FASTA reference
                var_seq = apply_edits(ref_seq, edits)
                mutcode_len = expected_target_length(len(ref_seq), edits)
                if len(var_seq) != mutcode_len:
                    raise MutCodeError(
                        f"internal length mismatch {len(var_seq)} != {mutcode_len}"
                    )
                if sample_len is not None and mutcode_len != sample_len:
                    n_sample_len_mismatch += 1
                    if strict_sample_length:
                        raise MutCodeError(
                            f"{gene_id}/{sample}: MutCode DNA length {mutcode_len} != "
                            f"Sample_Length {sample_len}"
                        )
            except MutCodeError as exc:
                errors.append(f"{gene_id}/{sample}: {exc}")
                if len(errors) >= 50:
                    break
                continue

            category, mge_group = category_from_replicon(replicon)
            bkey = (category, mge_group, gene_id)
            bucket = buckets.get(bkey)
            if bucket is None:
                bucket = GeneBucket(
                    gene_id=gene_id,
                    category=category,
                    mge_group=mge_group,
                    description=description,
                    ref_seq=ref_seq,
                )
                buckets[bkey] = bucket
            elif bucket.ref_seq != ref_seq:
                errors.append(f"{gene_id}: conflicting reference sequences across rows")
                continue

            p_idx = bucket.sample_counts.get(sample, 0)
            bucket.sample_counts[sample] = p_idx + 1
            rid = record_id(gene_id, sample, p_idx)
            bucket.ref_records.append((rid, ref_seq))
            bucket.var_records.append((rid, var_seq))
            bucket.manifest.append({
                "record_id": rid,
                "gene_id": gene_id,
                "category": category,
                "mge_group": mge_group,
                "sample": sample,
                "locus_tag": "",
                "gene_name": gene_id,
                "description": description or bucket.description,
                "ref_length_nt": len(ref_seq),
                "var_length_nt": len(var_seq),
                "matrix_reference_length": matrix_ref_len if matrix_ref_len is not None else "",
                "sample_length_matrix": sample_len if sample_len is not None else "",
                "n_edits": len(edits),
                "ref_source": (
                    "pan_genome_reference+mutcode"
                    if style == "panaroo"
                    else "genes.fa+mutcode"
                ),
                "genome_contig": (row.get("Genome_Contig") or "").strip(),
                "genome_start": (row.get("Genome_Start") or "").strip(),
                "genome_end": (row.get("Genome_End") or "").strip(),
                "genome_strand": (row.get("Genome_Strand") or "").strip(),
            })
            if edits:
                n_mutated += 1
            else:
                n_identical += 1

    if errors:
        for err in errors[:20]:
            logger.error("%s", err)
        raise LofImportError(
            f"failed to reconstruct {len(errors)} row(s) "
            f"(showing up to 20); fix the matrix/reference or re-run"
        )

    all_manifest, _index_rows, n_dirs = _write_buckets(buckets, out_dir, min_samples)

    stats = [{
        "reference_style": style,
        "n_matrix_rows": n_rows,
        "n_gene_dirs": n_dirs,
        "n_records": len(all_manifest),
        "n_identical": n_identical,
        "n_mutated": n_mutated,
        "n_skipped_missing_gene_rows": n_skipped_missing,
        "n_missing_gene_ids": len(missing_genes),
        "n_skipped_no_replicon_ref_rows": n_skipped_no_replicon_ref,
        "n_missing_replicon_refs": len(missing_replicons),
        "n_sample_length_mismatch": n_sample_len_mismatch,
        "n_matrix_ref_length_mismatch_rows": n_matrix_ref_len_mismatch,
        "n_matrix_ref_length_mismatch_genes": len(warned_ref_len),
        "n_matrix_ref_length_mismatch_mutated_rows": n_matrix_ref_len_mismatch_mutated,
        "n_replicons": len(seen_replicons),
    }]
    with open(out_dir / "build_stats.tsv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(stats)

    if missing_replicons:
        path = out_dir / "missing_replicon_refs.txt"
        path.write_text("\n".join(sorted(missing_replicons)) + "\n")
        logger.info(
            "Skipped %d rows for %d replicons with no %s reference → %s",
            n_skipped_no_replicon_ref,
            len(missing_replicons),
            style,
            path,
        )
    if missing_genes:
        miss_path = out_dir / "missing_genes.txt"
        miss_path.write_text("\n".join(sorted(missing_genes)) + "\n")
        logger.warning(
            "Skipped %d rows for %d gene IDs absent from the %s reference → %s",
            n_skipped_missing,
            len(missing_genes),
            style,
            miss_path,
        )

    logger.info(
        "Wrote %d gene folders, %d records (%d mutated, %d identical) → %s",
        n_dirs,
        len(all_manifest),
        n_mutated,
        n_identical,
        out_dir,
    )
    if n_matrix_ref_len_mismatch:
        # Almost always FASTA = matrix + 3 (terminal stop in Panaroo, not in matrix).
        logger.info(
            "Matrix Reference_Length ≠ FASTA for %d genes / %d rows "
            "(%d of those rows had non-empty codes); decode used the reference FASTA",
            len(warned_ref_len),
            n_matrix_ref_len_mismatch,
            n_matrix_ref_len_mismatch_mutated,
        )
    if n_sample_len_mismatch:
        logger.info(
            "%d rows had Sample_Length ≠ MutCode DNA length; sequences follow MutCode",
            n_sample_len_mismatch,
        )
    return all_manifest, stats


def build_from_mutcode_table(
    *,
    codes: Path,
    reference_fa: Path,
    out_dir: Path,
    replicon: str = "chromosome",
    min_samples: int = 1,
    verify_hashes: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Decode a native MutCode ``mutation_codes.tsv[.gz]`` against one FASTA.

    Verifies ``reference_sha256`` / ``target_sha256`` when ``verify_hashes`` is set.
    """
    if not reference_fa.is_file():
        raise LofImportError(f"reference FASTA not found: {reference_fa}")

    refs = read_fasta(reference_fa)
    logger.info("Loaded %d genes from %s", len(refs), reference_fa)

    category, mge_group = category_from_replicon(replicon)
    buckets: dict[tuple[str, str, str], GeneBucket] = {}
    n_rows = 0
    n_mutated = 0
    n_identical = 0
    errors: list[str] = []

    with open_text(codes) as fh:
        first = fh.readline()
        if not first.startswith("#mutcode"):
            # allow tables saved without metadata (still native columns)
            fh.seek(0)
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise LofImportError(f"empty MutCode table: {codes}")
        missing = [c for c in MUTCODE_NATIVE_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise LofImportError(
                f"MutCode table missing columns {missing}; have {list(reader.fieldnames)}"
            )

        for row in reader:
            n_rows += 1
            sample = row["sample"].strip()
            gene_id = row["gene"].strip()
            codes_raw = row["codes"]
            ref_seq = refs.get(gene_id)
            if ref_seq is None:
                errors.append(f"{gene_id}/{sample}: gene absent from reference FASTA")
                if len(errors) >= 50:
                    break
                continue
            try:
                declared_ref_len = int(row["reference_length"])
                if len(ref_seq) != declared_ref_len:
                    raise MutCodeError(
                        f"FASTA length {len(ref_seq)} != MutCode reference_length "
                        f"{declared_ref_len}"
                    )
                kwargs = {}
                if verify_hashes:
                    kwargs["reference_sha256"] = row["reference_sha256"].strip()
                    kwargs["target_sha256"] = row["target_sha256"].strip()
                    kwargs["target_length"] = int(row["target_length"])
                var_seq = reconstruct(ref_seq, codes_raw, **kwargs)
                edits = parse_codes(codes_raw)
            except (MutCodeError, ValueError) as exc:
                errors.append(f"{gene_id}/{sample}: {exc}")
                if len(errors) >= 50:
                    break
                continue

            bkey = (category, mge_group, gene_id)
            bucket = buckets.get(bkey)
            if bucket is None:
                bucket = GeneBucket(
                    gene_id=gene_id,
                    category=category,
                    mge_group=mge_group,
                    ref_seq=ref_seq,
                )
                buckets[bkey] = bucket

            p_idx = bucket.sample_counts.get(sample, 0)
            bucket.sample_counts[sample] = p_idx + 1
            rid = record_id(gene_id, sample, p_idx)
            bucket.ref_records.append((rid, ref_seq))
            bucket.var_records.append((rid, var_seq))
            bucket.manifest.append({
                "record_id": rid,
                "gene_id": gene_id,
                "category": category,
                "mge_group": mge_group,
                "sample": sample,
                "locus_tag": "",
                "gene_name": gene_id,
                "description": "",
                "ref_length_nt": len(ref_seq),
                "var_length_nt": len(var_seq),
                "n_edits": len(edits),
                "ref_source": "mutcode_native+pan_genome_reference",
                "reference_sha256": sha256_seq(ref_seq),
                "target_sha256": sha256_seq(var_seq),
            })
            if edits:
                n_mutated += 1
            else:
                n_identical += 1

    if errors:
        for err in errors[:20]:
            logger.error("%s", err)
        raise LofImportError(f"failed to reconstruct {len(errors)} row(s)")

    all_manifest, _, n_dirs = _write_buckets(buckets, out_dir, min_samples)
    stats = [{
        "n_matrix_rows": n_rows,
        "n_gene_dirs": n_dirs,
        "n_records": len(all_manifest),
        "n_identical": n_identical,
        "n_mutated": n_mutated,
        "format": "mutcode_native",
    }]
    with open(out_dir / "build_stats.tsv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(stats)
    logger.info(
        "Wrote %d gene folders, %d records (%d mutated, %d identical) → %s",
        n_dirs,
        len(all_manifest),
        n_mutated,
        n_identical,
        out_dir,
    )
    return all_manifest, stats
