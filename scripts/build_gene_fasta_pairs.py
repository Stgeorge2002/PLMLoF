"""Build per-gene reference/variant CDS FASTA pairs for PLMLoF inference.

For every gene cluster that Panaroo called from the Bakta GFFs (chromosome
and, separately, each plasmid backbone group), this writes one folder
containing:

    ref.fasta       one record per sample carrying the gene, each holding
                    the Panaroo pan-genome-reference CDS (the same sequence
                    repeated under a per-sample record ID)
    var.fasta       the sample's own CDS allele under that same record ID
    manifest.tsv    gene/sample/locus_tag bookkeeping for the pair above

Record IDs are "<gene_id>__<sample>" (or "...__p<k>" for paralogs), so
ref.fasta and var.fasta share exactly the IDs scripts/predict.py matches on
(see plmlof/inference/vcf_handler.py:parse_fasta_pairs), and the "gene"
column of predict.py's output already carries sample identity, following
the same convention as examples/blaOXA1.

Sequences are DNA/CDS (not protein): ESM2 still only ever embeds translated
protein internally, but keeping DNA lets plmlof.wreck detect true
frameshifts/indels and lets the nucleotide features reach the GoF heads.

Usage:

    python scripts/build_gene_fasta_pairs.py \\
        --results-dir N44_ST258_results \\
        --out-dir N44_ST258_results/plmlof_input

This only reorganizes existing Panaroo/GFF output into model-ready FASTA
pairs; it does not run any PLMLoF model.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)


@dataclass
class GeneCluster:
    gene_id: str
    non_unique_name: str
    description: str
    # sample -> list of locus tags (usually one; >1 only for paralogs)
    sample_loci: dict[str, list[str]] = field(default_factory=dict)


def read_gene_presence_absence(path: Path) -> list[GeneCluster]:
    clusters = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sample_cols = header[3:]
        for row in reader:
            gene_id, non_unique, description = row[0], row[1], row[2]
            sample_loci: dict[str, list[str]] = {}
            for sample, cell in zip(sample_cols, row[3:]):
                if not cell:
                    continue
                loci = [x.strip() for x in cell.replace("\t", ";").split(";") if x.strip()]
                if loci:
                    sample_loci[sample] = loci
            clusters.append(GeneCluster(gene_id, non_unique, description, sample_loci))
    return clusters


def read_gene_data(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """(sample, locus_tag) -> {dna, prot, gene_name, description}."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["gff_file"], row["annotation_id"])
            out[key] = {
                "dna": row["dna_sequence"],
                "prot": row["prot_sequence"],
                "gene_name": row.get("gene_name", ""),
                "description": row.get("description", ""),
            }
    return out


def read_pan_genome_reference(path: Path) -> dict[str, str]:
    """FASTA header (first token) -> DNA sequence."""
    seqs: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            seqs[name] = "".join(chunks)
    return seqs


def read_gff_cds_locus_tags(path: Path) -> set[str]:
    """Locus tags of CDS features, stopping before the embedded ##FASTA block."""
    tags: set[str] = set()
    if not path.exists():
        return tags
    with open(path) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            for attr in fields[8].split(";"):
                if attr.startswith("locus_tag="):
                    tags.add(attr.split("=", 1)[1])
    return tags


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w") as fh:
        for rec_id, seq in records:
            fh.write(f">{rec_id}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def process_category(
    *,
    category: str,
    mge_group: str,
    panaroo_dir: Path,
    gff_dir: Path,
    out_dir: Path,
    min_samples: int,
) -> tuple[list[dict], dict]:
    """Build gene folders for one Panaroo run (chromosome, or one plasmid mge_N).

    Returns (manifest_rows, stats).
    """
    clusters = read_gene_presence_absence(panaroo_dir / "gene_presence_absence.csv")
    gene_data = read_gene_data(panaroo_dir / "gene_data.csv")
    pan_ref = read_pan_genome_reference(panaroo_dir / "pan_genome_reference.fa")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    index_rows: list[dict] = []
    n_fallback_ref = 0
    n_skipped_no_seq = 0

    for cluster in clusters:
        n_present = sum(len(v) for v in cluster.sample_loci.values())
        if n_present < min_samples:
            continue

        ref_seq = pan_ref.get(cluster.gene_id)
        ref_source = "pan_genome_reference"
        if ref_seq is None:
            # Fall back to the longest allele we actually observed for this
            # cluster (Panaroo occasionally has no representative sequence
            # for a handful of clusters, e.g. pure refound calls).
            candidates = []
            for sample, loci in cluster.sample_loci.items():
                for locus in loci:
                    info = gene_data.get((sample, locus))
                    if info and info["dna"]:
                        candidates.append(info["dna"])
            if not candidates:
                n_skipped_no_seq += 1
                continue
            ref_seq = max(candidates, key=len)
            ref_source = "fallback_longest_allele"
            n_fallback_ref += 1

        gene_dir = out_dir / cluster.gene_id
        ref_records: list[tuple[str, str]] = []
        var_records: list[tuple[str, str]] = []
        gene_manifest: list[dict] = []

        for sample in sorted(cluster.sample_loci):
            for idx, locus in enumerate(cluster.sample_loci[sample]):
                info = gene_data.get((sample, locus))
                if info is None or not info["dna"]:
                    continue
                rec_id = f"{cluster.gene_id}__{sample}" + (f"__p{idx}" if idx else "")
                ref_records.append((rec_id, ref_seq))
                var_records.append((rec_id, info["dna"]))
                gene_manifest.append({
                    "record_id": rec_id,
                    "gene_id": cluster.gene_id,
                    "category": category,
                    "mge_group": mge_group,
                    "sample": sample,
                    "locus_tag": locus,
                    "gene_name": info["gene_name"] or cluster.non_unique_name,
                    "description": info["description"] or cluster.description,
                    "ref_length_nt": len(ref_seq),
                    "var_length_nt": len(info["dna"]),
                    "ref_source": ref_source,
                })

        if not var_records:
            n_skipped_no_seq += 1
            continue

        gene_dir.mkdir(parents=True, exist_ok=True)
        write_fasta(gene_dir / "ref.fasta", ref_records)
        write_fasta(gene_dir / "var.fasta", var_records)
        with open(gene_dir / "manifest.tsv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(gene_manifest[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(gene_manifest)

        manifest_rows.extend(gene_manifest)
        index_rows.append({
            "gene_id": cluster.gene_id,
            "non_unique_name": cluster.non_unique_name,
            "description": cluster.description,
            "n_records": len(var_records),
            "ref_source": ref_source,
            "dir": str(gene_dir),
        })

    with open(out_dir / "index.tsv", "w", newline="") as fh:
        if index_rows:
            writer = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(index_rows)

    # GFF coverage audit: how many CDS locus_tags per sample GFF actually
    # ended up represented in the gene folders we just wrote.
    covered_loci: dict[str, set[str]] = {}
    for row in manifest_rows:
        covered_loci.setdefault(row["sample"], set()).add(row["locus_tag"])

    coverage_rows = []
    all_samples = {s for c in clusters for s in c.sample_loci}
    for sample in sorted(all_samples):
        gff_tags = read_gff_cds_locus_tags(gff_dir / f"{sample}.gff3")
        matched = covered_loci.get(sample, set())
        coverage_rows.append({
            "category": category,
            "mge_group": mge_group,
            "sample": sample,
            "gff_cds_count": len(gff_tags),
            "matched_in_gene_fasta": len(matched & gff_tags) if gff_tags else len(matched),
            "unmatched_gff_cds": len(gff_tags - matched) if gff_tags else "",
        })

    stats = {
        "category": category,
        "mge_group": mge_group,
        "n_gene_clusters_total": len(clusters),
        "n_gene_dirs_written": len(index_rows),
        "n_fallback_ref": n_fallback_ref,
        "n_skipped_no_seq": n_skipped_no_seq,
        "n_records_total": len(manifest_rows),
    }
    return manifest_rows, stats, coverage_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=str, default="N44_ST258_results")
    p.add_argument("--out-dir", type=str, default=None, help="default: <results-dir>/plmlof_input")
    p.add_argument("--min-samples", type=int, default=1, help="skip gene clusters present in fewer samples than this")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "plmlof_input"

    all_manifest: list[dict] = []
    all_stats: list[dict] = []
    all_coverage: list[dict] = []

    logger.info("Chromosome ...")
    rows, stats, cov = process_category(
        category="chromosome",
        mge_group="",
        panaroo_dir=results_dir / "panaroo" / "chromosome",
        gff_dir=results_dir / "gff" / "chromosome",
        out_dir=out_dir / "chromosome",
        min_samples=args.min_samples,
    )
    all_manifest.extend(rows)
    all_stats.append(stats)
    all_coverage.extend(cov)
    logger.info("chromosome: %s", stats)

    plasmid_root = results_dir / "panaroo" / "plasmids"
    for mge_dir in sorted(plasmid_root.glob("mge_*")):
        logger.info("Plasmid %s ...", mge_dir.name)
        rows, stats, cov = process_category(
            category="plasmid",
            mge_group=mge_dir.name,
            panaroo_dir=mge_dir,
            gff_dir=results_dir / "gff" / "plasmid",
            out_dir=out_dir / "plasmid" / mge_dir.name,
            min_samples=args.min_samples,
        )
        all_manifest.extend(rows)
        all_stats.append(stats)
        all_coverage.extend(cov)
        logger.info("%s: %s", mge_dir.name, stats)

    with open(out_dir / "manifest_all.tsv", "w", newline="") as fh:
        if all_manifest:
            writer = csv.DictWriter(fh, fieldnames=list(all_manifest[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(all_manifest)

    with open(out_dir / "build_stats.tsv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_stats[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(all_stats)

    with open(out_dir / "gff_coverage_report.tsv", "w", newline="") as fh:
        if all_coverage:
            writer = csv.DictWriter(fh, fieldnames=list(all_coverage[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(all_coverage)

    logger.info("Wrote %d gene folders, %d sample records -> %s", sum(s["n_gene_dirs_written"] for s in all_stats), len(all_manifest), out_dir)


if __name__ == "__main__":
    main()
