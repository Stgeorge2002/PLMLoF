"""Run PLMLoF (lof + mlof heads) over every gene produced by
scripts/build_gene_fasta_pairs.py, in a single process.

Loads ESM2 once, walks N44_ST258_results/plmlof_input/{chromosome,plasmid/mge_*}/<gene>/
{ref.fasta,var.fasta,manifest.tsv}, scores everything with one Predictor,
and writes a combined TSV joined back to gene/sample/category via the
per-record manifest rows.

This is NOT run automatically — invoke it yourself once you are ready to
classify, e.g.:

    python scripts/run_predict_batch.py \\
        --input-dir N44_ST258_results/plmlof_input \\
        --model best-models \\
        --output N44_ST258_results/plmlof_input/predictions.tsv \\
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=str, default="N44_ST258_results/plmlof_input")
    p.add_argument("--model", type=str, required=True, help="Output dir with lof/ mlof/ (best-models)")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--domains", type=str, default=None)
    p.add_argument("--hmm", type=str, default=None)
    p.add_argument("--hmm-cpus", type=int, default=1)
    return p.parse_args()


def iter_gene_dirs(input_dir: Path):
    chrom = input_dir / "chromosome"
    if chrom.exists():
        for gene_dir in sorted(chrom.iterdir()):
            if (gene_dir / "manifest.tsv").exists():
                yield gene_dir
    plasmid_root = input_dir / "plasmid"
    if plasmid_root.exists():
        for mge_dir in sorted(plasmid_root.glob("mge_*")):
            for gene_dir in sorted(mge_dir.iterdir()):
                if (gene_dir / "manifest.tsv").exists():
                    yield gene_dir


def load_manifest_rows(input_dir: Path) -> dict[str, dict]:
    """record_id -> manifest row, read once from the global manifest_all.tsv."""
    manifest_path = input_dir / "manifest_all.tsv"
    rows: dict[str, dict] = {}
    with open(manifest_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows[row["record_id"]] = row
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    import torch

    from plmlof.inference.vcf_handler import parse_fasta_pairs
    from plmlof.predictor import Predictor

    input_dir = Path(args.input_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    manifest_rows = load_manifest_rows(input_dir)

    records = []
    for gene_dir in iter_gene_dirs(input_dir):
        ref_fasta = gene_dir / "ref.fasta"
        var_fasta = gene_dir / "var.fasta"
        records.extend(parse_fasta_pairs(ref_fasta, var_fasta))
    logger.info("Loaded %d ref/var records from %s", len(records), input_dir)

    pred = Predictor(
        Path(args.model), device=device, batch_size=args.batch_size,
        domains=args.domains, hmm=args.hmm, hmm_cpus=args.hmm_cpus,
    )
    results = pred.predict_records(records, pair=True)

    out_path = Path(args.output)
    extra_cols = ["category", "mge_group", "gene_id", "sample", "locus_tag", "gene_name"]
    base_cols = [
        "gene", "wreck", "wreck_kind", "wreck_prior",
        "lof_score", "lof_sd", "lof_p", "lof_q",
        "mlof_score", "mlof_bin", "mlof_mean", "mlof_n_sites", "mlof_sd", "mlof_p", "mlof_q",
        "in_family", "nearest_train_gene", "ref_cosine",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(extra_cols + base_cols)
        for r in results:
            m = manifest_rows.get(r["gene"], {})
            writer.writerow(
                [m.get(c, "") for c in extra_cols]
                + [r.get(c, "") for c in base_cols]
            )
    logger.info("Wrote %d predictions -> %s", len(results), out_path)


if __name__ == "__main__":
    main()
