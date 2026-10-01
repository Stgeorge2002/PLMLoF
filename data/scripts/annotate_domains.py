"""Annotate unique WT proteins with Pfam domains (laptop, not Isambard).

    python data/scripts/annotate_domains.py \
        --gbff-dir data/raw/refseq_gbff \
        --hmm data/raw/pfam/Pfam-A.hmm.gz \
        --out data/processed/pfam_domains.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from plmlof.v2.gbff import iter_cds
from plmlof.v2.hmmer import annotate_proteins, unique_protein_pairs

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Pfam/HMMER domain coordinates for synthetic LoF")
    p.add_argument("--gbff-dir", type=Path, required=True)
    p.add_argument("--hmm", type=Path, required=True, help="Pfam-A.hmm or Pfam-A.hmm.gz")
    p.add_argument("--out", type=Path, default=Path("data/processed/pfam_domains.parquet"))
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--max-proteins", type=int, default=0, help="0 = all unique CDS")
    args = p.parse_args()

    if not args.hmm.exists():
        raise SystemExit(f"HMM not found: {args.hmm}\n  python data/scripts/download_pfam.py")

    files = sorted(args.gbff_dir.glob("*.gbff*"))
    if not files:
        raise SystemExit(f"No GBFFs in {args.gbff_dir}")

    records = []
    for i, gbff in enumerate(files, 1):
        cds = iter_cds(gbff)
        logger.info("[%s/%s] %s CDS=%s", i, len(files), gbff.name, len(cds))
        records.extend(cds)

    pairs = unique_protein_pairs(records)
    if args.max_proteins:
        pairs = pairs[: args.max_proteins]
    logger.info("Unique proteins to scan: %s", len(pairs))

    hits = annotate_proteins(pairs, args.hmm, cpus=args.cpus)
    if not hits:
        raise SystemExit("No Pfam hits. Check the HMM file and that pyhmmer or hmmscan is installed.")

    df = pd.DataFrame(hits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    n_prot = df["protein_id"].nunique()
    logger.info("Wrote %s domain hits on %s proteins → %s", len(df), n_prot, args.out)


if __name__ == "__main__":
    main()
