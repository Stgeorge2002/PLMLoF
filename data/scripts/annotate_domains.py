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

from plmlof.domains import protein_id
from plmlof.gbff import iter_cds
from plmlof.hmmer import annotate_proteins

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Pfam/HMMER domain coordinates for synthetic LoF")
    p.add_argument("--gbff-dir", type=Path, required=True)
    p.add_argument("--hmm", type=Path, required=True, help="Pfam-A.hmm or Pfam-A.hmm.gz")
    p.add_argument("--out", type=Path, default=Path("data/processed/pfam_domains.parquet"))
    p.add_argument("--cpus", type=int, default=20)
    p.add_argument(
        "--max-proteins",
        type=int,
        default=25_000,
        help="Unique WT CDS to scan (0 = all). 150 genomes is ~5e5 unique; that is days of hmmscan.",
    )
    args = p.parse_args()

    if not args.hmm.exists():
        raise SystemExit(f"HMM not found: {args.hmm}\n  python data/scripts/download_pfam.py")

    files = sorted(args.gbff_dir.glob("*.gbff*"))
    if not files:
        raise SystemExit(f"No GBFFs in {args.gbff_dir}")

    seen: dict[str, str] = {}
    for i, gbff in enumerate(files, 1):
        cds = iter_cds(gbff)
        logger.info("[%s/%s] %s CDS=%s unique=%s", i, len(files), gbff.name, len(cds), len(seen))
        for _, _, prot, _ in cds:
            seen.setdefault(protein_id(prot), prot)
            if args.max_proteins and len(seen) >= args.max_proteins:
                break
        if args.max_proteins and len(seen) >= args.max_proteins:
            logger.info("Hit --max-proteins %s", args.max_proteins)
            break

    pairs = list(seen.items())
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
