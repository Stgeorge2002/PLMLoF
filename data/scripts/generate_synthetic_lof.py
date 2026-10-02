"""Generate synthetic LoF protein pairs from bacterial GBFFs.

Stops / frameshifts / indels are rules. If Pfam coordinates are available,
the prior is graded by where the wreck sits relative to domains:

  before or inside a domain  → sure (~1.0)
  strictly after last domain → tail (~0.4)
  in-domain missense vs extra-domain conservative missense is the other contrast

Without HMMER, the last 10% of the ORF is the tail proxy (no NMD in bacteria;
the question is still domain completeness).

    python data/scripts/generate_synthetic_lof.py \
        --gbff-dir data/raw/refseq_gbff \
        --domains data/processed/pfam_domains.parquet \
        --out data/processed/synthetic_lof.parquet
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd

from plmlof.domains import load_domain_map, protein_id, sample_events
from plmlof.gbff import iter_cds

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Synthetic bacterial LoF pairs (domain-graded)")
    p.add_argument("--gbff-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed/synthetic_lof.parquet"))
    p.add_argument("--domains", type=Path, default=None, help="pfam_domains.parquet from annotate_domains.py")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-rows", type=int, default=250_000, help="Hard cap before write")
    p.add_argument("--wrecks-per-gene", type=int, default=3, help="Events per CDS (sure / tail / contrast)")
    args = p.parse_args()

    files = sorted(args.gbff_dir.glob("*.gbff*"))
    if not files:
        raise SystemExit(f"No GBFFs in {args.gbff_dir}")

    domain_map: dict[str, list[tuple[int, int]]] = {}
    if args.domains is not None and args.domains.exists():
        domain_map = load_domain_map(args.domains)
        logger.info("Loaded domain map for %s proteins from %s", len(domain_map), args.domains)
    elif args.domains is not None:
        logger.warning("Domain file %s missing — last 10%% of each ORF is the tail proxy", args.domains)
    else:
        logger.warning("No --domains; last 10%% of each ORF is the tail proxy (not Pfam geometry)")

    rng = random.Random(args.seed)
    rows = []
    n_with_domains = 0
    n_cds = 0
    for i, gbff in enumerate(files, 1):
        cds = iter_cds(gbff)
        logger.info("[%s/%s] %s CDS=%s", i, len(files), gbff.name, len(cds))
        for gene, species, prot, dna in cds:
            n_cds += 1
            spans = domain_map.get(protein_id(prot), [])
            if spans:
                n_with_domains += 1
            for ev in sample_events(prot, dna, spans, rng, n=args.wrecks_per_gene):
                rows.append({
                    "gene": gene,
                    "species": species,
                    "ref_protein": prot,
                    "var_protein": ev.var_protein,
                    "ref_dna": dna if ev.var_dna or "frameshift" in ev.kind else "",
                    "var_dna": ev.var_dna,
                    "wreck_type": ev.kind,
                    "lof_score": ev.lof_score,
                    "geometry": ev.geometry,
                    "first_affected": ev.first_affected,
                    "n_domains": len(spans),
                    "label": 0 if ev.lof_score >= 0.4 else 1,
                    "source": "synthetic_lof",
                    "dms_score": float("nan"),
                    "dms_zscore": float("nan"),
                })
                if len(rows) >= args.max_rows:
                    break
            if len(rows) >= args.max_rows:
                break
        if len(rows) >= args.max_rows:
            logger.info("Hit --max-rows %s", args.max_rows)
            break

    if not rows:
        raise SystemExit("No synthetic rows produced")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("Wrote %s rows → %s", len(df), args.out)
    logger.info("CDS with Pfam spans: %s / %s", n_with_domains, n_cds)
    logger.info("By wreck_type:\n%s", df["wreck_type"].value_counts().to_string())
    logger.info("By geometry:\n%s", df["geometry"].value_counts().to_string())
    logger.info("lof_score mean=%.3f  (tail types should sit well below 1.0)", df["lof_score"].mean())
    logger.info("Species: %s", df["species"].nunique())


if __name__ == "__main__":
    main()
