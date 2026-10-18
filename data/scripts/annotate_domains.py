#!/usr/bin/env python3
"""Annotate unique WT proteins with approximate domain spans for LoF priors.

Default backend is **MMseqs2** vs Pfam-A seed FASTA (fast; good enough for
domain-geometry labels). Optional ``--backend hmmer`` keeps the old Pfam-HMM
path (cluster → annotate reps → map).

    conda install -c bioconda mmseqs2

    python data/scripts/download_pfam.py --kind seed
    # → data/raw/pfam/Pfam-A.seed.faa (~ungapped seed seqs)

    python data/scripts/annotate_domains.py \
        --gbff-dir data/raw/refseq_gbff \
        --target data/raw/pfam/Pfam-A.seed.faa \
        --out data/processed/pfam_domains.parquet \
        --threads 20 \
        --max-proteins 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from plmlof.domains import protein_id
from plmlof.gbff import iter_cds
from plmlof.hmmer import (
    DOM_EVALUE,
    annotate_proteins,
    annotate_proteins_sharded,
    finalize_domain_hits,
)
from plmlof.mmseqs_domains import annotate_proteins_mmseqs
from plmlof.seqcluster import cluster_proteins, expand_hits_to_members

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Domain spans for LoF priors (MMseqs default)")
    p.add_argument("--gbff-dir", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=("mmseqs", "hmmer"),
        default="mmseqs",
        help="mmseqs = fast Pfam-seed search (default); hmmer = exact Pfam GA",
    )
    p.add_argument(
        "--target",
        type=Path,
        default=Path("data/raw/pfam/Pfam-A.seed.faa"),
        help="Pfam seed FASTA for --backend mmseqs (from download_pfam.py --kind seed)",
    )
    p.add_argument(
        "--hmm",
        type=Path,
        default=Path("data/raw/pfam/Pfam-A.hmm.gz"),
        help="Pfam-A.hmm[.gz] for --backend hmmer",
    )
    p.add_argument("--out", type=Path, default=Path("data/processed/pfam_domains.parquet"))
    p.add_argument("--threads", type=int, default=20, help="MMseqs threads (mmseqs backend)")
    p.add_argument("--max-evalue", type=float, default=DOM_EVALUE)
    p.add_argument(
        "--sensitivity",
        type=float,
        default=5.0,
        help="MMseqs -s sensitivity (default 5.0; raise for more remote hits)",
    )
    p.add_argument(
        "--max-proteins",
        type=int,
        default=0,
        help="Unique WT CDS to keep (0 = all)",
    )
    # HMMER-only options (ignored for mmseqs)
    p.add_argument("--min-seq-id", type=float, default=0.90)
    p.add_argument("--cluster-cov", type=float, default=0.80)
    p.add_argument("--no-cluster", action="store_true")
    p.add_argument("--shards", type=int, default=20)
    p.add_argument("--cpus-per-shard", type=int, default=1)
    p.add_argument("--cluster-threads", type=int, default=20)
    p.add_argument("--cpus", type=int, default=None)
    args = p.parse_args()

    files = sorted(args.gbff_dir.glob("*.gbff*"))
    if not files:
        raise SystemExit(f"No GBFFs in {args.gbff_dir}")

    seen: dict[str, str] = {}
    n_cds = 0
    for gbff in tqdm(files, desc="GBFF", unit="genome", dynamic_ncols=True):
        cds = iter_cds(gbff)
        n_cds += len(cds)
        for _, _, prot, _ in cds:
            seen.setdefault(protein_id(prot), prot)
            if args.max_proteins and len(seen) >= args.max_proteins:
                break
        if args.max_proteins and len(seen) >= args.max_proteins:
            logger.info("Hit --max-proteins %s", args.max_proteins)
            break

    pairs = list(seen.items())
    seq_by_id = dict(pairs)
    logger.info("Collected %s CDS → %s unique sequences", n_cds, len(pairs))

    if args.backend == "mmseqs":
        if not args.target.exists():
            raise SystemExit(
                f"Target FASTA missing: {args.target}\n"
                "  python data/scripts/download_pfam.py --kind fasta"
            )
        raw = annotate_proteins_mmseqs(
            pairs,
            args.target,
            threads=args.threads,
            max_evalue=args.max_evalue,
            sensitivity=args.sensitivity,
        )
    else:
        if not args.hmm.exists():
            raise SystemExit(
                f"HMM missing: {args.hmm}\n"
                "  python data/scripts/download_pfam.py --kind hmm"
            )
        if args.no_cluster:
            member_to_rep = {pid: pid for pid, _ in pairs}
            rep_pairs = pairs
        else:
            member_to_rep = cluster_proteins(
                pairs,
                min_seq_id=args.min_seq_id,
                coverage=args.cluster_cov,
                threads=args.cluster_threads,
            )
            rep_ids = sorted(set(member_to_rep.values()))
            rep_pairs = [(rid, seq_by_id[rid]) for rid in rep_ids if rid in seq_by_id]
            logger.info("HMMER on %s representatives", len(rep_pairs))

        shards = max(1, int(args.shards))
        cpus_per = max(1, int(args.cpus_per_shard))
        if shards == 1:
            cpus = int(args.cpus) if args.cpus is not None else cpus_per
            raw_reps = annotate_proteins(rep_pairs, args.hmm, cpus=cpus, backend="auto")
        else:
            raw_reps = annotate_proteins_sharded(
                rep_pairs, args.hmm, shards=shards, cpus_per_shard=cpus_per,
            )
        if args.no_cluster:
            raw = raw_reps
        else:
            rep_hits = finalize_domain_hits(raw_reps, max_evalue=args.max_evalue)
            raw = expand_hits_to_members(rep_hits, member_to_rep, seq_by_id)

    if not raw:
        raise SystemExit("No domain hits. Check mmseqs/target (or HMMER) setup.")

    hits = finalize_domain_hits(raw, max_evalue=args.max_evalue)
    if not hits:
        raise SystemExit(
            f"No hits survived e-value≤{args.max_evalue:g} + non-overlap filter"
        )

    df = pd.DataFrame(hits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info(
        "Wrote %s domain rows on %s / %s proteins → %s",
        len(df),
        df["protein_id"].nunique(),
        len(pairs),
        args.out,
    )


if __name__ == "__main__":
    main()
