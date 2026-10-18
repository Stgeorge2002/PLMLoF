"""Cluster proteins and map Pfam spans from representatives onto members.

Used to cut HMMER cost: annotate cluster reps only, then transfer domain
coordinates through a pairwise alignment (identity transfer when lengths match).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from Bio.Align import PairwiseAligner
from tqdm import tqdm

from plmlof.hmmer import write_fasta

logger = logging.getLogger(__name__)


def cluster_proteins(
    pairs: list[tuple[str, str]],
    *,
    min_seq_id: float = 0.90,
    coverage: float = 0.80,
    threads: int = 20,
    work_dir: Path | None = None,
) -> dict[str, str]:
    """Return member_id → representative_id (reps map to themselves).

    Prefers MMseqs2 ``easy-cluster``; falls back to CD-HIT if installed.
    """
    if not pairs:
        return {}
    if shutil.which("mmseqs"):
        return _cluster_mmseqs(
            pairs,
            min_seq_id=min_seq_id,
            coverage=coverage,
            threads=threads,
            work_dir=work_dir,
        )
    if shutil.which("cd-hit"):
        logger.warning("mmseqs not found; falling back to cd-hit")
        return _cluster_cdhit(
            pairs,
            min_seq_id=min_seq_id,
            coverage=coverage,
            threads=threads,
            work_dir=work_dir,
        )
    raise FileNotFoundError(
        "Need MMseqs2 or CD-HIT on PATH for cluster→annotate-reps.\n"
        "  conda install -c bioconda mmseqs2\n"
        "  # or: conda install -c bioconda cd-hit"
    )


def _cluster_mmseqs(
    pairs: list[tuple[str, str]],
    *,
    min_seq_id: float,
    coverage: float,
    threads: int,
    work_dir: Path | None,
) -> dict[str, str]:
    own_tmp = work_dir is None
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="plmlof_mmseqs_"))
    root.mkdir(parents=True, exist_ok=True)
    fasta = root / "all.faa"
    out_prefix = root / "clu"
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    write_fasta(pairs, fasta)
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta), str(out_prefix), str(tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "1",
        "--threads", str(max(1, threads)),
        "-v", "1",
    ]
    logger.info("MMseqs2 easy-cluster id≥%.2f cov≥%.2f threads=%s", min_seq_id, coverage, threads)
    subprocess.run(cmd, check=True)
    tsv = Path(str(out_prefix) + "_cluster.tsv")
    if not tsv.exists():
        # some mmseqs versions write clusterRes_cluster.tsv from the prefix name
        candidates = list(root.glob("*_cluster.tsv"))
        if not candidates:
            raise FileNotFoundError(f"MMseqs cluster TSV missing under {root}")
        tsv = candidates[0]
    member_to_rep: dict[str, str] = {}
    with tsv.open("rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            member_to_rep[member] = rep
    for pid, _ in pairs:
        member_to_rep.setdefault(pid, pid)
    n_rep = len(set(member_to_rep.values()))
    logger.info(
        "Clustered %s sequences → %s representatives (%.1fx fewer HMMER queries)",
        len(pairs),
        n_rep,
        len(pairs) / max(n_rep, 1),
    )
    if own_tmp:
        shutil.rmtree(root, ignore_errors=True)
    return member_to_rep


def _cluster_cdhit(
    pairs: list[tuple[str, str]],
    *,
    min_seq_id: float,
    coverage: float,
    threads: int,
    work_dir: Path | None,
) -> dict[str, str]:
    own_tmp = work_dir is None
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="plmlof_cdhit_"))
    root.mkdir(parents=True, exist_ok=True)
    fasta = root / "all.faa"
    out = root / "clu"
    write_fasta(pairs, fasta)
    # cd-hit word size depends on identity threshold
    if min_seq_id >= 0.7:
        word = 5
    elif min_seq_id >= 0.6:
        word = 4
    else:
        word = 3
    cmd = [
        "cd-hit",
        "-i", str(fasta),
        "-o", str(out),
        "-c", str(min_seq_id),
        "-aS", str(coverage),
        "-n", str(word),
        "-T", str(max(1, threads)),
        "-M", "0",
        "-d", "0",
    ]
    logger.info("CD-HIT -c %.2f threads=%s", min_seq_id, threads)
    subprocess.run(cmd, check=True)
    clstr = Path(str(out) + ".clstr")
    member_to_rep: dict[str, str] = {}
    rep: str | None = None
    with clstr.open("rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">Cluster"):
                rep = None
                continue
            # 0	123aa, >abc123... *
            if ">" not in line:
                continue
            name = line.split(">", 1)[1]
            name = name.split("...")[0]
            if line.endswith("*"):
                rep = name
                member_to_rep[name] = name
            elif rep is not None:
                member_to_rep[name] = rep
    for pid, _ in pairs:
        member_to_rep.setdefault(pid, pid)
    n_rep = len(set(member_to_rep.values()))
    logger.info(
        "Clustered %s sequences → %s representatives (%.1fx fewer HMMER queries)",
        len(pairs),
        n_rep,
        len(pairs) / max(n_rep, 1),
    )
    if own_tmp:
        shutil.rmtree(root, ignore_errors=True)
    return member_to_rep


def _rep_to_mem_index(rep: str, mem: str) -> list[int | None]:
    """For each 0-based rep residue, the aligned 0-based member residue (or None)."""
    if rep == mem:
        return list(range(len(rep)))
    if len(rep) == len(mem):
        # Same length at ≥cluster identity: 1:1 is almost always correct and far cheaper.
        return list(range(len(rep)))

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -0.5
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    aln = next(aligner.align(rep, mem))
    # Biopython 1.81+: alignment.indices → (seqA_idx, seqB_idx) arrays with -1 for gaps
    try:
        a_idx, b_idx = aln.indices
        mapping: list[int | None] = [None] * len(rep)
        for i, j in zip(a_idx, b_idx):
            if i < 0 or j < 0:
                continue
            mapping[int(i)] = int(j)
        return mapping
    except Exception:
        pass

    # Fallback: aligned strings
    try:
        a_str = aln[0]
        b_str = aln[1]
    except Exception:
        return [None] * len(rep)
    mapping = [None] * len(rep)
    ri = mi = 0
    for ca, cb in zip(a_str, b_str):
        if ca != "-" and cb != "-":
            mapping[ri] = mi
            ri += 1
            mi += 1
        elif ca == "-":
            mi += 1
        else:
            ri += 1
    return mapping


def map_span_to_member(
    rep_seq: str,
    mem_seq: str,
    start_1: int,
    end_1: int,
) -> tuple[int, int] | None:
    """Map a 1-based inclusive span on the rep onto the member sequence."""
    if start_1 < 1 or end_1 < start_1 or end_1 > len(rep_seq):
        return None
    idx = _rep_to_mem_index(rep_seq, mem_seq)
    mapped = [idx[i] for i in range(start_1 - 1, end_1) if idx[i] is not None]
    if not mapped:
        return None
    return min(mapped) + 1, max(mapped) + 1


def expand_hits_to_members(
    rep_hits: list[dict],
    member_to_rep: dict[str, str],
    seq_by_id: dict[str, str],
) -> list[dict]:
    """Copy / map representative domain hits onto every cluster member."""
    hits_by_rep: dict[str, list[dict]] = defaultdict(list)
    for hit in rep_hits:
        hits_by_rep[str(hit["protein_id"])].append(hit)

    members_by_rep: dict[str, list[str]] = defaultdict(list)
    for member, rep in member_to_rep.items():
        members_by_rep[rep].append(member)

    out: list[dict] = []
    reps = sorted(hits_by_rep)
    for rep in tqdm(reps, desc="Map domains", unit="rep", dynamic_ncols=True):
        rep_seq = seq_by_id.get(rep, "")
        if not rep_seq:
            continue
        rep_hit_list = hits_by_rep[rep]
        for member in members_by_rep.get(rep, [rep]):
            mem_seq = seq_by_id.get(member, "")
            if not mem_seq:
                continue
            if member == rep:
                for hit in rep_hit_list:
                    row = dict(hit)
                    row["protein_id"] = member
                    row["cluster_rep"] = rep
                    out.append(row)
                continue
            for hit in rep_hit_list:
                mapped = map_span_to_member(
                    rep_seq, mem_seq, int(hit["start"]), int(hit["end"]),
                )
                if mapped is None:
                    continue
                start, end = mapped
                if end < start or end > len(mem_seq):
                    continue
                row = dict(hit)
                row["protein_id"] = member
                row["start"] = start
                row["end"] = end
                row["cluster_rep"] = rep
                out.append(row)
    return out
