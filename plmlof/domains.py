"""Pfam/HMMER domain geometry for synthetic LoF priors.

Stops and frameshifts stay rules. Domain coordinates only change how hard
the label is: early or in-domain wreck = sure; post-domain tail = less sure.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from plmlof.gbff import AA, MAX_AA
from plmlof.utils.sequence_utils import translate_dna

# 1-based protein coordinates. Priors are labels, not biology.
# Scores scale with how many domains / how much domain mass is destroyed.
PRIOR_SURE = 1.00
PRIOR_INDOMAIN_INDEL = 0.90
PRIOR_DOMAIN_DROP = 0.70
PRIOR_INDOMAIN_MISSENSE = 0.70
PRIOR_TAIL_STOP = 0.35  # max when zero domain residues are hit (long pure tail)
PRIOR_TAIL_MIN = 0.05   # tiny pure-tail indel
PRIOR_NTERM_INDEL = 0.25  # indel entirely before first domain (domains intact)
PRIOR_TAIL_WEAK = 0.15
PRIOR_EXTRA_MISSENSE = 0.15  # unused in LoF synth; kept for API compat
SURE_MIN = 0.80
TAIL_FRACTION = 0.10

CONSERVATIVE = {
    "A": "GSTV", "G": "AS", "S": "ATN", "T": "AS",
    "D": "EN", "E": "DQ", "N": "DSQ", "Q": "EN",
    "I": "LMV", "L": "IMV", "M": "ILV", "V": "ILM",
    "F": "YW", "Y": "FW", "W": "FY",
    "K": "R", "R": "KH", "H": "KR",
    "C": "S", "P": "A",
}


def protein_id(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]


def merge_intervals(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted((int(s), int(e)) for s, e in spans if int(e) >= int(s))
    out = [ordered[0]]
    for start, end in ordered[1:]:
        prev_s, prev_e = out[-1]
        if start <= prev_e + 1:
            out[-1] = (prev_s, max(prev_e, end))
        else:
            out.append((start, end))
    return out


def last_domain_end(domains: Sequence[tuple[int, int]]) -> int | None:
    if not domains:
        return None
    return max(e for _, e in domains)


def first_domain_start(domains: Sequence[tuple[int, int]]) -> int | None:
    if not domains:
        return None
    return min(s for s, _ in domains)


def classify_position(pos: int, domains: Sequence[tuple[int, int]]) -> str:
    """Where a 1-based residue sits relative to Pfam spans."""
    if not domains:
        return "unknown"
    if pos < first_domain_start(domains):
        return "pre_domain"
    if pos > last_domain_end(domains):
        return "tail"
    for start, end in domains:
        if start <= pos <= end:
            return "in_domain"
    return "linker"


def in_c_terminal_tail(
    pos: int,
    protein_len: int,
    domains: Sequence[tuple[int, int]] = (),
) -> bool:
    if protein_len <= 0:
        return False
    if domains:
        return pos > last_domain_end(domains)
    return pos > int((1.0 - TAIL_FRACTION) * protein_len)


def extra_domain_positions(protein_len: int, domains: Sequence[tuple[int, int]]) -> list[int]:
    """0-based indices not covered by any Pfam span (N-term, linker, C-term)."""
    covered = [False] * protein_len
    for start, end in domains:
        for i in range(max(0, start - 1), min(protein_len, end)):
            covered[i] = True
    return [i for i, hit in enumerate(covered) if not hit]


def domain_positions(protein_len: int, domains: Sequence[tuple[int, int]]) -> list[int]:
    idx: list[int] = []
    for start, end in domains:
        for i in range(max(0, start - 1), min(protein_len, end)):
            idx.append(i)
    return idx


def _unknown_geom(pos: int, protein_len: int) -> str:
    return "unknown_tail" if in_c_terminal_tail(pos, protein_len, ()) else "unknown_early"


def _domains_fully_covered(start: int, end: int, domains: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(s, e) for s, e in domains if start <= s and e <= end]


def _domains_overlapping(start: int, end: int, domains: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(s, e) for s, e in domains if not (end < s or start > e)]


def _domain_residue_count(domains: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, e - s + 1) for s, e in domains)


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo + 1)


def _affected_domain_residues(
    start: int,
    end: int,
    domains: Sequence[tuple[int, int]],
) -> int:
    return sum(_overlap_len(start, end, s, e) for s, e in domains)


def _pure_tail_score(affected_len: int, protein_len: int) -> float:
    """Events that hit zero domain residues stay low; longer tails a bit higher."""
    if protein_len <= 0 or affected_len <= 0:
        return PRIOR_TAIL_MIN
    frac = min(1.0, affected_len / max(protein_len, 1))
    return PRIOR_TAIL_MIN + (PRIOR_TAIL_STOP - PRIOR_TAIL_MIN) * min(1.0, frac / 0.20)


def lof_prior(
    event: str,
    first_affected: int,
    protein_len: int,
    domains: Sequence[tuple[int, int]] = (),
    last_affected: int | None = None,
    n_domains_dropped: int = 0,
    n_domains_remaining: int | None = None,
) -> tuple[float, str]:
    """Map a wreck onto the synthetic LoF prior from domain impact.

    event is one of: stop, frameshift, indel, domain_drop, missense.

    Score tracks how many domains are hit / fully removed and what fraction of
    domain residue mass is destroyed. Pure C-terminal tails that leave every
    domain intact stay low.
    """
    last = last_affected if last_affected is not None else first_affected
    if first_affected < 1:
        first_affected = 1
    if last < first_affected:
        last = first_affected
    if protein_len < 1:
        return PRIOR_SURE, "unknown"

    if event in {"stop", "frameshift"}:
        aff_start, aff_end = first_affected, protein_len
    else:
        aff_start, aff_end = first_affected, last

    geom = (
        classify_position(first_affected, domains)
        if domains
        else _unknown_geom(first_affected, protein_len)
    )

    if event == "missense":
        # Not emitted in LoF synthetic generation; kept for callers/tests.
        if geom == "in_domain":
            return PRIOR_INDOMAIN_MISSENSE, geom
        return PRIOR_EXTRA_MISSENSE, geom if geom != "unknown_early" else "unknown_extra"

    if not domains:
        if geom in {"tail", "unknown_tail"}:
            return _pure_tail_score(aff_end - aff_start + 1, protein_len), geom
        return PRIOR_SURE, geom

    n_dom = len(domains)
    total_dom_res = max(_domain_residue_count(domains), 1)
    hit = _domains_overlapping(aff_start, aff_end, domains)
    full = _domains_fully_covered(aff_start, aff_end, domains)
    hit_res = _affected_domain_residues(aff_start, aff_end, domains)
    n_hit = len(hit)
    n_full = len(full)

    if event == "domain_drop":
        dropped = n_domains_dropped if n_domains_dropped > 0 else max(n_full, 1)
        remaining = n_domains_remaining
        if remaining is None:
            remaining = max(n_dom - dropped, 0)
        if remaining <= 0:
            return PRIOR_SURE, "in_domain"
        frac_n = dropped / max(n_dom, 1)
        frac_res = hit_res / total_dom_res
        score = 0.45 + 0.40 * frac_n + 0.15 * frac_res
        return float(min(PRIOR_SURE, max(PRIOR_DOMAIN_DROP * 0.8, score))), "in_domain"

    if n_hit == 0:
        aff_len = aff_end - aff_start + 1
        first_start = first_domain_start(domains)
        last_end = last_domain_end(domains)
        if first_start is not None and aff_end < first_start:
            if event in {"stop", "frameshift"}:
                return PRIOR_SURE, "pre_domain"
            return PRIOR_NTERM_INDEL, "pre_domain"
        if last_end is not None and aff_start > last_end:
            return _pure_tail_score(aff_len, protein_len), "tail"
        if geom in {"tail", "unknown_tail"}:
            return _pure_tail_score(aff_len, protein_len), "tail"
        return PRIOR_TAIL_MIN, "linker"

    frac_res = hit_res / total_dom_res
    frac_n = n_full / max(n_dom, 1) if n_full else n_hit / max(n_dom, 1)

    if event in {"stop", "frameshift"}:
        intact_res = _domain_residue_count(
            [(s, e) for s, e in domains if e < first_affected]
        )
        if intact_res == 0:
            return PRIOR_SURE, geom if geom != "unknown" else "in_domain"
        destroyed = 1.0 - intact_res / total_dom_res
        score = 0.55 + 0.45 * destroyed
        return float(min(PRIOR_SURE, score)), geom

    score = 0.40 + 0.45 * frac_res + 0.15 * frac_n
    if n_full >= n_dom and n_dom > 0:
        score = PRIOR_SURE
    return float(min(PRIOR_SURE, max(0.40, score))), geom


def load_domain_map(path: Path) -> dict[str, list[tuple[int, int]]]:
    """protein_id → sorted non-overlapping Pfam intervals (1-based inclusive).

    Adjacent domains stay separate so priors can count how many are impacted.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    required = {"protein_id", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    out: dict[str, list[tuple[int, int]]] = {}
    for key, sub in df.groupby("protein_id"):
        spans = sorted(
            (int(s), int(e))
            for s, e in zip(sub["start"].tolist(), sub["end"].tolist())
            if int(e) >= int(s)
        )
        out[str(key)] = spans
    return out


class DomainIndex:
    """Pfam spans for wreck_grade. Parquet first; optional live HMMER for misses."""

    def __init__(
        self,
        parquet: Path | None = None,
        hmm: Path | None = None,
        cpus: int = 1,
    ):
        self.hmm = Path(hmm) if hmm is not None else None
        self.cpus = max(1, int(cpus))
        self._cache: dict[str, list[tuple[int, int]]] = {}
        if parquet is not None and Path(parquet).exists():
            self._cache.update(load_domain_map(Path(parquet)))

    def spans(self, protein: str) -> list[tuple[int, int]]:
        seq = (protein or "").replace("*", "")
        if not seq:
            return []
        pid = protein_id(seq)
        if pid in self._cache:
            return self._cache[pid]
        return []

    def ensure(self, proteins: Sequence[str]) -> None:
        """Fill cache for unique proteins. One HMMER pass for anything not in parquet."""
        missing: list[tuple[str, str]] = []
        seen: set[str] = set()
        for protein in proteins:
            seq = (protein or "").replace("*", "")
            if not seq:
                continue
            pid = protein_id(seq)
            if pid in self._cache or pid in seen:
                continue
            seen.add(pid)
            missing.append((pid, seq))
        if not missing:
            return
        if self.hmm is None or not self.hmm.exists():
            for pid, _ in missing:
                self._cache.setdefault(pid, [])
            return
        from plmlof.hmmer import annotate_proteins, finalize_domain_hits

        hits = finalize_domain_hits(annotate_proteins(missing, self.hmm, cpus=self.cpus))
        by: dict[str, list[tuple[int, int]]] = {}
        for hit in hits:
            by.setdefault(str(hit["protein_id"]), []).append((int(hit["start"]), int(hit["end"])))
        for pid, _ in missing:
            # Already non-overlapping; keep domains separate (do not merge adjacent).
            self._cache[pid] = sorted(by.get(pid, []), key=lambda se: se[0])


@dataclass(frozen=True)
class WreckEvent:
    kind: str
    var_protein: str
    var_dna: str
    first_affected: int
    lof_score: float
    geometry: str


def _apply_stop(prot: str, cut_0: int, domains, kind: str) -> WreckEvent | None:
    if not (1 <= cut_0 < len(prot)):
        return None
    var = prot[:cut_0]
    if not var or var == prot:
        return None
    first = cut_0 + 1
    score, geom = lof_prior("stop", first, len(prot), domains)
    return WreckEvent(kind, var, "", first, score, geom)


def _frameshift_at(dna: str, aa_pos: int, rng: random.Random) -> tuple[str, str] | None:
    if len(dna) < 12:
        return None
    nuc = max(0, min((aa_pos - 1) * 3, len(dna) - 2))
    if rng.random() < 0.5 and nuc + 1 < len(dna):
        mutant = dna[:nuc] + dna[nuc + 1:]
    else:
        mutant = dna[:nuc] + rng.choice("ATGC") + dna[nuc:]
    if abs(len(mutant) - len(dna)) % 3 == 0:
        return None
    var = translate_dna(mutant, to_stop=True).replace("*", "")
    if not var:
        return None
    return var, mutant


def _apply_frameshift(prot: str, dna: str, aa_pos: int, rng: random.Random, domains, kind: str) -> WreckEvent | None:
    pair = _frameshift_at(dna, aa_pos, rng)
    if pair is None:
        return None
    var, mutant = pair
    if var == prot or not (20 <= len(var) <= MAX_AA):
        return None
    first = max(1, min(aa_pos, len(prot)))
    score, geom = lof_prior("frameshift", first, len(prot), domains)
    return WreckEvent(kind, var, mutant, first, score, geom)


def make_stop_pre_or_in(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    if domains:
        last = last_domain_end(domains)
        cut = rng.randint(1, max(1, last - 1))
        kind = "stop_pre_domain" if cut + 1 < first_domain_start(domains) else "stop_in_domain"
        return _apply_stop(prot, cut, domains, kind)
    cut = rng.randint(max(1, int(0.15 * len(prot))), max(2, int(0.60 * len(prot))))
    return _apply_stop(prot, cut, domains, "stop")


def make_stop_tail(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    if domains:
        last = last_domain_end(domains)
        if last >= len(prot) - 1:
            return None
        cut = rng.randint(last, len(prot) - 1)
        return _apply_stop(prot, cut, domains, "stop_tail")
    floor = max(1, int((1.0 - TAIL_FRACTION) * len(prot)))
    if floor >= len(prot) - 1:
        return None
    cut = rng.randint(floor, len(prot) - 1)
    return _apply_stop(prot, cut, domains, "stop_tail")


def make_frameshift_pre_or_in(prot: str, dna: str, domains, rng: random.Random) -> WreckEvent | None:
    if not dna:
        return None
    if domains:
        last = last_domain_end(domains)
        aa_pos = rng.randint(1, max(1, last))
        kind = "frameshift_pre_domain" if aa_pos < first_domain_start(domains) else "frameshift_in_domain"
        return _apply_frameshift(prot, dna, aa_pos, rng, domains, kind)
    aa_pos = rng.randint(4, max(5, len(prot) // 2))
    return _apply_frameshift(prot, dna, aa_pos, rng, domains, "frameshift")


def make_frameshift_tail(prot: str, dna: str, domains, rng: random.Random) -> WreckEvent | None:
    if not dna:
        return None
    if domains:
        last = last_domain_end(domains)
        if last >= len(prot) - 1:
            return None
        aa_pos = rng.randint(last + 1, len(prot))
        return _apply_frameshift(prot, dna, aa_pos, rng, domains, "frameshift_tail")
    floor = max(1, int((1.0 - TAIL_FRACTION) * len(prot)) + 1)
    if floor >= len(prot):
        return None
    aa_pos = rng.randint(floor, len(prot))
    return _apply_frameshift(prot, dna, aa_pos, rng, domains, "frameshift_tail")


def make_indel_in_domain(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    if domains:
        start, end = rng.choice(list(domains))
        span = max(8, int(0.35 * (end - start + 1)))
        lo = start - 1
        hi = max(lo + 1, end - span)
        cut0 = rng.randint(lo, hi) if hi > lo else lo
        var = prot[:cut0] + prot[cut0 + span:]
        if not var or var == prot:
            return None
        first = cut0 + 1
        last = min(len(prot), cut0 + span)
        score, geom = lof_prior("indel", first, len(prot), domains, last_affected=last)
        return WreckEvent("indel_in_domain", var, "", first, score, geom)
    span = rng.randint(int(0.30 * len(prot)), int(0.70 * len(prot)))
    start = rng.randint(1, max(1, len(prot) - span - 1))
    var = prot[:start] + prot[start + span:]
    first = start + 1
    score, geom = lof_prior("indel", first, len(prot), (), last_affected=start + span)
    return WreckEvent("deletion", var, "", first, score, geom)


def make_indel_tail(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    if domains:
        last = last_domain_end(domains)
        tail_len = len(prot) - last
        if tail_len < 4:
            return None
        span = rng.randint(1, min(8, tail_len - 1))
        start0 = rng.randint(last, len(prot) - span - 1) if len(prot) - span - 1 > last else last
        var = prot[:start0] + prot[start0 + span:]
        first = start0 + 1
        score, geom = lof_prior("indel", first, len(prot), domains, last_affected=start0 + span)
        return WreckEvent("indel_tail", var, "", first, score, geom)
    floor = max(1, int((1.0 - TAIL_FRACTION) * len(prot)))
    if floor >= len(prot) - 2:
        return None
    span = rng.randint(1, min(6, len(prot) - floor - 1))
    start0 = rng.randint(floor, len(prot) - span - 1)
    var = prot[:start0] + prot[start0 + span:]
    first = start0 + 1
    score, geom = lof_prior("indel", first, len(prot), (), last_affected=start0 + span)
    return WreckEvent("indel_tail", var, "", first, score, geom)


def make_domain_drop(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    if len(domains) < 1:
        return None
    start, end = rng.choice(list(domains))
    var = prot[: start - 1] + prot[end:]
    if not var or var == prot:
        return None
    remaining = len(domains) - 1
    score, geom = lof_prior(
        "domain_drop", start, len(prot), domains,
        last_affected=end, n_domains_dropped=1, n_domains_remaining=remaining,
    )
    kind = "domain_drop" if remaining > 0 else "indel_in_domain"
    return WreckEvent(kind, var, "", start, score, geom)


def _swap(prot: str, i: int, aa: str) -> str:
    return prot[:i] + aa + prot[i + 1:]


def _radical_aa(wt: str, rng: random.Random) -> str:
    forbid = set(CONSERVATIVE.get(wt, "") + wt)
    choices = [a for a in AA if a not in forbid]
    return rng.choice(choices or [a for a in AA if a != wt])


def _conservative_aa(wt: str, rng: random.Random) -> str | None:
    opts = [a for a in CONSERVATIVE.get(wt, "") if a != wt]
    if not opts:
        return None
    return rng.choice(opts)


def make_missense_in_domain(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    if not domains:
        return None
    idx = domain_positions(len(prot), domains)
    idx = [i for i in idx if i > 0]
    if not idx:
        return None
    i = rng.choice(idx)
    var = _swap(prot, i, _radical_aa(prot[i], rng))
    first = i + 1
    score, geom = lof_prior("missense", first, len(prot), domains)
    return WreckEvent("missense_in_domain", var, "", first, score, geom)


def make_missense_extra_domain(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    if not domains:
        return None
    idx = extra_domain_positions(len(prot), domains)
    idx = [i for i in idx if i > 0]
    if not idx:
        return None
    rng.shuffle(idx)
    for i in idx:
        aa = _conservative_aa(prot[i], rng)
        if aa is None:
            continue
        var = _swap(prot, i, aa)
        first = i + 1
        score, geom = lof_prior("missense", first, len(prot), domains)
        return WreckEvent("missense_extra_domain", var, "", first, score, geom)
    return None


def make_heavy_missense(prot: str, rng: random.Random) -> WreckEvent:
    frac = rng.uniform(0.30, 0.60)
    n = max(1, int(frac * len(prot)))
    positions = rng.sample(range(len(prot)), n)
    chars = list(prot)
    for i in positions:
        chars[i] = rng.choice([a for a in AA if a != chars[i]])
    var = "".join(chars)
    return WreckEvent("heavy_missense", var, "", positions[0] + 1, PRIOR_INDOMAIN_MISSENSE, "unknown_early")


def make_insertion(prot: str, domains, rng: random.Random) -> WreckEvent | None:
    insert_n = rng.randint(max(20, int(0.20 * len(prot))), max(21, int(0.50 * len(prot))))
    insert = "".join(rng.choice(AA) for _ in range(insert_n))
    if domains:
        start, end = rng.choice(list(domains))
        pos = rng.randint(start - 1, end)
    else:
        pos = rng.randint(1, max(1, len(prot) - 1))
    var = (prot[:pos] + insert + prot[pos:])[:MAX_AA]
    first = pos + 1
    last = pos + insert_n
    score, geom = lof_prior("indel", first, len(prot), domains, last_affected=min(last, len(prot)))
    kind = "insertion" if score >= SURE_MIN else "indel_tail"
    return WreckEvent(kind, var, "", first, score, geom)


def sample_events(
    prot: str,
    dna: str,
    domains: Sequence[tuple[int, int]],
    rng: random.Random,
    n: int = 3,
) -> list[WreckEvent]:
    """Per CDS: sure wreck, pure-tail wreck if possible, domain-drop contrast.

    Extra-domain / random missense are not emitted — the LoF head is a wreck
    comparator; missense damage belongs to MLoF.
    """
    sure = [
        lambda: make_stop_pre_or_in(prot, domains, rng),
        lambda: make_frameshift_pre_or_in(prot, dna, domains, rng),
        lambda: make_indel_in_domain(prot, domains, rng),
        lambda: make_insertion(prot, domains, rng),
    ]
    tail = [
        lambda: make_stop_tail(prot, domains, rng),
        lambda: make_frameshift_tail(prot, dna, domains, rng),
        lambda: make_indel_tail(prot, domains, rng),
    ]
    contrast = [
        lambda: make_domain_drop(prot, domains, rng),
        lambda: make_indel_in_domain(prot, domains, rng),
        lambda: make_stop_pre_or_in(prot, domains, rng),
    ]
    buckets = [sure, tail, contrast]
    rng.shuffle(buckets[0])
    rng.shuffle(buckets[1])
    rng.shuffle(buckets[2])
    order = [buckets[0], buckets[1], buckets[2]]
    if n > 3:
        extra = sure + tail + contrast
        rng.shuffle(extra)
        order.extend([[fn] for fn in extra])

    out: list[WreckEvent] = []
    seen: set[str] = set()
    for bucket in order:
        if len(out) >= n:
            break
        for fn in bucket:
            ev = fn()
            if ev is None or not ev.var_protein or ev.var_protein == prot:
                continue
            if not (20 <= len(ev.var_protein) <= MAX_AA):
                continue
            if ev.var_protein in seen:
                continue
            seen.add(ev.var_protein)
            out.append(ev)
            break
    return out
