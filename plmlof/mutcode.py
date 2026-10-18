"""MutCode called-consensus reconstruction for PLMLoF.

Applies lossless gene-reference edit codes (MutCode / LoF-matrix
``Mutation_Codes``) to Panaroo ``pan_genome_reference.fa`` sequences.

Reconstruction policy (mutcode-v1):

    cursor = 0
    output = empty
    for [start, delete_length, inserted] in codes:
        append reference[cursor : start]
        append inserted
        cursor = start + delete_length
    append reference[cursor : reference_length]

``start`` is 0-based on the **original** reference; later edits are never
renumbered. Unlisted positions keep the reference base.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Literal IUPAC DNA accepted in MutCode inserted_sequence (either case).
_IUPAC_RE = re.compile(r"^[ACGTRYSWKMBDHVNacgtryswkmbdhvn]*$")

REQUIRED_LOF_COLUMNS = (
    "Replicon",
    "Gene",
    "Sample",
    "Reference_Length",
    "Sample_Length",
    "Mutation_Codes",
)


class MutCodeError(ValueError):
    """Invalid MutCode edit list or LoF-matrix row."""


@dataclass(frozen=True, slots=True)
class Edit:
    start: int
    delete_length: int
    inserted: str

    def validate(self, reference_length: int) -> None:
        if self.start < 0:
            raise MutCodeError(f"edit start {self.start} is negative")
        if self.delete_length < 0:
            raise MutCodeError(f"delete_length {self.delete_length} is negative")
        if self.start > reference_length:
            raise MutCodeError(
                f"edit start {self.start} past reference end {reference_length}"
            )
        end = self.start + self.delete_length
        if end > reference_length:
            raise MutCodeError(
                f"edit [{self.start},{end}) past reference end {reference_length}"
            )
        if self.delete_length == 0 and self.inserted == "":
            raise MutCodeError("empty insertion with delete_length 0 is illegal")
        if not _IUPAC_RE.fullmatch(self.inserted):
            raise MutCodeError(
                f"inserted sequence contains non-IUPAC DNA: {self.inserted!r}"
            )


def parse_codes(raw: str | Sequence[Sequence] | None) -> list[Edit]:
    """Parse a MutCode ``codes`` field into validated Edit objects (no ref length)."""
    if raw is None or raw == "" or raw == "[]":
        return []
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MutCodeError(f"Mutation_Codes is not JSON: {raw!r}") from exc
    else:
        payload = raw
    if not isinstance(payload, (list, tuple)):
        raise MutCodeError(f"codes must be a JSON array, got {type(payload).__name__}")

    edits: list[Edit] = []
    for i, item in enumerate(payload):
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise MutCodeError(f"codes[{i}] must be [start, delete_length, inserted]")
        start, delete_length, inserted = item
        if not isinstance(start, int) or isinstance(start, bool):
            raise MutCodeError(f"codes[{i}].start must be an integer")
        if not isinstance(delete_length, int) or isinstance(delete_length, bool):
            raise MutCodeError(f"codes[{i}].delete_length must be an integer")
        if not isinstance(inserted, str):
            raise MutCodeError(f"codes[{i}].inserted must be a string")
        edits.append(Edit(start, delete_length, inserted))
    return edits


def validate_edits(edits: Sequence[Edit], reference_length: int) -> None:
    """Enforce MutCode ordering / non-overlap rules against a known reference length."""
    prev_end = 0
    seen_starts: set[int] = set()
    for edit in edits:
        edit.validate(reference_length)
        if edit.start in seen_starts:
            raise MutCodeError(f"duplicate edit start {edit.start}")
        seen_starts.add(edit.start)
        # Adjacent edits are legal: next start may equal previous end.
        if edit.start < prev_end:
            raise MutCodeError(
                f"overlapping or out-of-order edit at start={edit.start} "
                f"(previous ended at {prev_end})"
            )
        prev_end = edit.start + edit.delete_length


def expected_target_length(reference_length: int, edits: Sequence[Edit]) -> int:
    deleted = sum(e.delete_length for e in edits)
    inserted = sum(len(e.inserted) for e in edits)
    return reference_length - deleted + inserted


def _as_edits(edits: Sequence[Edit] | str | Sequence[Sequence] | None) -> list[Edit]:
    if edits is None or edits == "" or edits == "[]":
        return []
    if isinstance(edits, (list, tuple)) and (not edits or isinstance(edits[0], Edit)):
        return list(edits)  # type: ignore[arg-type]
    return parse_codes(edits)  # type: ignore[arg-type]


def apply_edits(reference: str, edits: Sequence[Edit] | str | Sequence[Sequence] | None) -> str:
    """Rebuild the called-consensus DNA for one gene.

    ``edits`` may be Edit objects, a MutCode JSON string, a raw JSON array, or None.
    """
    parsed = _as_edits(edits)
    ref_len = len(reference)
    validate_edits(parsed, ref_len)
    if not parsed:
        return reference

    cursor = 0
    parts: list[str] = []
    for edit in parsed:
        parts.append(reference[cursor : edit.start])
        parts.append(edit.inserted)
        cursor = edit.start + edit.delete_length
    parts.append(reference[cursor:ref_len])
    return "".join(parts)


def reconstruct(
    reference: str,
    codes: str | Sequence[Sequence] | Sequence[Edit] | None,
    *,
    reference_length: int | None = None,
    target_length: int | None = None,
) -> str:
    """Apply codes and optionally check declared lengths from a LoF-matrix row."""
    if reference_length is not None and len(reference) != reference_length:
        raise MutCodeError(
            f"reference length {len(reference)} != declared Reference_Length "
            f"{reference_length}"
        )
    edits = _as_edits(codes)
    out = apply_edits(reference, edits)
    if target_length is not None and len(out) != target_length:
        raise MutCodeError(
            f"reconstructed length {len(out)} != declared Sample_Length {target_length} "
            f"(MutCode expected {expected_target_length(len(reference), edits)})"
        )
    return out


def read_fasta(path: Path | str) -> dict[str, str]:
    """FASTA header first token → sequence (case preserved; no whitespace)."""
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
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


def write_fasta(path: Path | str, records: Iterable[tuple[str, str]], *, width: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec_id, seq in records:
            fh.write(f">{rec_id}\n")
            if width <= 0:
                fh.write(seq)
                fh.write("\n")
            else:
                for i in range(0, len(seq), width):
                    fh.write(seq[i : i + width])
                    fh.write("\n")


def mge_group_from_replicon(replicon: str) -> str:
    """``mge_0__SRR...`` → ``mge_0``; ``chromosome`` → ``\"\"``."""
    if replicon == "chromosome":
        return ""
    if replicon.startswith("mge_"):
        return replicon.split("__", 1)[0]
    raise MutCodeError(f"unrecognised Replicon value: {replicon!r}")


def resolve_panaroo_dir(results_dir: Path | str, replicon: str) -> Path:
    """Locate a Panaroo output directory for a LoF-matrix Replicon value.

    Accepts both nested ``.../panaroo_output/`` layouts (N44_ST258) and flat
    ``panaroo/chromosome/`` layouts expected by older helpers.
    """
    results_dir = Path(results_dir)
    if replicon == "chromosome":
        candidates = (
            results_dir / "panaroo" / "chromosome" / "panaroo_output",
            results_dir / "panaroo" / "chromosome",
        )
    else:
        mge = mge_group_from_replicon(replicon)
        candidates = (
            results_dir / "panaroo" / "plasmids" / mge / "panaroo_output",
            results_dir / "panaroo" / "plasmids" / mge,
        )
    for cand in candidates:
        if (cand / "pan_genome_reference.fa").is_file():
            return cand
    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"no pan_genome_reference.fa for replicon {replicon!r}; tried: {checked}"
    )


def category_from_replicon(replicon: str) -> tuple[str, str]:
    """Return ``(category, mge_group)`` for output directory layout."""
    if replicon == "chromosome":
        return "chromosome", ""
    mge = mge_group_from_replicon(replicon)
    return "plasmid", mge
