"""Attribution module for explaining PLMLoF predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from plmlof.utils.sequence_utils import find_mutations

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Attribution result for a single prediction."""
    gene: str
    prediction: str  # LoF / WT / GoF
    confidence: float
    position_scores: list[float]  # Per-amino-acid importance scores
    top_positions: list[dict]  # Top contributing positions with details
    summary: str  # Human-readable explanation


def compute_rule_based_attribution(
    ref_protein: str,
    var_protein: str,
) -> list[dict]:
    """Compute rule-based attribution by identifying mutation positions.

    Returns a list of annotated mutation positions with their likely impact.
    """
    mutations = find_mutations(ref_protein, var_protein)

    annotated = []
    for mut in mutations:
        impact = "unknown"
        if mut["type"] == "nonsense":
            impact = "high"
        elif mut["type"] == "truncation":
            impact = "high"
        elif mut["type"] == "missense":
            # Very rough heuristic: position in protein
            pos_frac = mut["position"] / max(len(ref_protein), 1)
            if pos_frac < 0.1 or pos_frac > 0.9:
                impact = "moderate"
            else:
                impact = "moderate"
        elif mut["type"] == "readthrough":
            impact = "moderate"
        elif mut["type"] == "extension":
            impact = "low"

        annotated.append({
            "position": mut["position"],
            "ref_aa": mut["ref_aa"],
            "var_aa": mut["var_aa"],
            "type": mut["type"],
            "impact": impact,
        })

    return annotated


def generate_attribution(
    gene: str,
    prediction: str,
    confidence: float,
    ref_protein: str,
    var_protein: str,
    gradient_scores: torch.Tensor | None = None,
) -> AttributionResult:
    """Generate a complete attribution result combining gradient and rule-based evidence.

    Args:
        gene: Gene name.
        prediction: Predicted class (LoF/WT/GoF).
        confidence: Prediction confidence.
        ref_protein: Reference protein sequence.
        var_protein: Variant protein sequence.
        gradient_scores: Per-position gradient attribution scores (optional).

    Returns:
        AttributionResult with explanation.
    """
    # Rule-based mutation detection
    mutations = compute_rule_based_attribution(ref_protein, var_protein)

    # Position scores: use gradient if available, otherwise uniform
    max_len = max(len(ref_protein), len(var_protein), 1)
    if gradient_scores is not None:
        position_scores = gradient_scores.detach().cpu().tolist()
        # Pad or truncate to match protein length
        if len(position_scores) < max_len:
            position_scores.extend([0.0] * (max_len - len(position_scores)))
        position_scores = position_scores[:max_len]
    else:
        position_scores = [0.0] * max_len
        # Set high scores at mutation positions
        for mut in mutations:
            pos = mut["position"] - 1  # 0-based
            if 0 <= pos < max_len:
                score = 1.0 if mut["impact"] == "high" else 0.5
                position_scores[pos] = score

    # Top positions
    top_positions = sorted(mutations, key=lambda m: {"high": 3, "moderate": 2, "low": 1, "unknown": 0}[m["impact"]], reverse=True)[:10]

    # Generate summary
    summary = _generate_summary(gene, prediction, confidence, mutations, ref_protein, var_protein)

    return AttributionResult(
        gene=gene,
        prediction=prediction,
        confidence=confidence,
        position_scores=position_scores,
        top_positions=top_positions,
        summary=summary,
    )


def _generate_summary(
    gene: str,
    prediction: str,
    confidence: float,
    mutations: list[dict],
    ref_protein: str,
    var_protein: str,
) -> str:
    """Generate a human-readable summary of the attribution."""
    parts = [f"{gene}: {prediction} (confidence={confidence:.2%})"]

    high_impact = [m for m in mutations if m["impact"] == "high"]
    mod_impact = [m for m in mutations if m["impact"] == "moderate"]

    if not mutations:
        parts.append("No mutations detected — wildtype sequence.")
        return " | ".join(parts)

    if high_impact:
        for m in high_impact[:3]:
            if m["type"] == "nonsense":
                parts.append(f"Premature stop at position {m['position']} ({m['ref_aa']}→*)")
            elif m["type"] == "truncation":
                frac = 1 - len(var_protein) / max(len(ref_protein), 1)
                parts.append(f"Truncation: {frac:.0%} of protein lost")
            elif m["type"] == "frameshift":
                parts.append("Frameshift mutation detected")

    if mod_impact:
        parts.append(f"{len(mod_impact)} missense mutation(s)")

    return " | ".join(parts)
