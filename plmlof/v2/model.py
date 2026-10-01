"""Frozen-ESM comparison + task head for LoF regression or binary GoF."""

from __future__ import annotations

import torch
import torch.nn as nn

from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES
from plmlof.models.comparison import ComparisonModule


class LofScoreHead(nn.Module):
    """Scalar LoF score in [0, 1]."""

    def __init__(self, input_size: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(features).squeeze(-1))


class BinaryLogitHead(nn.Module):
    """Single logit for conservative GoF (sigmoid applied at call time)."""

    def __init__(self, input_size: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


class V2TaskNet(nn.Module):
    """Comparison module + engineered-feature norm + task head.

    Independent copies are trained per task so wreck length features cannot
    poison a GoF caller.
    """

    def __init__(
        self,
        hidden_size: int,
        task: str,
        pool_strategy: str = "mean_max",
        use_cross_attention: bool = False,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
        head_hidden: int = 128,
        dropout: float = 0.2,
        num_nuc_features: int = NUM_NUCLEOTIDE_FEATURES,
    ):
        super().__init__()
        if task not in {"lof", "growth_gof", "amr_gof"}:
            raise ValueError(f"Unknown v2 task: {task}")
        self.task = task
        self.comparison = ComparisonModule(
            hidden_size=hidden_size,
            pool_strategy=pool_strategy,
            use_cross_attention=use_cross_attention,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dropout=cross_attn_dropout,
        )
        self.feature_norm = nn.LayerNorm(num_nuc_features)
        input_size = self.comparison.output_size + num_nuc_features
        if task == "lof":
            self.head = LofScoreHead(input_size, hidden_dim=head_hidden, dropout=dropout)
        else:
            self.head = BinaryLogitHead(input_size, hidden_dim=head_hidden, dropout=dropout)

    def forward_from_pooled(
        self,
        ref_mean: torch.Tensor,
        ref_max: torch.Tensor,
        var_mean: torch.Tensor,
        var_max: torch.Tensor,
        nucleotide_features: torch.Tensor,
    ) -> torch.Tensor:
        comparison = self.comparison.compare_pooled(ref_mean, ref_max, var_mean, var_max)
        nuc = self.feature_norm(nucleotide_features)
        features = torch.cat([comparison, nuc], dim=-1)
        return self.head(features)

    def probability(self, raw: torch.Tensor) -> torch.Tensor:
        """Map head output to a [0, 1] measure (LoF score or GoF probability)."""
        if self.task == "lof":
            return raw
        return torch.sigmoid(raw)
