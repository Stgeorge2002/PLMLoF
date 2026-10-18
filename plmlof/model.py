"""Task net: comparative LoF head only."""

from __future__ import annotations

import torch
import torch.nn as nn

from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES
from plmlof.models.comparison import ComparisonModule


class LofScoreHead(nn.Module):
    """Scalar score in [0, 1]."""

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


class TaskNet(nn.Module):
    """LoF comparative head: pooled ref vs var (ComparisonModule + LoF score)."""

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
        site_window: int = 5,  # ignored, kept for old checkpoint compatibility
        chem_dim: int = 0,  # ignored, kept for old checkpoint compatibility
    ):
        super().__init__()
        if task != "lof":
            raise ValueError(f"TaskNet only supports task='lof', got {task}")
        self.task = task
        self.hidden_size = hidden_size
        self.alignment_free = False
        self.comparison = ComparisonModule(
            hidden_size=hidden_size,
            pool_strategy=pool_strategy,
            use_cross_attention=use_cross_attention,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dropout=cross_attn_dropout,
        )
        self.feature_norm = nn.LayerNorm(num_nuc_features)
        input_size = self.comparison.output_size + num_nuc_features
        self.head = LofScoreHead(input_size, hidden_dim=head_hidden, dropout=dropout)

    def forward_from_cache(
        self,
        ref_mean: torch.Tensor,
        ref_max: torch.Tensor,
        var_mean: torch.Tensor,
        var_max: torch.Tensor,
        nucleotide_features: torch.Tensor,
        site_ref: torch.Tensor | None = None,
        site_var: torch.Tensor | None = None,
        site_chem: torch.Tensor | None = None,
        site_ref2: torch.Tensor | None = None,
        site_var2: torch.Tensor | None = None,
        site_chem2: torch.Tensor | None = None,
        n_sites: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass. site_* kwargs ignored (kept for cache compatibility)."""
        comparison = self.comparison.compare_pooled(ref_mean, ref_max, var_mean, var_max)
        nuc = self.feature_norm(nucleotide_features)
        return self.head(torch.cat([comparison, nuc], dim=-1))

    def forward_from_pooled(
        self,
        ref_mean: torch.Tensor,
        ref_max: torch.Tensor,
        var_mean: torch.Tensor,
        var_max: torch.Tensor,
        nucleotide_features: torch.Tensor,
        site_ref: torch.Tensor | None = None,
        site_var: torch.Tensor | None = None,
        site_chem: torch.Tensor | None = None,
        site_ref2: torch.Tensor | None = None,
        site_var2: torch.Tensor | None = None,
        site_chem2: torch.Tensor | None = None,
        n_sites: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for forward_from_cache. site_* kwargs ignored."""
        return self.forward_from_cache(
            ref_mean, ref_max, var_mean, var_max, nucleotide_features,
        )

    def probability(self, raw: torch.Tensor) -> torch.Tensor:
        """LoF is already a probability (sigmoid in the head)."""
        return raw
