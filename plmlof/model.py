"""Task nets: alignment-free LoF, pairwise MLoF, conservative GoF."""

from __future__ import annotations

import torch
import torch.nn as nn

from plmlof.constants import REGRESSION_TASKS, TASKS
from plmlof.data.features import LOF_LEAK_NUC_INDICES, NUM_NUCLEOTIDE_FEATURES
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


class TaskNet(nn.Module):
    """One head per task.

    ``lof`` is alignment-free: it scores a single protein (the isolate allele)
    from pooled ESM2. No ref, no pair features, no MSA.

    ``mlof`` / GoF compare ref vs var so wreck length cannot leak into those heads.
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
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.hidden_size = hidden_size
        self.alignment_free = task == "lof"
        self.comparison: ComparisonModule | None = None
        self.feature_norm: nn.LayerNorm | None = None
        self.seq_norm: nn.LayerNorm | None = None

        if self.alignment_free:
            self.seq_norm = nn.LayerNorm(hidden_size * 2)
            self.head = LofScoreHead(hidden_size * 2, hidden_dim=head_hidden, dropout=dropout)
            return

        self.comparison = ComparisonModule(
            hidden_size=hidden_size,
            pool_strategy=pool_strategy,
            use_cross_attention=use_cross_attention,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dropout=cross_attn_dropout,
        )
        self.feature_norm = nn.LayerNorm(num_nuc_features)
        input_size = self.comparison.output_size + num_nuc_features
        if task == "mlof":
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
        if self.alignment_free:
            return self.head(self.seq_norm(torch.cat([var_mean, var_max], dim=-1)))
        comparison = self.comparison.compare_pooled(ref_mean, ref_max, var_mean, var_max)
        nuc = nucleotide_features
        if self.task == "mlof":
            nuc = nucleotide_features.clone()
            nuc[..., list(LOF_LEAK_NUC_INDICES)] = 0
        nuc = self.feature_norm(nuc)
        return self.head(torch.cat([comparison, nuc], dim=-1))

    def probability(self, raw: torch.Tensor) -> torch.Tensor:
        if self.task in REGRESSION_TASKS:
            return raw
        return torch.sigmoid(raw)
