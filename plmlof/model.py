"""Task nets: alignment-free LoF, site-level MLoF, conservative GoF."""

from __future__ import annotations

import torch
import torch.nn as nn

from plmlof.constants import REGRESSION_TASKS, TASKS
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


class SiteCompare(nn.Module):
    """Compare ref vs var at the mutated residue, not the pooled protein.

    Input is a window ``[B, 3, D]`` of (left, centre, right) residue tokens
    for each side. Pooled sequence identity never enters, so the head cannot
    learn a gene prior from ``ref_mean``.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        # centre_ref, centre_var, delta_centre, delta_left, delta_right
        raw = 5 * hidden_size
        self.norm = nn.LayerNorm(raw)
        self.proj = nn.Sequential(
            nn.Linear(raw, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.output_size = hidden_size
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, site_ref: torch.Tensor, site_var: torch.Tensor) -> torch.Tensor:
        if site_ref.ndim != 3 or site_var.ndim != 3 or site_ref.size(1) != 3:
            raise ValueError(
                f"site tensors must be [B, 3, D], got ref={tuple(site_ref.shape)} var={tuple(site_var.shape)}"
            )
        ref_l, ref_c, ref_r = site_ref[:, 0], site_ref[:, 1], site_ref[:, 2]
        var_l, var_c, var_r = site_var[:, 0], site_var[:, 1], site_var[:, 2]
        raw = torch.cat(
            [ref_c, var_c, ref_c - var_c, ref_l - var_l, ref_r - var_r],
            dim=-1,
        )
        return self.proj(self.norm(raw))


class TaskNet(nn.Module):
    """One head per task.

    ``lof`` is alignment-free: it scores a single protein (the isolate allele)
    from pooled ESM2. No ref, no pair features, no MSA.

    ``mlof`` compares residue windows at the substitution. Pooled ref is not
    an input, so gene identity cannot dominate.

    GoF still uses pooled ComparisonModule (missense SNPs plus wreck negatives).
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
        self.uses_sites = task == "mlof"
        self.comparison: ComparisonModule | None = None
        self.site_compare: SiteCompare | None = None
        self.feature_norm: nn.LayerNorm | None = None
        self.seq_norm: nn.LayerNorm | None = None

        if self.alignment_free:
            self.seq_norm = nn.LayerNorm(hidden_size * 2)
            self.head = LofScoreHead(hidden_size * 2, hidden_dim=head_hidden, dropout=dropout)
            return

        if self.uses_sites:
            self.site_compare = SiteCompare(hidden_size, dropout=dropout)
            self.head = LofScoreHead(self.site_compare.output_size, hidden_dim=head_hidden, dropout=dropout)
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
        self.head = BinaryLogitHead(input_size, hidden_dim=head_hidden, dropout=dropout)

    def forward_from_cache(
        self,
        ref_mean: torch.Tensor,
        ref_max: torch.Tensor,
        var_mean: torch.Tensor,
        var_max: torch.Tensor,
        nucleotide_features: torch.Tensor,
        site_ref: torch.Tensor | None = None,
        site_var: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.alignment_free:
            return self.head(self.seq_norm(torch.cat([var_mean, var_max], dim=-1)))
        if self.uses_sites:
            if site_ref is None or site_var is None:
                raise ValueError("MLoF requires site_ref and site_var [B, 3, D]")
            return self.head(self.site_compare(site_ref, site_var))
        comparison = self.comparison.compare_pooled(ref_mean, ref_max, var_mean, var_max)
        nuc = nucleotide_features
        nuc = self.feature_norm(nuc)
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
    ) -> torch.Tensor:
        return self.forward_from_cache(
            ref_mean, ref_max, var_mean, var_max, nucleotide_features,
            site_ref=site_ref, site_var=site_var,
        )

    def probability(self, raw: torch.Tensor) -> torch.Tensor:
        if self.task in REGRESSION_TASKS:
            return raw
        return torch.sigmoid(raw)
