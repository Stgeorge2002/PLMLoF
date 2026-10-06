"""Task nets: alignment-free LoF, site-level MLoF, conservative GoF."""

from __future__ import annotations

import torch
import torch.nn as nn

from plmlof.chem import ablate_chem_channels, pad_site_chem
from plmlof.constants import NUM_SITE_CHEM, REGRESSION_TASKS, SITE_WINDOW, TASKS
from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES
from plmlof.models.comparison import ComparisonModule


def _align_chem(site_chem: torch.Tensor, dim: int) -> torch.Tensor:
    """Match chemistry width to the head: pad old 5-dim caches, slice if wider."""
    if site_chem.ndim == 1:
        site_chem = site_chem.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if site_chem.ndim != 2:
        raise ValueError(f"site_chem must be [B, C], got {tuple(site_chem.shape)}")
    out = pad_site_chem(site_chem, dim)
    return out.squeeze(0) if squeeze else out


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
    """Compare ref vs var on a residue window, not the pooled protein.

    Input is ``[B, W, D]`` with W odd (centre at ``W // 2``). Features are
    centre tokens plus the flattened per-position delta. Pooled sequence
    identity never enters, so the head cannot learn a gene prior from
    ``ref_mean``.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1, window: int = SITE_WINDOW):
        super().__init__()
        if int(window) < 1 or int(window) % 2 == 0:
            raise ValueError(f"site window must be odd and >= 1, got {window}")
        self.hidden_size = hidden_size
        self.window = int(window)
        self.radius = self.window // 2
        raw = (self.window + 2) * hidden_size
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
        if site_ref.ndim != 3 or site_var.ndim != 3 or site_ref.size(1) != self.window:
            raise ValueError(
                f"site tensors must be [B, {self.window}, D], "
                f"got ref={tuple(site_ref.shape)} var={tuple(site_var.shape)}"
            )
        centre = self.radius
        ref_c = site_ref[:, centre]
        var_c = site_var[:, centre]
        delta = (site_ref - site_var).reshape(site_ref.size(0), -1)
        raw = torch.cat([ref_c, var_c, delta], dim=-1)
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
        site_window: int = SITE_WINDOW,
        chem_dim: int = NUM_SITE_CHEM,
    ):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.hidden_size = hidden_size
        self.alignment_free = task == "lof"
        self.uses_sites = task == "mlof"
        self.site_window = int(site_window)
        self.chem_dim = int(chem_dim) if self.uses_sites else 0
        self.comparison: ComparisonModule | None = None
        self.site_compare: SiteCompare | None = None
        self.feature_norm: nn.LayerNorm | None = None
        self.seq_norm: nn.LayerNorm | None = None
        self.chem_norm: nn.LayerNorm | None = None
        self.ablate_logodds = False
        self.ablate_domain = False

        if self.alignment_free:
            self.seq_norm = nn.LayerNorm(hidden_size * 2)
            self.head = LofScoreHead(hidden_size * 2, hidden_dim=head_hidden, dropout=dropout)
            return

        if self.uses_sites:
            self.site_compare = SiteCompare(hidden_size, dropout=dropout, window=self.site_window)
            head_in = self.site_compare.output_size + self.chem_dim
            if self.chem_dim:
                self.chem_norm = nn.LayerNorm(self.chem_dim)
            self.head = LofScoreHead(head_in, hidden_dim=head_hidden, dropout=dropout)
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

    def _prep_chem(self, site_chem: torch.Tensor | None, batch: int, like: torch.Tensor) -> torch.Tensor | None:
        if not self.chem_dim:
            return None
        if site_chem is None:
            return like.new_zeros(batch, self.chem_dim)
        chem = _align_chem(site_chem.float(), self.chem_dim)
        return ablate_chem_channels(
            chem, logodds=self.ablate_logodds, domain=self.ablate_domain,
        )

    def _score_site(
        self,
        site_ref: torch.Tensor,
        site_var: torch.Tensor,
        site_chem: torch.Tensor | None,
    ) -> torch.Tensor:
        feat = self.site_compare(site_ref, site_var)
        chem = self._prep_chem(site_chem, feat.size(0), feat)
        if chem is not None:
            feat = torch.cat([feat, self.chem_norm(chem)], dim=-1)
        return self.head(feat)

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
        if self.alignment_free:
            return self.head(self.seq_norm(torch.cat([var_mean, var_max], dim=-1)))
        if self.uses_sites:
            if site_ref is None or site_var is None:
                raise ValueError(
                    f"MLoF requires site_ref and site_var [B, {self.site_window}, D]"
                )
            score = self._score_site(site_ref, site_var, site_chem)
            if site_ref2 is None or site_var2 is None:
                return score
            score2 = self._score_site(site_ref2, site_var2, site_chem2)
            both = torch.maximum(score, score2)
            if n_sites is None:
                live = site_ref2.abs().reshape(site_ref2.size(0), -1).sum(dim=-1) > 0
                return torch.where(live, both, score)
            return torch.where(n_sites.reshape(-1) >= 2, both, score)
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
        site_chem: torch.Tensor | None = None,
        site_ref2: torch.Tensor | None = None,
        site_var2: torch.Tensor | None = None,
        site_chem2: torch.Tensor | None = None,
        n_sites: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_from_cache(
            ref_mean, ref_max, var_mean, var_max, nucleotide_features,
            site_ref=site_ref, site_var=site_var, site_chem=site_chem,
            site_ref2=site_ref2, site_var2=site_var2, site_chem2=site_chem2,
            n_sites=n_sites,
        )

    def probability(self, raw: torch.Tensor) -> torch.Tensor:
        if self.task in REGRESSION_TASKS:
            return raw
        return torch.sigmoid(raw)
