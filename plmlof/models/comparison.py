"""Comparison module for reference vs variant protein embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledCrossAttention(nn.Module):
    """Lightweight cross-attention between pooled ref and var embeddings.

    Takes 4 pooled vectors (ref_mean, ref_max, var_mean, var_max) as a
    sequence of 4 "tokens" and applies multi-head self-attention to learn
    interactions between pooling strategies and ref/var sides.

    Compatible with the cached training path (no per-residue tokens needed).
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_tokens: [B, 4, D] — ref_mean, ref_max, var_mean, var_max

        Returns:
            Attended tokens [B, 4, D] (residual connection).
        """
        attn_out, _ = self.attn(pooled_tokens, pooled_tokens, pooled_tokens)
        return self.norm(pooled_tokens + attn_out)


class ComparisonModule(nn.Module):
    """Compares reference and variant protein embeddings using multiple strategies.

    Produces a fixed-size comparison vector regardless of input sequence lengths
    using element-wise operations and pooling.
    """

    def __init__(self, hidden_size: int, pool_strategy: str = "mean_max"):
        """
        Args:
            hidden_size: Dimensionality of ESM2 embeddings.
            pool_strategy: Pooling strategy - 'mean_max' or 'mean'.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.pool_strategy = pool_strategy

        # Output size: 4 * hidden_size
        #   - pooled diff [D]
        #   - pooled product [D]
        #   - pooled ref [D]
        #   - pooled var [D]
        self.output_size = 4 * hidden_size
        if pool_strategy == "mean_max":
            # mean + max for each → doubles the output
            self.output_size = 4 * hidden_size * 2

        # Normalize before projection (critical: diff/prod/ref/var have very different scales)
        raw_size = self.output_size
        self._pre_norm = nn.LayerNorm(raw_size)

        # Two-stage projection with activation for better gradient flow
        proj_intermediate = 2 * hidden_size
        self._proj = nn.Sequential(
            nn.Linear(raw_size, proj_intermediate),
            nn.GELU(),
            nn.Linear(proj_intermediate, 4 * hidden_size),
        )
        self.output_size = 4 * hidden_size

        # Initialize projection with small gains for stable early training
        self._init_proj_weights()

    def _init_proj_weights(self):
        """Initialize projection weights for stable training."""
        for m in self._proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def project(self, raw_comparison: torch.Tensor) -> torch.Tensor:
        """Normalize and project raw comparison features.

        Use this method in both cached and non-cached training paths
        to ensure consistent processing.
        """
        return self._proj(self._pre_norm(raw_comparison))

    def _pool(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool variable-length embeddings to fixed size.

        Args:
            embeddings: [batch, seq_len, hidden_size]
            mask: [batch, seq_len]

        Returns:
            Pooled vector [batch, hidden_size] or [batch, 2*hidden_size] for mean_max.
        """
        mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]

        # Mean pool
        sum_emb = (embeddings * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        mean_pool = sum_emb / count  # [B, D]

        if self.pool_strategy == "mean_max":
            # Max pool (set padding to -inf)
            embeddings_masked = embeddings.masked_fill(
                ~mask.unsqueeze(-1).bool(), float("-inf")
            )
            max_pool = embeddings_masked.max(dim=1).values  # [B, D]
            # Replace -inf with 0 for fully-padded sequences
            max_pool = max_pool.masked_fill(max_pool == float("-inf"), 0.0)
            return torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2D]

        return mean_pool  # [B, D]

    def forward(
        self,
        ref_embeddings: dict[str, torch.Tensor],
        var_embeddings: dict[str, torch.Tensor],
        ref_mask: torch.Tensor,
        var_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compare reference and variant embeddings.

        Args:
            ref_embeddings: Dict with 'per_residue' [B, L_ref, D] and 'pooled' [B, D].
            var_embeddings: Dict with 'per_residue' [B, L_var, D] and 'pooled' [B, D].
            ref_mask: Attention mask for reference [B, L_ref].
            var_mask: Attention mask for variant [B, L_var].

        Returns:
            Comparison vector [B, 4*D].
        """
        # Per-residue pooling for richer features
        ref_per_res = ref_embeddings["per_residue"]  # [B, L_ref, D]
        var_per_res = var_embeddings["per_residue"]  # [B, L_var, D]

        # Pool per-residue embeddings
        ref_pool = self._pool(ref_per_res, ref_mask)
        var_pool = self._pool(var_per_res, var_mask)

        # Compute per-residue diff and product after pooling
        diff_pool = ref_pool - var_pool
        prod_pool = ref_pool * var_pool

        # Concatenate all comparison features
        comparison = torch.cat(
            [diff_pool, prod_pool, ref_pool, var_pool], dim=-1
        )  # [B, output_size_raw]

        # Normalize and project to standardized size
        return self.project(comparison)  # [B, 4*D]
