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

    def __init__(
        self,
        hidden_size: int,
        pool_strategy: str = "mean_max",
        use_cross_attention: bool = False,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: Dimensionality of ESM2 embeddings.
            pool_strategy: Pooling strategy - 'mean_max' or 'mean'.
            use_cross_attention: Apply PooledCrossAttention over the pooled
                ref/var tokens before comparing. Owned here (not by the
                trainer) so it's used consistently by the live ESM2 forward
                path, cached-embedding training, and inference.
            cross_attn_heads: Attention heads for the cross-attention module.
            cross_attn_dropout: Dropout for the cross-attention module.
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

        self.cross_attn = (
            PooledCrossAttention(
                hidden_size=hidden_size, num_heads=cross_attn_heads, dropout=cross_attn_dropout,
            )
            if use_cross_attention
            else None
        )

        # Normalize before projection (critical: diff/prod/ref/var have very different scales)
        raw_size = self.output_size
        self._pre_norm = nn.LayerNorm(raw_size)

        # Gated feature selection: learn which comparison features are important
        # Gate outputs [0, 1] weights for each feature dimension
        self._gate = nn.Sequential(
            nn.Linear(raw_size, raw_size // 4),
            nn.ReLU(),
            nn.Linear(raw_size // 4, raw_size),
            nn.Sigmoid(),
        )

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

        # Initialize the gate's final (sigmoid) layer bias to 3.0 so it starts
        # mostly open (sigmoid(3.0) ≈ 0.95); the hidden layer keeps a zero bias.
        gate_linears = [m for m in self._gate.modules() if isinstance(m, nn.Linear)]
        for m in gate_linears:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if gate_linears and gate_linears[-1].bias is not None:
            nn.init.constant_(gate_linears[-1].bias, 3.0)

    def project(self, raw_comparison: torch.Tensor) -> torch.Tensor:
        """Normalize, gate, and project raw comparison features.

        Use this method in both cached and non-cached training paths
        to ensure consistent processing.
        """
        # Normalize raw features
        normalized = self._pre_norm(raw_comparison)
        
        # Apply learned gating (element-wise feature selection)
        gates = self._gate(normalized)
        gated_features = normalized * gates
        
        # Project to output size
        return self._proj(gated_features)

    def _pool_mean_max(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pool variable-length embeddings to fixed-size mean (and max) vectors.

        Args:
            embeddings: [batch, seq_len, hidden_size]
            mask: [batch, seq_len]

        Returns:
            (mean_pool, max_pool) — max_pool is None when pool_strategy == "mean".
        """
        mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]

        sum_emb = (embeddings * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        mean_pool = sum_emb / count  # [B, D]

        if self.pool_strategy != "mean_max":
            return mean_pool, None

        embeddings_masked = embeddings.masked_fill(
            ~mask.unsqueeze(-1).bool(), float("-inf")
        )
        max_pool = embeddings_masked.max(dim=1).values  # [B, D]
        # Replace -inf with 0 for fully-padded sequences
        max_pool = max_pool.masked_fill(max_pool == float("-inf"), 0.0)
        return mean_pool, max_pool

    def compare_pooled(
        self,
        ref_mean: torch.Tensor,
        ref_max: torch.Tensor | None,
        var_mean: torch.Tensor,
        var_max: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compare already-pooled ref/var vectors.

        Shared by the live ESM2 forward path (see `forward`) and the
        cached-embeddings training/inference path, so cross-attention and
        the diff/product/projection logic only exist in one place.

        Args:
            ref_mean, var_mean: Mean-pooled vectors [B, D].
            ref_max, var_max: Max-pooled vectors [B, D], or None when
                pool_strategy == "mean".

        Returns:
            Comparison vector [B, 4*D].
        """
        if self.cross_attn is not None:
            if ref_max is not None:
                tokens = torch.stack([ref_mean, ref_max, var_mean, var_max], dim=1)
            else:
                tokens = torch.stack([ref_mean, var_mean], dim=1)
            tokens = self.cross_attn(tokens)
            if ref_max is not None:
                ref_mean, ref_max, var_mean, var_max = tokens.unbind(dim=1)
            else:
                ref_mean, var_mean = tokens.unbind(dim=1)

        ref_pool = torch.cat([ref_mean, ref_max], dim=-1) if ref_max is not None else ref_mean
        var_pool = torch.cat([var_mean, var_max], dim=-1) if var_max is not None else var_mean

        diff_pool = ref_pool - var_pool
        prod_pool = ref_pool * var_pool
        comparison = torch.cat(
            [diff_pool, prod_pool, ref_pool, var_pool], dim=-1
        )  # [B, output_size_raw]

        return self.project(comparison)  # [B, 4*D]

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
        ref_mean, ref_max = self._pool_mean_max(ref_embeddings["per_residue"], ref_mask)
        var_mean, var_max = self._pool_mean_max(var_embeddings["per_residue"], var_mask)
        return self.compare_pooled(ref_mean, ref_max, var_mean, var_max)
