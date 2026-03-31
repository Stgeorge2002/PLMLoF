"""Reference embedding cache for fast inference."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

from plmlof.models.esm2_encoder import ESM2Encoder

logger = logging.getLogger(__name__)


class ReferenceCache:
    """Caches ESM2 embeddings of reference protein sequences to disk.

    When running inference on many variants against the same reference strain,
    the reference embeddings only need to be computed once.
    """

    def __init__(self, cache_dir: str | Path = "outputs/ref_cache/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _hash_sequence(self, sequence: str) -> str:
        """Generate a deterministic hash for a protein sequence."""
        return hashlib.sha256(sequence.encode()).hexdigest()[:16]

    def _cache_path(self, seq_hash: str) -> Path:
        return self.cache_dir / f"{seq_hash}.pt"

    def has(self, sequence: str) -> bool:
        """Check if a sequence embedding is cached."""
        h = self._hash_sequence(sequence)
        return h in self._memory_cache or self._cache_path(h).exists()

    def get(self, sequence: str, device: torch.device | str = "cpu") -> dict[str, torch.Tensor] | None:
        """Retrieve cached embedding for a sequence.

        Returns:
            Dict with 'per_residue' and 'pooled' tensors, or None if not cached.
        """
        h = self._hash_sequence(sequence)

        # Check memory cache first
        if h in self._memory_cache:
            emb = self._memory_cache[h]
            return {k: v.to(device) for k, v in emb.items()}

        # Check disk cache
        path = self._cache_path(h)
        if path.exists():
            emb = torch.load(path, map_location=device, weights_only=True)
            self._memory_cache[h] = emb
            return emb

        return None

    def put(self, sequence: str, embeddings: dict[str, torch.Tensor]) -> None:
        """Cache embeddings for a sequence (memory + disk)."""
        h = self._hash_sequence(sequence)

        # Store CPU copies
        cpu_emb = {k: v.detach().cpu() for k, v in embeddings.items()}
        self._memory_cache[h] = cpu_emb

        # Persist to disk
        path = self._cache_path(h)
        torch.save(cpu_emb, path)

    @torch.no_grad()
    def precompute(
        self,
        encoder: ESM2Encoder,
        sequences: dict[str, str],
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        """Pre-compute and cache embeddings for a set of reference sequences.

        Args:
            encoder: ESM2Encoder instance.
            sequences: Dict of gene_name → protein_sequence.
            batch_size: Batch size for encoding.
            max_length: Maximum sequence length.
        """
        encoder.eval()
        device = encoder.device

        # Filter to only uncached sequences
        to_encode = {
            name: seq for name, seq in sequences.items()
            if not self.has(seq)
        }

        if not to_encode:
            logger.info("All reference sequences already cached.")
            return

        logger.info(f"Pre-computing embeddings for {len(to_encode)} reference sequences...")

        names = list(to_encode.keys())
        seqs = list(to_encode.values())

        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            batch_names = names[i : i + batch_size]

            encoded = encoder.tokenize(batch_seqs, max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

            # Cache each sequence individually
            for j, (name, seq) in enumerate(zip(batch_names, batch_seqs)):
                mask_len = encoded["attention_mask"][j].sum().item()
                emb = {
                    "per_residue": outputs["per_residue"][j, :mask_len].detach().cpu(),
                    "pooled": outputs["pooled"][j].detach().cpu(),
                }
                self.put(seq, emb)

        logger.info("Reference embedding caching complete.")

    def clear(self) -> None:
        """Clear all cached embeddings (memory and disk)."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.pt"):
            f.unlink()
        logger.info("Cache cleared.")
