"""Dynamic padding collator for PLMLoF datasets."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer


class PLMLoFCollator:
    """Collates PLMLoF dataset samples into batches.

    Tokenizes protein sequences using the ESM2 tokenizer and pads dynamically.
    """

    def __init__(
        self,
        tokenizer_name: str = "facebook/esm2_t33_650M_UR50D",
        max_length: int = 1024,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a list of dataset samples into a batch.

        Args:
            batch: List of dicts from PLMLoFDataset.__getitem__.

        Returns:
            Dict with:
                - ref_input_ids [B, L_ref]
                - ref_attention_mask [B, L_ref]
                - var_input_ids [B, L_var]
                - var_attention_mask [B, L_var]
                - nucleotide_features [B, 12]
                - labels [B]
        """
        ref_seqs = [sample["ref_protein"] for sample in batch]
        var_seqs = [sample["var_protein"] for sample in batch]

        ref_encoded = self.tokenizer(
            ref_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        var_encoded = self.tokenizer(
            var_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        nuc_features = torch.stack([s["nucleotide_features"] for s in batch])
        labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)

        return {
            "ref_input_ids": ref_encoded["input_ids"],
            "ref_attention_mask": ref_encoded["attention_mask"],
            "var_input_ids": var_encoded["input_ids"],
            "var_attention_mask": var_encoded["attention_mask"],
            "nucleotide_features": nuc_features,
            "labels": labels,
        }
