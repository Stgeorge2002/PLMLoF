"""ESM2 encoder with optional LoRA adapters for protein sequence embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType


class ESM2Encoder(nn.Module):
    """Wrapper around ESM2 for extracting protein embeddings.

    Supports frozen inference and LoRA fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        freeze: bool = True,
        lora_config: dict | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: EsmModel = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if lora_config is not None:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_config.get("rank", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["query", "value"]),
            )
            self.model = get_peft_model(self.model, peft_config)

        self._lora_enabled = lora_config is not None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def enable_lora(self) -> None:
        """Enable gradient computation for LoRA parameters."""
        if self._lora_enabled:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    def disable_lora(self) -> None:
        """Freeze LoRA parameters."""
        if self._lora_enabled:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False

    def tokenize(
        self,
        sequences: list[str],
        max_length: int = 1024,
    ) -> dict[str, torch.Tensor]:
        """Tokenize protein sequences for ESM2.

        Args:
            sequences: List of amino acid sequences.
            max_length: Maximum sequence length (will truncate).

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through ESM2.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].

        Returns:
            Dict with:
                - 'per_residue': Per-residue embeddings [batch, seq_len, hidden_size]
                - 'pooled': Mean-pooled embeddings [batch, hidden_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # [B, L, D]

        # Mean pooling over non-padding positions
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        sum_hidden = (last_hidden * mask_expanded).sum(dim=1)  # [B, D]
        count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = sum_hidden / count  # [B, D]

        return {
            "per_residue": last_hidden,
            "pooled": pooled,
        }
