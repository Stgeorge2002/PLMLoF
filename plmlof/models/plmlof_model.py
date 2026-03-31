"""Full PLMLoF model combining ESM2 encoder, comparison module, and classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from plmlof.models.esm2_encoder import ESM2Encoder
from plmlof.models.comparison import ComparisonModule
from plmlof.models.classifier import ClassifierHead
from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES


class PLMLoFModel(nn.Module):
    """PLM-based LoF/GoF variant classifier.

    Architecture:
        1. Shared ESM2 encoder embeds both reference and variant protein sequences
        2. Comparison module computes diff/product/pool features between embeddings
        3. Nucleotide features (engineered, rule-based) are concatenated
        4. Classifier head predicts LoF / WT / GoF
    """

    def __init__(
        self,
        esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
        freeze_esm2: bool = True,
        lora_config: dict | None = None,
        pool_strategy: str = "mean_max",
        classifier_hidden_dims: list[int] | None = None,
        classifier_dropout: float = 0.3,
        num_classes: int = 3,
        num_nuc_features: int = NUM_NUCLEOTIDE_FEATURES,
    ):
        super().__init__()

        # ESM2 encoder (shared between ref and var)
        self.encoder = ESM2Encoder(
            model_name=esm2_model_name,
            freeze=freeze_esm2,
            lora_config=lora_config,
        )
        hidden_size = self.encoder.hidden_size

        # Comparison module
        self.comparison = ComparisonModule(
            hidden_size=hidden_size,
            pool_strategy=pool_strategy,
        )

        # Classifier head
        classifier_input_size = self.comparison.output_size + num_nuc_features
        self.classifier = ClassifierHead(
            input_size=classifier_input_size,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

        self.num_classes = num_classes
        self.num_nuc_features = num_nuc_features

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def enable_lora_training(self) -> None:
        """Switch to Stage 2: enable LoRA gradient updates."""
        self.encoder.enable_lora()

    def disable_lora_training(self) -> None:
        """Switch back to Stage 1: freeze LoRA parameters."""
        self.encoder.disable_lora()

    def forward(
        self,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: torch.Tensor,
        var_input_ids: torch.Tensor,
        var_attention_mask: torch.Tensor,
        nucleotide_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ref_input_ids: Reference protein token IDs [B, L_ref].
            ref_attention_mask: Reference attention mask [B, L_ref].
            var_input_ids: Variant protein token IDs [B, L_var].
            var_attention_mask: Variant attention mask [B, L_var].
            nucleotide_features: Engineered nucleotide features [B, num_nuc_features].

        Returns:
            Logits [B, num_classes].
        """
        # Encode reference and variant with shared ESM2
        ref_emb = self.encoder(ref_input_ids, ref_attention_mask)
        var_emb = self.encoder(var_input_ids, var_attention_mask)

        # Compare embeddings
        comparison = self.comparison(
            ref_emb, var_emb, ref_attention_mask, var_attention_mask
        )

        # Concatenate comparison features with nucleotide features
        features = torch.cat([comparison, nucleotide_features], dim=-1)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict(
        self,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: torch.Tensor,
        var_input_ids: torch.Tensor,
        var_attention_mask: torch.Tensor,
        nucleotide_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass and return predictions with probabilities.

        Returns:
            Dict with 'logits' [B, 3], 'probabilities' [B, 3], 'predictions' [B].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                ref_input_ids, ref_attention_mask,
                var_input_ids, var_attention_mask,
                nucleotide_features,
            )
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        }
