"""Main prediction interface for PLMLoF inference."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plmlof import LABEL_MAP
from plmlof.models.plmlof_model import PLMLoFModel
from plmlof.models.esm2_encoder import ESM2Encoder
from plmlof.data.features import extract_nucleotide_features
from plmlof.inference.reference_cache import ReferenceCache
from plmlof.inference.vcf_handler import (
    VariantRecord,
    load_reference_dna,
    load_reference_proteins,
    parse_vcf_variants,
    parse_fasta_pairs,
)
from plmlof.inference.attribution import generate_attribution, AttributionResult

logger = logging.getLogger(__name__)


class PLMLoFPredictor:
    """High-level interface for PLMLoF variant effect prediction.

    Usage:
        predictor = PLMLoFPredictor("outputs/best_model/", device="cuda")
        predictor.load_reference("reference.fasta")
        results = predictor.predict_fasta("variants.fasta")
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: PLMLoFModel | None = None,
        device: str = "cpu",
        cache_dir: str = "outputs/ref_cache/",
        batch_size: int = 32,
        max_seq_length: int = 1024,
    ):
        """
        Args:
            model_path: Path to saved model checkpoint (.pt file or directory).
            model: Pre-built model instance (alternative to model_path).
            device: Device for inference.
            cache_dir: Directory for reference embedding cache.
            batch_size: Batch size for inference.
            max_seq_length: Maximum protein sequence length.
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")

        self.model.eval()
        self.cache = ReferenceCache(cache_dir)
        self._ref_dna: dict[str, str] = {}
        self._ref_proteins: dict[str, str] = {}

    def _load_model(self, path: str | Path) -> PLMLoFModel:
        """Load model from checkpoint."""
        path = Path(path)
        if path.is_dir():
            candidates = list(path.glob("model_best.pt")) + list(path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No .pt files found in {path}")
            path = candidates[0]

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Reconstruct model from saved config if available
        model_cfg = checkpoint.get("model_config", {})
        esm2_name = model_cfg.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D")
        lora_config = model_cfg.get("lora_config", None)

        model = PLMLoFModel(
            esm2_model_name=esm2_name,
            freeze_esm2=True,
            lora_config=lora_config,
            classifier_hidden_dims=model_cfg.get("classifier_hidden_dims", [256, 64]),
            classifier_dropout=model_cfg.get("classifier_dropout", 0.3),
            pool_strategy=model_cfg.get("pool_strategy", "mean_max"),
        )

        if checkpoint.get("cached_training"):
            # Cached checkpoint stores component state dicts separately
            model.comparison.load_state_dict(checkpoint["comparison_state_dict"])
            model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
            if "feature_norm_state_dict" in checkpoint:
                model.feature_norm.load_state_dict(checkpoint["feature_norm_state_dict"])
            logger.info("Assembled PLMLoFModel from cached-training checkpoint")
        else:
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)

        logger.info(f"Loaded model with ESM2: {esm2_name}")
        return model.to(self.device)

    def load_reference(self, fasta_path: str | Path) -> None:
        """Load and cache reference sequences for a strain.

        Args:
            fasta_path: Path to reference FASTA (gene CDS sequences).
        """
        logger.info(f"Loading reference from {fasta_path}")
        self._ref_dna = load_reference_dna(fasta_path)
        self._ref_proteins = load_reference_proteins(fasta_path)

        # Pre-compute and cache ESM2 embeddings for all reference genes
        self.cache.precompute(
            encoder=self.model.encoder,
            sequences=self._ref_proteins,
            batch_size=self.batch_size,
            max_length=self.max_seq_length,
        )
        logger.info(f"Loaded and cached {len(self._ref_proteins)} reference gene embeddings")

    def predict_records(
        self,
        records: list[VariantRecord],
        compute_attribution: bool = True,
    ) -> list[dict]:
        """Run predictions on a list of VariantRecords.

        Args:
            records: List of VariantRecord instances.
            compute_attribution: Whether to generate attribution explanations.

        Returns:
            List of result dicts with keys:
                gene, prediction, confidence, probabilities, attribution
        """
        results = []
        tokenizer = self.model.encoder.tokenizer

        for i in range(0, len(records), self.batch_size):
            batch_records = records[i : i + self.batch_size]

            ref_seqs = [r.ref_protein[:self.max_seq_length] for r in batch_records]
            var_seqs = [r.var_protein[:self.max_seq_length] for r in batch_records]

            # Tokenize
            ref_encoded = tokenizer(
                ref_seqs, padding=True, truncation=True,
                max_length=self.max_seq_length, return_tensors="pt",
            )
            var_encoded = tokenizer(
                var_seqs, padding=True, truncation=True,
                max_length=self.max_seq_length, return_tensors="pt",
            )

            # Nucleotide features
            nuc_features = torch.stack([
                extract_nucleotide_features(
                    r.ref_dna, r.var_dna, r.ref_protein, r.var_protein
                )
                for r in batch_records
            ])

            # Move to device
            ref_ids = ref_encoded["input_ids"].to(self.device)
            ref_mask = ref_encoded["attention_mask"].to(self.device)
            var_ids = var_encoded["input_ids"].to(self.device)
            var_mask = var_encoded["attention_mask"].to(self.device)
            nuc_features = nuc_features.to(self.device)

            # Predict
            output = self.model.predict(
                ref_ids, ref_mask, var_ids, var_mask, nuc_features
            )

            # Process results
            for j, record in enumerate(batch_records):
                pred_id = output["predictions"][j].item()
                probs = output["probabilities"][j].cpu().tolist()
                confidence = max(probs)
                prediction = LABEL_MAP[pred_id]

                result = {
                    "gene": record.gene,
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": {
                        LABEL_MAP[k]: probs[k] for k in range(3)
                    },
                    "variant_details": record.variant_details,
                }

                if compute_attribution:
                    attr = generate_attribution(
                        gene=record.gene,
                        prediction=prediction,
                        confidence=confidence,
                        ref_protein=record.ref_protein,
                        var_protein=record.var_protein,
                        ref_dna=record.ref_dna,
                        var_dna=record.var_dna,
                    )
                    result["attribution"] = {
                        "summary": attr.summary,
                        "top_positions": attr.top_positions,
                    }

                results.append(result)

        return results

    def predict_vcf(
        self,
        vcf_path: str | Path,
        gene_annotations: dict | None = None,
        compute_attribution: bool = True,
    ) -> list[dict]:
        """Predict variant effects from a VCF file.

        Requires load_reference() to have been called first.

        Args:
            vcf_path: Path to VCF file.
            gene_annotations: Optional gene coordinate mapping.
            compute_attribution: Whether to generate attribution explanations.

        Returns:
            List of prediction result dicts.
        """
        if not self._ref_dna:
            raise RuntimeError("Call load_reference() before predict_vcf()")

        records = parse_vcf_variants(vcf_path, self._ref_dna, gene_annotations)
        return self.predict_records(records, compute_attribution=compute_attribution)

    def predict_fasta(
        self,
        variant_fasta: str | Path,
        reference_fasta: str | Path | None = None,
        compute_attribution: bool = True,
    ) -> list[dict]:
        """Predict variant effects from paired FASTA files.

        Args:
            variant_fasta: Path to variant sequence FASTA.
            reference_fasta: Path to reference FASTA. If None, uses loaded reference.
            compute_attribution: Whether to generate attribution explanations.

        Returns:
            List of prediction result dicts.
        """
        if reference_fasta is None:
            if not self._ref_dna:
                raise RuntimeError("Provide reference_fasta or call load_reference() first")
            # Create a temp reference fasta path — just use already-loaded data
            # Parse variant fasta against loaded reference
            from plmlof.data.preprocessing import load_fasta
            var_seqs = load_fasta(variant_fasta)
            records = []
            for gene_id, var_seq in var_seqs.items():
                if gene_id not in self._ref_dna:
                    continue
                ref_dna = self._ref_dna[gene_id]
                ref_protein = self._ref_proteins.get(gene_id, "")
                from plmlof.utils.sequence_utils import translate_dna
                dna_chars = set("ATGCN")
                if all(c in dna_chars for c in var_seq[:50]):
                    var_protein = translate_dna(var_seq, to_stop=True)
                    var_dna = var_seq
                else:
                    var_protein = var_seq.replace("*", "")
                    var_dna = ""
                records.append(VariantRecord(
                    gene=gene_id,
                    ref_protein=ref_protein,
                    var_protein=var_protein,
                    ref_dna=ref_dna,
                    var_dna=var_dna,
                ))
            return self.predict_records(records, compute_attribution=compute_attribution)
        else:
            records = parse_fasta_pairs(reference_fasta, variant_fasta)
            return self.predict_records(records, compute_attribution=compute_attribution)
