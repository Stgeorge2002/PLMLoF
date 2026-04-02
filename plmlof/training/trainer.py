"""Training loop for PLMLoF with two-stage training support."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from plmlof.models.plmlof_model import PLMLoFModel
from plmlof.training.metrics import compute_metrics


logger = logging.getLogger(__name__)


class PLMLoFTrainer:
    """Two-stage trainer for PLMLoF model.

    Stage 1: Train classification head only (ESM2 frozen, LoRA disabled).
    Stage 2: Fine-tune with LoRA adapters enabled.
    Supports mixed precision (fp16/bf16) for GPU acceleration.
    """

    def __init__(
        self,
        model: PLMLoFModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str | torch.device = "cpu",
        output_dir: str = "outputs/",
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.0,
        mixed_precision: str = "no",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss with optional class weights
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

        self.best_metric = 0.0
        self.best_epoch = -1

        # Mixed precision setup
        self.mixed_precision = mixed_precision
        self.use_amp = mixed_precision in ("fp16", "bf16") and self.device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            self.scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision == "fp16"))
            logger.info(f"Mixed precision enabled: {mixed_precision}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)

    def _build_optimizer(self, lr: float, weight_decay: float) -> AdamW:
        """Build optimizer for trainable parameters only."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    def _train_epoch(self, optimizer: AdamW, grad_accum_steps: int = 1) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(
                    ref_input_ids=batch["ref_input_ids"],
                    ref_attention_mask=batch["ref_attention_mask"],
                    var_input_ids=batch["var_input_ids"],
                    var_attention_mask=batch["var_attention_mask"],
                    nucleotide_features=batch["nucleotide_features"],
                )
                loss = self.criterion(logits, batch["labels"])
                loss = loss / grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(
            np.array(all_preds), np.array(all_labels)
        )
        metrics["loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def _eval_epoch(self) -> dict[str, float]:
        """Run one evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits = self.model(
                ref_input_ids=batch["ref_input_ids"],
                ref_attention_mask=batch["ref_attention_mask"],
                var_input_ids=batch["var_input_ids"],
                var_attention_mask=batch["var_attention_mask"],
                nucleotide_features=batch["nucleotide_features"],
            )

            loss = self.criterion(logits, batch["labels"])
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(probs)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        metrics = compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs),
        )
        metrics["loss"] = avg_loss
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str = "best") -> None:
        """Save model checkpoint."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"model_{tag}.pt"
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        # Include model config if available (for reconstruction during inference)
        if hasattr(self, "model_config"):
            save_dict["model_config"] = self.model_config
        torch.save(save_dict, path)
        logger.info(f"Saved checkpoint to {path}")

    def train(
        self,
        stage: int = 1,
        max_epochs: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 1,
        patience: int = 5,
    ) -> dict[str, float]:
        """Run training for a given stage.

        Args:
            stage: 1 = head only, 2 = with LoRA.
            max_epochs: Maximum number of epochs.
            learning_rate: Learning rate.
            weight_decay: Weight decay for AdamW.
            grad_accum_steps: Gradient accumulation steps.
            patience: Early stopping patience.

        Returns:
            Best validation metrics.
        """
        logger.info(f"Starting Stage {stage} training (lr={learning_rate}, epochs={max_epochs})")

        if stage == 2:
            self.model.enable_lora_training()
        else:
            self.model.disable_lora_training()

        optimizer = self._build_optimizer(learning_rate, weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        epochs_without_improvement = 0
        best_val_metrics = {}

        for epoch in range(1, max_epochs + 1):
            logger.info(f"Epoch {epoch}/{max_epochs}")

            train_metrics = self._train_epoch(optimizer, grad_accum_steps)
            val_metrics = self._eval_epoch()
            scheduler.step()

            logger.info(
                f"  Train loss={train_metrics['loss']:.4f}, "
                f"macro_f1={train_metrics['macro_f1']:.4f}"
            )
            logger.info(
                f"  Val   loss={val_metrics['loss']:.4f}, "
                f"macro_f1={val_metrics['macro_f1']:.4f}"
            )

            # Check for improvement
            if val_metrics["macro_f1"] > self.best_metric:
                self.best_metric = val_metrics["macro_f1"]
                self.best_epoch = epoch
                best_val_metrics = val_metrics
                self._save_checkpoint(epoch, val_metrics, tag="best")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"Best macro_f1={self.best_metric:.4f} at epoch {self.best_epoch}")
        return best_val_metrics


class CachedTrainer:
    """Fast trainer using pre-computed ESM2 embeddings.

    Trains only the ComparisonModule + ClassifierHead on cached pooled
    embeddings — no ESM2 forward passes needed.
    """

    def __init__(
        self,
        comparison: nn.Module,
        classifier: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str | torch.device = "cpu",
        output_dir: str = "outputs/",
        label_smoothing: float = 0.0,
        mixed_precision: str = "no",
        # Model config for saving a reconstructable checkpoint
        esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
        pool_strategy: str = "mean_max",
        classifier_hidden_dims: list[int] | None = None,
        classifier_dropout: float = 0.3,
        lora_config: dict | None = None,
    ):
        self.device = torch.device(device)
        self.comparison = comparison.to(self.device)
        self.classifier = classifier.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.best_metric = 0.0
        self.best_epoch = -1

        self.mixed_precision = mixed_precision
        self.use_amp = mixed_precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision == "fp16" and self.device.type == "cuda"))

        self.model_config = {
            "esm2_model_name": esm2_model_name,
            "classifier_hidden_dims": classifier_hidden_dims,
            "classifier_dropout": classifier_dropout,
            "pool_strategy": pool_strategy,
            "lora_config": lora_config,
        }

    def _forward(self, batch: dict) -> torch.Tensor:
        """Reconstruct comparison features from cached pooled embeddings and classify."""
        # Reconstruct pooled ref/var as [mean, max] concatenation
        ref_pool = torch.cat([batch["ref_mean"], batch["ref_max"]], dim=-1)
        var_pool = torch.cat([batch["var_mean"], batch["var_max"]], dim=-1)

        diff_pool = ref_pool - var_pool
        prod_pool = ref_pool * var_pool
        comparison = torch.cat([diff_pool, prod_pool, ref_pool, var_pool], dim=-1)
        comparison = self.comparison._proj(comparison)

        features = torch.cat([comparison, batch["nucleotide_features"]], dim=-1)
        return self.classifier(features)

    def _collate(self, batch: list[dict]) -> dict:
        return {
            "ref_mean": torch.stack([b["ref_mean"] for b in batch]).to(self.device),
            "ref_max": torch.stack([b["ref_max"] for b in batch]).to(self.device),
            "var_mean": torch.stack([b["var_mean"] for b in batch]).to(self.device),
            "var_max": torch.stack([b["var_max"] for b in batch]).to(self.device),
            "nucleotide_features": torch.stack([b["nucleotide_features"] for b in batch]).to(self.device),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long, device=self.device),
        }

    def _train_epoch(self, optimizer, grad_accum_steps: int = 1) -> dict[str, float]:
        self.comparison.train()
        self.classifier.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self._forward(batch)
                loss = self.criterion(logits, batch["labels"]) / grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics["loss"] = total_loss / max(len(self.train_loader), 1)
        return metrics

    @torch.no_grad()
    def _eval_epoch(self) -> dict[str, float]:
        self.comparison.eval()
        self.classifier.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits = self._forward(batch)
            loss = self.criterion(logits, batch["labels"])
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels), np.array(all_probs))
        metrics["loss"] = total_loss / max(len(self.val_loader), 1)
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save checkpoint compatible with the full PLMLoFModel for inference."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "model_best.pt"

        # Save comparison + classifier state, plus model config for reconstruction
        save_dict = {
            "epoch": epoch,
            "comparison_state_dict": self.comparison.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "model_config": self.model_config,
            "metrics": metrics,
            "cached_training": True,  # Flag indicating this needs assembly
        }
        torch.save(save_dict, path)
        logger.info(f"Saved checkpoint to {path}")

    def train(
        self,
        max_epochs: int = 5,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        patience: int = 5,
        grad_accum_steps: int = 1,
    ) -> dict[str, float]:
        params = list(self.comparison.parameters()) + list(self.classifier.parameters())
        optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        epochs_without_improvement = 0
        best_val_metrics = {}

        logger.info(f"Starting cached training (lr={learning_rate}, epochs={max_epochs})")

        for epoch in range(1, max_epochs + 1):
            logger.info(f"Epoch {epoch}/{max_epochs}")
            train_m = self._train_epoch(optimizer, grad_accum_steps)
            val_m = self._eval_epoch()
            scheduler.step()

            logger.info(f"  Train loss={train_m['loss']:.4f}, macro_f1={train_m['macro_f1']:.4f}")
            logger.info(f"  Val   loss={val_m['loss']:.4f}, macro_f1={val_m['macro_f1']:.4f}")

            if val_m["macro_f1"] > self.best_metric:
                self.best_metric = val_m["macro_f1"]
                self.best_epoch = epoch
                best_val_metrics = val_m
                self._save_checkpoint(epoch, val_m)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"Best macro_f1={self.best_metric:.4f} at epoch {self.best_epoch}")
        return best_val_metrics
