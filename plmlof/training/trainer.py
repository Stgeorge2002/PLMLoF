"""Training loop for PLMLoF with two-stage training support."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
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

            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
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
        # Linear warmup + cosine decay for stable training with randomly initialized heads
        warmup_epochs = min(10, max(1, max_epochs // 10))
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

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

    Trains only the ComparisonModule + ClassifierHead + optional RegressionHead
    on cached pooled embeddings — no ESM2 forward passes needed.

    Multi-task loss: L = CE_classification + α·SmoothL1_regression
    (α warms up linearly over the first few epochs)
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
        # Multi-task regression
        regressor: nn.Module | None = None,
        regression_weight: float = 0.1,
        regression_warmup_epochs: int = 3,
        # Model config for saving a reconstructable checkpoint
        esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
        pool_strategy: str = "mean_max",
        classifier_hidden_dims: list[int] | None = None,
        classifier_dropout: float = 0.3,
        lora_config: dict | None = None,
        focal_gamma: float = 2.0,
        use_cross_attention: bool = False,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
    ):
        self.device = torch.device(device)
        self.comparison = comparison.to(self.device)
        self.classifier = classifier.to(self.device)
        self.regressor = regressor.to(self.device) if regressor is not None else None
        self.regression_weight = regression_weight
        self.regression_warmup_epochs = regression_warmup_epochs
        self._current_epoch = 1
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LayerNorm for the 12-dim engineered features (different scales)
        # Use LayerNorm instead of BatchNorm to avoid crashes with batch_size=1
        from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES
        self.feature_norm = nn.LayerNorm(NUM_NUCLEOTIDE_FEATURES).to(self.device)

        # Optional pooled cross-attention between ref/var embeddings
        self.use_cross_attention = use_cross_attention
        self.cross_attn = None
        if use_cross_attention:
            from plmlof.models.comparison import PooledCrossAttention
            # hidden_size = dim of each pooled vector (1280 for ESM2-650M)
            hidden_size = comparison.hidden_size
            self.cross_attn = PooledCrossAttention(
                hidden_size=hidden_size, num_heads=cross_attn_heads,
                dropout=cross_attn_dropout,
            ).to(self.device)

        # CrossEntropyLoss — balanced data doesn't benefit from focal loss
        # (focal loss reduces gradient magnitude, causing mode collapse on balanced data)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.reg_criterion = nn.SmoothL1Loss()
        self.best_metric = 0.0
        self.best_epoch = -1

        # Persistent loss EMA across epochs for spike rejection
        self._loss_ema: float | None = None

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
            "has_regressor": regressor is not None,
            "regression_weight": regression_weight,
            "regression_warmup_epochs": regression_warmup_epochs,
            "focal_gamma": focal_gamma,
            "use_cross_attention": use_cross_attention,
            "cross_attn_heads": cross_attn_heads,
            "cross_attn_dropout": cross_attn_dropout,
        }

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reconstruct comparison features from cached pooled embeddings and classify.

        Returns:
            Tuple of (logits [batch, num_classes], regression_pred [batch] or None).
        """
        # Reconstruct pooled ref/var as [mean, max] concatenation
        ref_mean, ref_max = batch["ref_mean"], batch["ref_max"]
        var_mean, var_max = batch["var_mean"], batch["var_max"]

        # Optional cross-attention between the 4 pooled vectors
        if self.cross_attn is not None:
            # Stack as [B, 4, D] sequence: ref_mean, ref_max, var_mean, var_max
            tokens = torch.stack([ref_mean, ref_max, var_mean, var_max], dim=1)
            tokens = self.cross_attn(tokens)  # [B, 4, D]
            ref_mean, ref_max, var_mean, var_max = tokens[:, 0], tokens[:, 1], tokens[:, 2], tokens[:, 3]

        ref_pool = torch.cat([ref_mean, ref_max], dim=-1)
        var_pool = torch.cat([var_mean, var_max], dim=-1)

        diff_pool = ref_pool - var_pool
        prod_pool = ref_pool * var_pool
        comparison = torch.cat([diff_pool, prod_pool, ref_pool, var_pool], dim=-1)
        comparison = self.comparison.project(comparison)

        nuc_features = self.feature_norm(batch["nucleotide_features"])
        features = torch.cat([comparison, nuc_features], dim=-1)
        logits = self.classifier(features)

        reg_pred = None
        if self.regressor is not None:
            reg_pred = self.regressor(features)

        return logits, reg_pred

    def _compute_loss(self, logits, reg_pred, batch):
        """Compute combined classification + regression loss."""
        cls_loss = self.criterion(logits, batch["labels"])
        if reg_pred is not None and self.regressor is not None:
            dms_targets = batch["dms_scores"]
            reg_loss = self.reg_criterion(reg_pred, dms_targets)
            warmup_scale = min(1.0, self._current_epoch / max(self.regression_warmup_epochs, 1))
            effective_weight = self.regression_weight * warmup_scale
            total_loss = cls_loss + effective_weight * reg_loss
            return total_loss, cls_loss.item(), reg_loss.item()
        return cls_loss, cls_loss.item(), 0.0

    def _train_epoch(self, optimizer, grad_accum_steps: int = 1) -> dict[str, float]:
        self.comparison.train()
        self.classifier.train()
        self.feature_norm.train()
        if self.cross_attn is not None:
            self.cross_attn.train()
        if self.regressor is not None:
            self.regressor.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        all_preds, all_labels = [], []
        all_reg_preds, all_dms_targets = [], []
        optimizer.zero_grad()
        n_skipped = 0
        accum_count = 0
        n_counted = 0  # batches contributing to epoch loss

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits, reg_pred = self._forward(batch)
                loss, cls_l, reg_l = self._compute_loss(logits, reg_pred, batch)
                loss = loss / grad_accum_steps

            # Per-batch spike rejection using persistent EMA across epochs
            raw_loss = loss.item() * grad_accum_steps
            if self._loss_ema is None:
                self._loss_ema = raw_loss
            is_spike = self._loss_ema > 0.05 and raw_loss > 3.0 * self._loss_ema
            if is_spike:
                n_skipped += 1
                optimizer.zero_grad()
                accum_count = 0
                continue
            # Only update EMA for non-spike batches
            self._loss_ema = 0.02 * raw_loss + 0.98 * self._loss_ema

            # Mid-epoch instability detection: if running epoch loss is already
            # >2x the EMA after a significant number of batches, abort early.
            n_counted += 1
            total_loss += raw_loss
            if n_counted >= 50:
                running_avg = total_loss / n_counted
                if self._loss_ema > 0.05 and running_avg > 2.0 * self._loss_ema:
                    logger.warning(
                        f"  Mid-epoch abort: running loss {running_avg:.4f} > "
                        f"2x EMA {self._loss_ema:.4f} after {n_counted} batches"
                    )
                    optimizer.zero_grad()
                    # Signal the epoch loop to rollback
                    return {"loss": running_avg, "macro_f1": 0.0, "cls_loss": 0.0,
                            "reg_loss": 0.0, "spearman": 0.0, "_aborted": True}

            self.scaler.scale(loss).backward()
            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                self.scaler.unscale_(optimizer)
                params = (
                    list(self.comparison.parameters())
                    + list(self.classifier.parameters())
                    + list(self.feature_norm.parameters())
                    + (list(self.cross_attn.parameters()) if self.cross_attn else [])
                    + (list(self.regressor.parameters()) if self.regressor else [])
                )
                for p in params:
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                accum_count = 0

            total_cls_loss += cls_l
            total_reg_loss += reg_l
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            if reg_pred is not None:
                all_reg_preds.extend(reg_pred.detach().float().cpu().numpy())
                all_dms_targets.extend(batch["dms_scores"].cpu().numpy())
            pbar.set_postfix({"loss": f"{raw_loss:.4f}"})

        if n_skipped > 0:
            logger.warning(f"  Skipped {n_skipped} spike batches (loss > 3x EMA)")

        n_batches = max(n_counted, 1)
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics["loss"] = total_loss / n_batches
        metrics["cls_loss"] = total_cls_loss / n_batches
        metrics["reg_loss"] = total_reg_loss / n_batches
        if all_reg_preds:
            from scipy.stats import spearmanr
            arr_t, arr_p = np.array(all_dms_targets), np.array(all_reg_preds)
            if arr_t.std() > 0 and arr_p.std() > 0:
                rho, _ = spearmanr(arr_t, arr_p)
                metrics["spearman"] = float(rho) if not np.isnan(rho) else 0.0
            else:
                metrics["spearman"] = 0.0
        return metrics

    @torch.no_grad()
    def _eval_epoch(self) -> dict[str, float]:
        self.comparison.eval()
        self.classifier.eval()
        self.feature_norm.eval()
        if self.cross_attn is not None:
            self.cross_attn.eval()
        if self.regressor is not None:
            self.regressor.eval()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        all_reg_preds, all_dms_targets = [], []

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits, reg_pred = self._forward(batch)
            loss, cls_l, reg_l = self._compute_loss(logits, reg_pred, batch)
            total_loss += loss.item()
            total_cls_loss += cls_l
            total_reg_loss += reg_l
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=-1).float().cpu().numpy())
            if reg_pred is not None:
                all_reg_preds.extend(reg_pred.float().cpu().numpy())
                all_dms_targets.extend(batch["dms_scores"].cpu().numpy())

        n_batches = max(len(self.val_loader), 1)
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels), np.array(all_probs))
        metrics["loss"] = total_loss / n_batches
        metrics["cls_loss"] = total_cls_loss / n_batches
        metrics["reg_loss"] = total_reg_loss / n_batches
        if all_reg_preds:
            from scipy.stats import spearmanr
            arr_t, arr_p = np.array(all_dms_targets), np.array(all_reg_preds)
            if arr_t.std() > 0 and arr_p.std() > 0:
                rho, _ = spearmanr(arr_t, arr_p)
                metrics["spearman"] = float(rho) if not np.isnan(rho) else 0.0
            else:
                metrics["spearman"] = 0.0
            metrics["mae"] = float(np.mean(np.abs(arr_t - arr_p)))
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save checkpoint compatible with the full PLMLoFModel for inference."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "model_best.pt"

        # Save comparison + classifier + regressor state, plus model config for reconstruction
        save_dict = {
            "epoch": epoch,
            "comparison_state_dict": self.comparison.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "feature_norm_state_dict": self.feature_norm.state_dict(),
            "model_config": self.model_config,
            "metrics": metrics,
            "cached_training": True,  # Flag indicating this needs assembly
        }
        if self.cross_attn is not None:
            save_dict["cross_attn_state_dict"] = self.cross_attn.state_dict()
        if self.regressor is not None:
            save_dict["regressor_state_dict"] = self.regressor.state_dict()
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
        # Resume from checkpoint if it exists
        ckpt_path = self.output_dir / "checkpoints" / "model_best.pt"
        start_epoch = 1
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            if ckpt.get("cached_training"):
                try:
                    self.comparison.load_state_dict(ckpt["comparison_state_dict"])
                    self.classifier.load_state_dict(ckpt["classifier_state_dict"])
                    if "feature_norm_state_dict" in ckpt:
                        self.feature_norm.load_state_dict(ckpt["feature_norm_state_dict"])
                    if self.cross_attn is not None and "cross_attn_state_dict" in ckpt:
                        self.cross_attn.load_state_dict(ckpt["cross_attn_state_dict"])
                    if self.regressor is not None and "regressor_state_dict" in ckpt:
                        self.regressor.load_state_dict(ckpt["regressor_state_dict"])
                    start_epoch = ckpt.get("epoch", 0) + 1
                    self.best_metric = ckpt.get("metrics", {}).get("macro_f1", 0.0)
                    self.best_epoch = ckpt.get("epoch", 0)
                    logger.info(f"Resumed from epoch {ckpt.get('epoch')}, best macro_f1={self.best_metric:.4f}")
                except (RuntimeError, KeyError) as e:
                    logger.warning(f"Could not resume from checkpoint (architecture changed): {e}")
                    logger.warning("Training from scratch.")

        params = list(self.comparison.parameters()) + list(self.classifier.parameters()) + list(self.feature_norm.parameters())
        if self.cross_attn is not None:
            params += list(self.cross_attn.parameters())
        if self.regressor is not None:
            params += list(self.regressor.parameters())
        optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        # Linear warmup + cosine decay — critical for large randomly-initialized layers
        warmup_epochs = min(10, max(1, max_epochs // 10))
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
        )

        # Fast-forward scheduler if resuming from a checkpoint
        for _ in range(1, start_epoch):
            scheduler.step()

        epochs_without_improvement = 0
        best_val_metrics = {}

        logger.info(f"Starting cached training (lr={learning_rate}, epochs={start_epoch}-{max_epochs})")

        for epoch in range(start_epoch, max_epochs + 1):
            self._current_epoch = epoch
            logger.info(f"Epoch {epoch}/{max_epochs}")
            train_m = self._train_epoch(optimizer, grad_accum_steps)

            # Mid-epoch abort: restore best weights and retry this epoch
            if train_m.get("_aborted"):
                logger.warning(f"  Restoring best weights after mid-epoch abort")
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    self.comparison.load_state_dict(ckpt["comparison_state_dict"])
                    self.classifier.load_state_dict(ckpt["classifier_state_dict"])
                    if "feature_norm_state_dict" in ckpt:
                        self.feature_norm.load_state_dict(ckpt["feature_norm_state_dict"])
                    if self.cross_attn is not None and "cross_attn_state_dict" in ckpt:
                        self.cross_attn.load_state_dict(ckpt["cross_attn_state_dict"])
                    if self.regressor is not None and "regressor_state_dict" in ckpt:
                        self.regressor.load_state_dict(ckpt["regressor_state_dict"])
                scheduler.step()
                continue

            val_m = self._eval_epoch()
            scheduler.step()

            logger.info(f"  Train loss={train_m['loss']:.4f}, macro_f1={train_m['macro_f1']:.4f}" +
                        (f", spearman={train_m.get('spearman', 0):.4f}" if self.regressor else ""))
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"  Val   loss={val_m['loss']:.4f}, macro_f1={val_m['macro_f1']:.4f}" +
                        (f", spearman={val_m.get('spearman', 0):.4f}" if self.regressor else "") +
                        f", lr={current_lr:.2e}")

            # Epoch-level rollback: if val F1 collapses (>50% drop from best),
            # restore best checkpoint weights and skip this epoch entirely.
            if self.best_metric > 0.3 and val_m["macro_f1"] < 0.5 * self.best_metric:
                logger.warning(
                    f"  Epoch-level rollback: val macro_f1={val_m['macro_f1']:.4f} "
                    f"collapsed vs best={self.best_metric:.4f} — restoring best weights"
                )
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                self.comparison.load_state_dict(ckpt["comparison_state_dict"])
                self.classifier.load_state_dict(ckpt["classifier_state_dict"])
                if "feature_norm_state_dict" in ckpt:
                    self.feature_norm.load_state_dict(ckpt["feature_norm_state_dict"])
                if self.cross_attn is not None and "cross_attn_state_dict" in ckpt:
                    self.cross_attn.load_state_dict(ckpt["cross_attn_state_dict"])
                if self.regressor is not None and "regressor_state_dict" in ckpt:
                    self.regressor.load_state_dict(ckpt["regressor_state_dict"])
                continue  # don't count toward patience

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
