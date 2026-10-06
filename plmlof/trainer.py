"""Trainer for a single task head on cached ESM2 embeddings."""

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

from plmlof.constants import REGRESSION_TASKS
from plmlof.metrics import collapse_is_fail, gof_metrics, lof_metrics, mlof_metrics
from plmlof.model import TaskNet
from plmlof.rank import ranknet_loss

logger = logging.getLogger(__name__)


def _as_str_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(v) for v in list(value)]


class Trainer:
    def __init__(
        self,
        net: TaskNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str | torch.device = "cpu",
        output_dir: str | Path = "outputs/",
        mixed_precision: str = "no",
        gof_threshold: float = 0.90,
        esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
        pool_strategy: str = "mean_max",
        seed: int = 0,
        rank_loss_weight: float = 0.7,
        regression_loss_weight: float = 0.3,
    ):
        self.net = net.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task = net.task
        self.gof_threshold = gof_threshold
        self.seed = seed
        self.rank_loss_weight = float(rank_loss_weight)
        self.regression_loss_weight = float(regression_loss_weight)
        self.use_amp = mixed_precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.amp_dtype = (
            torch.float16 if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16"
            else torch.float32
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision == "fp16" and self.use_amp))
        self.best_metric = -1e9
        self.best_epoch = -1
        mlp = net.head.mlp
        self.model_config = {
            "plmlof": True,
            "task": net.task,
            "esm2_model_name": esm2_model_name,
            "pool_strategy": pool_strategy,
            "seed": seed,
            "gof_threshold": gof_threshold,
            "hidden_size": net.hidden_size,
            "head_hidden": int(mlp[0].out_features),
            "dropout": float(getattr(mlp[2], "p", 0.2)),
            "alignment_free": net.alignment_free,
            "uses_sites": net.uses_sites,
            "use_cross_attention": (
                net.comparison is not None and net.comparison.cross_attn is not None
            ),
        }
        self.lof_loss = nn.SmoothL1Loss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _raw(self, batch: dict) -> torch.Tensor:
        return self.net.forward_from_cache(
            batch["ref_mean"], batch["ref_max"],
            batch["var_mean"], batch["var_max"],
            batch["nucleotide_features"],
            site_ref=batch.get("site_ref"),
            site_var=batch.get("site_var"),
        )

    def _step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._raw(batch)
        target = batch["target"].float()
        weight = batch["sample_weight"].float().clamp(min=1e-6)
        if self.task == "mlof":
            per = self.lof_loss(raw, target)
            reg = (per * weight).sum() / weight.sum()
            pids = _as_str_list(batch.get("protein_id", []))
            rank = ranknet_loss(raw, batch["dms_zscore"].float(), pids, batch["is_missense"])
            loss = self.regression_loss_weight * reg + self.rank_loss_weight * rank
        elif self.task in REGRESSION_TASKS:
            per = self.lof_loss(raw, target)
            loss = (per * weight).sum() / weight.sum()
        else:
            per = self.bce(raw, target)
            loss = (per * weight).sum() / weight.sum()
        measure = self.net.probability(raw)
        return loss, measure

    def _train_epoch(self, optimizer: AdamW, grad_accum: int, epoch: int) -> float:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.net.train()
        total = 0.0
        optimizer.zero_grad()
        n = 0
        pbar = tqdm(self.train_loader, desc="train", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                loss, _ = self._step(batch)
                loss = loss / grad_accum
            self.scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            total += loss.item() * grad_accum
            n += 1
            pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})
        return total / max(n, 1)

    @torch.no_grad()
    def _eval_epoch(self) -> dict[str, float]:
        self.net.eval()
        preds, targets, zs, wrecks, misses = [], [], [], [], []
        genes: list[str] = []
        pids: list[str] = []
        total = 0.0
        n = 0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, measure = self._step(batch)
            total += loss.item()
            n += 1
            preds.append(measure.float().cpu().numpy())
            targets.append(batch["target"].float().cpu().numpy())
            zs.append(batch["dms_zscore"].float().cpu().numpy())
            wrecks.append(batch["is_wreck"].cpu().numpy())
            misses.append(batch["is_missense"].cpu().numpy())
            genes.extend(_as_str_list(batch.get("gene", [])))
            pids.extend(_as_str_list(batch.get("protein_id", [])))
        pred = np.concatenate(preds) if preds else np.array([])
        target = np.concatenate(targets) if targets else np.array([])
        z = np.concatenate(zs) if zs else np.array([])
        is_wreck = np.concatenate(wrecks).astype(bool) if wrecks else np.array([], dtype=bool)
        is_missense = np.concatenate(misses).astype(bool) if misses else np.array([], dtype=bool)
        metrics: dict[str, float] = {"loss": total / max(n, 1)}
        if self.task == "mlof":
            metrics.update(mlof_metrics(
                pred, target, z, is_missense, gene=genes, protein_id=pids,
            ))
        elif self.task in REGRESSION_TASKS:
            metrics.update(lof_metrics(pred, target, z, is_wreck, is_missense))
        else:
            metrics.update(gof_metrics(pred, target, threshold=self.gof_threshold, gene=genes))
        return metrics

    def _save(self, epoch: int, metrics: dict) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "model_best.pt"
        torch.save(
            {
                "plmlof": True,
                "task": self.task,
                "epoch": epoch,
                "seed": self.seed,
                "state_dict": self.net.state_dict(),
                "model_config": self.model_config,
                "metrics": metrics,
            },
            path,
        )
        logger.info("Saved %s", path)

    def train(
        self,
        max_epochs: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        patience: int = 5,
        grad_accum_steps: int = 1,
        warmup_ratio: float = 0.1,
    ) -> dict[str, float]:
        optimizer = AdamW(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        warmup_epochs = max(1, round(max_epochs * warmup_ratio))
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

        best: dict[str, float] = {}
        stale = 0
        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(optimizer, grad_accum_steps, epoch)
            val = self._eval_epoch()
            scheduler.step()
            selection = float(val.get("selection", 0.0))
            logger.info(
                "Epoch %s/%s train_loss=%.4f val_loss=%.4f selection=%.4f %s",
                epoch, max_epochs, train_loss, val["loss"], selection,
                {k: round(v, 4) for k, v in val.items() if k not in {"n", "loss"}},
            )
            collapsed = self.task == "mlof" and collapse_is_fail(val)
            cheat = (
                self.task == "mlof"
                and float(val.get("wreck_auroc", 0)) > 0.97
                and abs(float(val.get("missense_spearman", 0))) < 0.05
            )
            if collapsed:
                logger.warning(
                    "Skipping checkpoint: gene-prior collapse fraction=%.3f",
                    float(val.get("collapse_fraction", 0)),
                )
            if cheat:
                logger.warning(
                    "Skipping checkpoint: wreck AUROC=%.3f missense Spearman=%.3f (length cheat)",
                    val["wreck_auroc"], val["missense_spearman"],
                )
            blocked = collapsed or cheat
            if blocked:
                if self.best_epoch < 0:
                    self._save(epoch, val)
                    self.best_epoch = epoch
                    best = dict(val)
                stale += 1
                if stale >= patience:
                    logger.info("Early stopping at epoch %s", epoch)
                    break
                continue
            if selection > self.best_metric:
                self.best_metric = selection
                self.best_epoch = epoch
                best = dict(val)
                self._save(epoch, val)
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    logger.info("Early stopping at epoch %s", epoch)
                    break
        if self.best_metric <= -1e8:
            logger.info("Only fallback checkpoint saved at epoch %s", self.best_epoch)
        else:
            logger.info("Best selection=%.4f at epoch %s", self.best_metric, self.best_epoch)
        return best
