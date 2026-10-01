"""Inference: wreck rule, then LoF ensemble + two GoF ensembles."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from plmlof.data.features import extract_nucleotide_features
from plmlof.inference.vcf_handler import VariantRecord, parse_fasta_pairs
from plmlof.v2 import GOF_CALL_THRESHOLD, IN_FAMILY_COSINE, TASKS
from plmlof.v2.applicability import in_family_flags, load_gallery
from plmlof.v2.embed import _pool
from plmlof.v2.model import V2TaskNet
from plmlof.v2.stats import benjamini_hochberg, empirical_p
from plmlof.v2.wreck import wreck_grade

logger = logging.getLogger(__name__)


def _load_net(ckpt_path: Path, device: torch.device) -> V2TaskNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    net = V2TaskNet(
        hidden_size=cfg["hidden_size"],
        task=ckpt["task"],
        pool_strategy=cfg.get("pool_strategy", "mean_max"),
        use_cross_attention=cfg.get("use_cross_attention", False),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net.to(device)


def discover_ensemble(task_dir: Path) -> list[Path]:
    """Prefer ensemble.json; else seed*/checkpoints/model_best.pt."""
    listed = task_dir / "ensemble.json"
    if listed.exists():
        payload = json.loads(listed.read_text())
        return [task_dir / p if not Path(p).is_absolute() else Path(p) for p in payload["members"]]
    found = sorted(task_dir.glob("seed*/checkpoints/model_best.pt"))
    if not found:
        best = task_dir / "checkpoints" / "model_best.pt"
        if best.exists():
            found = [best]
    return found


class V2Predictor:
    """One ESM2 encoder, up to three 5-member ensembles, wreck rule, p-values."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 1024,
        gof_threshold: float = GOF_CALL_THRESHOLD,
        in_family_threshold: float = IN_FAMILY_COSINE,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.gof_threshold = gof_threshold
        self.in_family_threshold = in_family_threshold

        meta_path = self.model_dir / "model_config.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {"esm2_model_name": "facebook/esm2_t33_650M_UR50D"}
        esm_name = meta.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D")
        logger.info("Loading ESM2 %s", esm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(esm_name)
        self.encoder = AutoModel.from_pretrained(esm_name).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.ensembles: dict[str, list[V2TaskNet]] = {}
        self.nulls: dict[str, np.ndarray] = {}
        self.galleries: dict[str, dict] = {}
        for task in TASKS:
            tdir = self.model_dir / task
            members = discover_ensemble(tdir) if tdir.exists() else []
            if not members:
                logger.warning("No %s ensemble under %s", task, tdir)
                continue
            self.ensembles[task] = [_load_net(p, self.device) for p in members]
            null_path = tdir / "null_scores.pt"
            if null_path.exists():
                self.nulls[task] = torch.load(null_path, map_location="cpu", weights_only=False)["scores"].numpy()
            gal_path = tdir / "train_gallery.pt"
            if gal_path.exists():
                self.galleries[task] = load_gallery(gal_path)
            logger.info("%s: %s members, null=%s", task, len(members), task in self.nulls)

    @torch.no_grad()
    def _embed_pairs(
        self, refs: list[str], vars_: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nuc = torch.stack([
            extract_nucleotide_features(r, v) for r, v in zip(refs, vars_)
        ]).to(self.device)
        seqs = refs + vars_
        enc = self.tokenizer(
            seqs, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        hidden = self.encoder(ids, attention_mask=mask).last_hidden_state
        mean_p, max_p = _pool(hidden, mask)
        n = len(refs)
        return mean_p[:n], max_p[:n], mean_p[n:], max_p[n:], nuc

    @torch.no_grad()
    def _ensemble_measure(
        self, task: str, ref_mean, ref_max, var_mean, var_max, nuc,
    ) -> tuple[np.ndarray, np.ndarray]:
        measures = []
        for net in self.ensembles[task]:
            raw = net.forward_from_pooled(ref_mean, ref_max, var_mean, var_max, nuc)
            measures.append(net.probability(raw).float().cpu())
        stacked = torch.stack(measures, dim=0)
        return stacked.mean(0).numpy(), stacked.std(0).numpy()

    def predict_records(self, records: list[VariantRecord]) -> list[dict]:
        results: list[dict] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start:start + self.batch_size]
            refs = [r.ref_protein.replace("*", "")[:self.max_seq_length] for r in batch]
            vars_ = [r.var_protein.replace("*", "")[:self.max_seq_length] for r in batch]
            wreck_flags = [
                wreck_grade(r.ref_protein, r.var_protein, r.ref_dna, r.var_dna) for r in batch
            ]
            ref_mean, ref_max, var_mean, var_max, nuc = self._embed_pairs(refs, vars_)
            measures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            family: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
            for task in self.ensembles:
                measures[task] = self._ensemble_measure(task, ref_mean, ref_max, var_mean, var_max, nuc)
                if task in self.galleries:
                    family[task] = in_family_flags(ref_mean.cpu(), self.galleries[task], self.in_family_threshold)

            lof_mean, lof_sd = measures.get("lof", (np.zeros(len(batch)), np.zeros(len(batch))))
            lof_mean = np.array(lof_mean, dtype=np.float64, copy=True)
            lof_sd = np.array(lof_sd, dtype=np.float64, copy=True)
            # Rule prior: sure wreck → 1.0; post-domain tail → ~0.4; missense → model.
            for i, (structural, kind, prior) in enumerate(wreck_flags):
                if prior is not None:
                    lof_mean[i] = float(prior)
                    lof_sd[i] = 0.0

            pvals: dict[str, np.ndarray] = {}
            for task, (mean, _) in measures.items():
                scores = lof_mean if task == "lof" else mean
                if task == "lof":
                    scores = lof_mean
                if task in self.nulls:
                    in_fam = family[task][0] if task in family else np.ones(len(batch), dtype=bool)
                    p = empirical_p(scores, self.nulls[task])
                    p = np.where(in_fam, p, np.nan)
                    pvals[task] = p
                else:
                    pvals[task] = np.full(len(batch), np.nan)

            qvals = {}
            for task, p in pvals.items():
                finite = np.isfinite(p)
                q = np.full_like(p, np.nan)
                if finite.any():
                    q[finite] = benjamini_hochberg(p[finite])
                qvals[task] = q

            for i, rec in enumerate(batch):
                structural, kind, prior = wreck_flags[i]
                row = {
                    "gene": rec.gene,
                    "wreck": structural,
                    "wreck_kind": kind,
                    "lof_score": float(lof_mean[i]),
                    "lof_sd": float(lof_sd[i]),
                    "lof_p": float(pvals["lof"][i]) if "lof" in pvals else float("nan"),
                    "lof_q": float(qvals["lof"][i]) if "lof" in qvals else float("nan"),
                }
                for task, key in (("growth_gof", "growth_gof"), ("amr_gof", "amr_gof")):
                    if task in measures:
                        mean, sd = measures[task]
                        in_fam, nearest, sim = family.get(
                            task, (np.ones(len(batch), dtype=bool), [""] * len(batch), np.ones(len(batch))),
                        )
                        p_hat = float(mean[i])
                        call = (
                            p_hat >= self.gof_threshold
                            and bool(in_fam[i])
                            and np.isfinite(pvals[task][i])
                            and pvals[task][i] < 0.05
                        )
                        row[f"{key}_p"] = p_hat
                        row[f"{key}_sd"] = float(sd[i])
                        row[f"{key}_p_emp"] = float(pvals[task][i])
                        row[f"{key}_q"] = float(qvals[task][i])
                        row[f"{key}_call"] = bool(call)
                        row[f"{key}_in_family"] = bool(in_fam[i])
                        row[f"{key}_nearest"] = nearest[i]
                        row[f"{key}_cosine"] = float(sim[i])
                    else:
                        row[f"{key}_p"] = float("nan")
                        row[f"{key}_sd"] = float("nan")
                        row[f"{key}_p_emp"] = float("nan")
                        row[f"{key}_q"] = float("nan")
                        row[f"{key}_call"] = False
                        row[f"{key}_in_family"] = False
                        row[f"{key}_nearest"] = ""
                        row[f"{key}_cosine"] = float("nan")

                if "lof" in family:
                    row["in_family"] = bool(family["lof"][0][i])
                    row["nearest_train_gene"] = family["lof"][1][i]
                    row["ref_cosine"] = float(family["lof"][2][i])
                else:
                    row["in_family"] = False
                    row["nearest_train_gene"] = ""
                    row["ref_cosine"] = float("nan")
                results.append(row)
        return results

    def predict_fasta(self, reference: str | Path, variants: str | Path) -> list[dict]:
        records = parse_fasta_pairs(reference, variants)
        return self.predict_records(records)
