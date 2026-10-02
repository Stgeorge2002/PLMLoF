"""Inference: alignment-free LoF, pairwise MLoF, two GoF ensembles."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from plmlof.applicability import in_family_flags, load_gallery
from plmlof.constants import GOF_CALL_THRESHOLD, IN_FAMILY_COSINE, TASKS
from plmlof.data.features import extract_nucleotide_features
from plmlof.embed import _pool
from plmlof.encoders import esm2_for_task
from plmlof.inference.vcf_handler import VariantRecord, parse_fasta_pairs, parse_protein_fasta
from plmlof.labels import display_bin
from plmlof.model import TaskNet
from plmlof.sites import aligned_site_index, gather_site_windows
from plmlof.stats import benjamini_hochberg, empirical_p
from plmlof.wreck import wreck_grade

logger = logging.getLogger(__name__)


def _load_net(ckpt_path: Path, device: torch.device) -> TaskNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    net = TaskNet(
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


class Predictor:
    """LoF on ESM2-35M; MLoF/GoF on ESM2-650M. Encoders load lazily."""

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
        self._encoders: dict[str, tuple[object, object]] = {}
        self.task_esm: dict[str, str] = {}

        self.ensembles: dict[str, list[TaskNet]] = {}
        self.nulls: dict[str, np.ndarray] = {}
        self.galleries: dict[str, dict] = {}
        for task in TASKS:
            tdir = self.model_dir / task
            members = discover_ensemble(tdir) if tdir.exists() else []
            if not members:
                logger.warning("No %s ensemble under %s", task, tdir)
                continue
            self.ensembles[task] = [_load_net(p, self.device) for p in members]
            self.task_esm[task] = self._task_encoder_name(task, tdir, members[0])
            null_path = tdir / "null_scores.pt"
            if null_path.exists():
                self.nulls[task] = torch.load(null_path, map_location="cpu", weights_only=False)["scores"].numpy()
            gal_path = tdir / "train_gallery.pt"
            if gal_path.exists():
                self.galleries[task] = load_gallery(gal_path)
            logger.info("%s: %s members, encoder=%s, null=%s", task, len(members), self.task_esm[task], task in self.nulls)

    def _task_encoder_name(self, task: str, tdir: Path, ckpt_path: Path) -> str:
        for path in (tdir / "model_config.json", ckpt_path):
            if not path.exists():
                continue
            if path.suffix == ".json":
                meta = json.loads(path.read_text())
            else:
                meta = torch.load(path, map_location="cpu", weights_only=False).get("model_config", {})
            name = meta.get("esm2_model_name")
            if name:
                return str(name)
        return esm2_for_task(task)

    def _get_encoder(self, esm_name: str):
        if esm_name not in self._encoders:
            logger.info("Loading ESM2 %s", esm_name)
            tokenizer = AutoTokenizer.from_pretrained(esm_name)
            encoder = AutoModel.from_pretrained(esm_name).to(self.device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            self._encoders[esm_name] = (tokenizer, encoder)
        return self._encoders[esm_name]

    @torch.no_grad()
    def _embed_hidden(self, seqs: list[str], esm_name: str):
        tokenizer, encoder = self._get_encoder(esm_name)
        enc = tokenizer(
            seqs, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        hidden = encoder(ids, attention_mask=mask).last_hidden_state
        mean_p, max_p = _pool(hidden, mask)
        return hidden, mean_p, max_p

    @torch.no_grad()
    def _embed_seqs(self, seqs: list[str], esm_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean_p, max_p = self._embed_hidden(seqs, esm_name)
        return mean_p, max_p

    @torch.no_grad()
    def _embed_pairs(
        self, refs: list[str], vars_: list[str], esm_name: str,
    ) -> tuple[torch.Tensor, ...]:
        nuc = torch.stack([
            extract_nucleotide_features(r, v) for r, v in zip(refs, vars_)
        ]).to(self.device)
        hidden, mean_p, max_p = self._embed_hidden(refs + vars_, esm_name)
        n = len(refs)
        centers = [aligned_site_index(r, v) for r, v in zip(refs, vars_)]
        site_ref = gather_site_windows(hidden[:n], refs, centers)
        site_var = gather_site_windows(hidden[n:], vars_, centers)
        return mean_p[:n], max_p[:n], mean_p[n:], max_p[n:], nuc, site_ref, site_var

    @torch.no_grad()
    def _ensemble_measure(
        self, task: str, ref_mean, ref_max, var_mean, var_max, nuc,
        site_ref=None, site_var=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        measures = []
        for net in self.ensembles[task]:
            raw = net.forward_from_cache(
                ref_mean, ref_max, var_mean, var_max, nuc,
                site_ref=site_ref, site_var=site_var,
            )
            measures.append(net.probability(raw).float().cpu())
        stacked = torch.stack(measures, dim=0)
        return stacked.mean(0).numpy(), stacked.std(0).numpy()

    def predict_records(self, records: list[VariantRecord], *, pair: bool = True) -> list[dict]:
        results: list[dict] = []
        tasks = list(self.ensembles)
        if not pair:
            tasks = [t for t in tasks if t == "lof"]
        for start in range(0, len(records), self.batch_size):
            batch = records[start:start + self.batch_size]
            refs = [r.ref_protein.replace("*", "")[:self.max_seq_length] for r in batch]
            vars_ = [r.var_protein.replace("*", "")[:self.max_seq_length] for r in batch]
            if pair:
                wreck_flags = [
                    wreck_grade(r.ref_protein, r.var_protein, r.ref_dna, r.var_dna) for r in batch
                ]
            else:
                wreck_flags = [(False, "none", None)] * len(batch)

            pooled: dict[str, tuple] = {}
            pair_tasks = [t for t in tasks if t != "lof"]
            if "lof" in tasks:
                lof_esm = self.task_esm["lof"]
                var_mean, var_max = self._embed_seqs(vars_, lof_esm)
                zeros = torch.zeros(len(batch), 12, device=self.device)
                pooled["lof"] = (var_mean, var_max, var_mean, var_max, zeros, None, None)
            if pair_tasks:
                pair_esm = self.task_esm[pair_tasks[0]]
                pooled["_pair"] = self._embed_pairs(refs, vars_, pair_esm)

            measures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            family: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
            for task in tasks:
                if task == "lof":
                    ref_mean, ref_max, var_mean, var_max, nuc, site_ref, site_var = pooled["lof"]
                else:
                    ref_mean, ref_max, var_mean, var_max, nuc, site_ref, site_var = pooled["_pair"]
                measures[task] = self._ensemble_measure(
                    task, ref_mean, ref_max, var_mean, var_max, nuc,
                    site_ref=site_ref, site_var=site_var,
                )
                if task in self.galleries:
                    query = var_mean if task == "lof" else ref_mean
                    family[task] = in_family_flags(query.cpu(), self.galleries[task], self.in_family_threshold)

            pvals: dict[str, np.ndarray] = {}
            for task, (mean, _) in measures.items():
                if task in self.nulls:
                    in_fam = family[task][0] if task in family else np.ones(len(batch), dtype=bool)
                    p = empirical_p(mean, self.nulls[task])
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
                lof_mean, lof_sd = measures.get("lof", (np.zeros(len(batch)), np.zeros(len(batch))))
                mlof_mean, mlof_sd = measures.get("mlof", (np.full(len(batch), np.nan), np.full(len(batch), np.nan)))
                row = {
                    "gene": rec.gene,
                    "wreck": structural,
                    "wreck_kind": kind,
                    "wreck_prior": float(prior) if prior is not None else float("nan"),
                    "lof_score": float(lof_mean[i]),
                    "lof_sd": float(lof_sd[i]),
                    "lof_p": float(pvals["lof"][i]) if "lof" in pvals else float("nan"),
                    "lof_q": float(qvals["lof"][i]) if "lof" in qvals else float("nan"),
                    "mlof_score": float(mlof_mean[i]),
                    "mlof_bin": float(display_bin(float(mlof_mean[i]))) if np.isfinite(mlof_mean[i]) else float("nan"),
                    "mlof_sd": float(mlof_sd[i]),
                    "mlof_p": float(pvals["mlof"][i]) if "mlof" in pvals else float("nan"),
                    "mlof_q": float(qvals["mlof"][i]) if "mlof" in qvals else float("nan"),
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

                fam_task = "lof" if "lof" in family else ("mlof" if "mlof" in family else None)
                if fam_task:
                    row["in_family"] = bool(family[fam_task][0][i])
                    row["nearest_train_gene"] = family[fam_task][1][i]
                    row["ref_cosine"] = float(family[fam_task][2][i])
                else:
                    row["in_family"] = False
                    row["nearest_train_gene"] = ""
                    row["ref_cosine"] = float("nan")
                results.append(row)
        return results

    def predict_fasta(self, reference: str | Path, variants: str | Path) -> list[dict]:
        records = parse_fasta_pairs(reference, variants)
        return self.predict_records(records, pair=True)

    def predict_proteins(self, proteins: str | Path) -> list[dict]:
        """Alignment-free LoF: one protein FASTA, no reference pair."""
        records = parse_protein_fasta(proteins)
        return self.predict_records(records, pair=False)
