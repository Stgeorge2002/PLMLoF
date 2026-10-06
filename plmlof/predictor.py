"""Inference: alignment-free LoF, pairwise MLoF, two GoF ensembles."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from plmlof.applicability import in_family_flags, load_gallery
from plmlof.chem import aa_token_ids, site_chem_at, site_llr
from plmlof.constants import GOF_CALL_THRESHOLD, IN_FAMILY_COSINE, SITE_RADIUS, TRAIN_TASKS
from plmlof.data.features import extract_nucleotide_features
from plmlof.domains import DomainIndex, classify_position
from plmlof.embed import _pool, forward_hidden_and_logits
from plmlof.encoders import esm2_for_task
from plmlof.inference.vcf_handler import VariantRecord, parse_fasta_pairs, parse_protein_fasta
from plmlof.labels import display_bin
from plmlof.model import TaskNet
from plmlof.sites import aligned_site_index, gather_site_windows, missense_sites, token_index
from plmlof.stats import benjamini_hochberg, empirical_p
from plmlof.wreck import wreck_grade

logger = logging.getLogger(__name__)


def _head_hparams(ckpt: dict) -> tuple[int, float]:
    """Head width/dropout as trained. Infer width from weights if older ckpts omit it."""
    cfg = ckpt.get("model_config") or {}
    hidden = cfg.get("head_hidden")
    if hidden is None:
        weight = ckpt["state_dict"].get("head.mlp.0.weight")
        hidden = int(weight.shape[0]) if weight is not None else 128
    return int(hidden), float(cfg.get("dropout", 0.2))


def _mlof_arch(ckpt: dict) -> tuple[int, int]:
    """(site_window, chem_dim) from config or weight shapes."""
    cfg = ckpt.get("model_config") or {}
    hidden = int(cfg["hidden_size"])
    sd = ckpt["state_dict"]
    window = cfg.get("site_window")
    if window is None:
        weight = sd.get("site_compare.proj.0.weight")
        window = int(weight.shape[1] / hidden) - 2 if weight is not None else 5
    chem = cfg.get("chem_dim")
    if chem is None:
        head_w = sd.get("head.mlp.0.weight")
        chem = max(int(head_w.shape[1]) - hidden, 0) if head_w is not None else 0
    return int(window), int(chem)


def _load_net(ckpt_path: Path, device: torch.device) -> TaskNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    head_hidden, dropout = _head_hparams(ckpt)
    extra: dict = {}
    if ckpt["task"] == "mlof":
        window, chem = _mlof_arch(ckpt)
        extra["site_window"] = window
        extra["chem_dim"] = chem
    else:
        extra["chem_dim"] = 0
    net = TaskNet(
        hidden_size=cfg["hidden_size"],
        task=ckpt["task"],
        pool_strategy=cfg.get("pool_strategy", "mean_max"),
        use_cross_attention=cfg.get("use_cross_attention", False),
        head_hidden=head_hidden,
        dropout=dropout,
        **extra,
    )
    net.ablate_logodds = bool(cfg.get("ablate_logodds", False))
    net.ablate_domain = bool(cfg.get("ablate_domain", False))
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
        domains: str | Path | None = None,
        hmm: str | Path | None = None,
        hmm_cpus: int = 1,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.gof_threshold = gof_threshold
        self.in_family_threshold = in_family_threshold
        self.domains = None
        if domains or hmm:
            self.domains = DomainIndex(
                parquet=Path(domains) if domains else None,
                hmm=Path(hmm) if hmm else None,
                cpus=hmm_cpus,
            )
        self._encoders: dict[str, tuple[object, object]] = {}
        self.task_esm: dict[str, str] = {}

        self.ensembles: dict[str, list[TaskNet]] = {}
        self.nulls: dict[str, np.ndarray] = {}
        self.galleries: dict[str, dict] = {}
        # GoF: iterate TASKS instead of TRAIN_TASKS to load growth_gof / amr_gof again.
        for task in TRAIN_TASKS:
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

    def _get_encoder(self, esm_name: str, *, masked_lm: bool = False):
        key = f"{esm_name}|mlm" if masked_lm else esm_name
        if key not in self._encoders:
            logger.info("Loading ESM2 %s%s", esm_name, " (MLM)" if masked_lm else "")
            tokenizer = AutoTokenizer.from_pretrained(esm_name)
            loader = AutoModelForMaskedLM if masked_lm else AutoModel
            encoder = loader.from_pretrained(esm_name).to(self.device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            self._encoders[key] = (tokenizer, encoder)
        return self._encoders[key]

    @torch.no_grad()
    def _embed_hidden(self, seqs: list[str], esm_name: str, *, masked_lm: bool = False):
        tokenizer, encoder = self._get_encoder(esm_name, masked_lm=masked_lm)
        enc = tokenizer(
            seqs, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        hidden, logits = forward_hidden_and_logits(encoder, ids, mask)
        mean_p, max_p = _pool(hidden, mask)
        return hidden, mean_p, max_p, logits

    @torch.no_grad()
    def _embed_seqs(self, seqs: list[str], esm_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean_p, max_p, _ = self._embed_hidden(seqs, esm_name)
        return mean_p, max_p

    @torch.no_grad()
    def _embed_pairs(
        self, refs: list[str], vars_: list[str], esm_name: str,
    ) -> tuple[torch.Tensor, ...]:
        nuc = torch.stack([
            extract_nucleotide_features(r, v) for r, v in zip(refs, vars_)
        ]).to(self.device)
        _, mean_p, max_p, _ = self._embed_hidden(refs + vars_, esm_name)
        n = len(refs)
        return mean_p[:n], max_p[:n], mean_p[n:], max_p[n:], nuc, None, None

    def _mlof_radius(self) -> int:
        net = self.ensembles["mlof"][0]
        if net.site_compare is not None:
            return int(net.site_compare.radius)
        return SITE_RADIUS

    @torch.no_grad()
    def _mlof_ensemble(self, refs: list[str], vars_: list[str], esm_name: str) -> dict[str, np.ndarray]:
        """Score every same-length missense; report max (damage) and mean."""
        n = len(refs)
        empty = {
            "mean": np.full(n, np.nan),
            "sd": np.full(n, np.nan),
            "n_sites": np.zeros(n, dtype=np.int32),
            "avg": np.full(n, np.nan),
            "ref_mean": None,
        }
        hidden, mean_p, max_p, logits = self._embed_hidden(refs + vars_, esm_name, masked_lm=True)
        h_ref, h_var = hidden[:n], hidden[n:]
        logit_ref = logits[:n] if logits is not None else None
        rows: list[tuple[int, int]] = []
        for i, (ref, var) in enumerate(zip(refs, vars_)):
            sites = missense_sites(ref, var)
            if not sites:
                idx = aligned_site_index(ref, var)
                if idx >= 0:
                    sites = [idx]
            rows.extend((i, s) for s in sites)
        if not rows:
            empty["ref_mean"] = mean_p[:n]
            return empty
        radius = self._mlof_radius()
        rec_idx = [i for i, _ in rows]
        seqs_ref = [refs[i] for i, _ in rows]
        seqs_var = [vars_[i] for i, _ in rows]
        centers = [s for _, s in rows]
        site_ref = gather_site_windows(h_ref[rec_idx], seqs_ref, centers, radius=radius)
        site_var = gather_site_windows(h_var[rec_idx], seqs_var, centers, radius=radius)
        tokenizer, _ = self._get_encoder(esm_name, masked_lm=True)
        aa_ids = aa_token_ids(tokenizer)
        chem_rows = []
        for k, (i, center) in enumerate(rows):
            ref, var = seqs_ref[k], seqs_var[k]
            llr = 0.0
            if logit_ref is not None and 0 <= center < len(ref) and center < len(var):
                tok = token_index(center)
                if 0 <= tok < logit_ref.size(1):
                    llr = site_llr(logit_ref[i, tok].detach().float().cpu().numpy(), ref[center], var[center], aa_ids)
            in_domain = extra = 0.0
            if self.domains is not None:
                spans = self.domains.spans(ref)
                if spans:
                    geom = classify_position(center + 1, spans)
                    if geom == "in_domain":
                        in_domain = 1.0
                    elif geom in {"pre_domain", "linker", "tail"}:
                        extra = 1.0
            chem_rows.append(site_chem_at(
                ref, var, center, llr=llr, in_domain=in_domain, extra_domain=extra,
            ))
        chem = torch.stack(chem_rows).to(self.device)
        dummy = torch.zeros(len(rows), dtype=mean_p.dtype, device=self.device)
        dummy_vec = dummy[:, None].expand(-1, mean_p.size(1))
        dummy_nuc = torch.zeros(len(rows), 12, device=self.device)
        parts = []
        for net in self.ensembles["mlof"]:
            raw = net.forward_from_cache(
                dummy_vec, dummy_vec, dummy_vec, dummy_vec, dummy_nuc,
                site_ref=site_ref, site_var=site_var, site_chem=chem,
            )
            parts.append(net.probability(raw).float().cpu())
        stacked = torch.stack(parts, dim=0)
        site_mean = stacked.mean(0).numpy()
        site_sd = stacked.std(0).numpy()
        out_max = np.full(n, np.nan)
        out_sd = np.full(n, np.nan)
        out_n = np.zeros(n, dtype=np.int32)
        out_avg = np.full(n, np.nan)
        buckets: dict[int, list[int]] = {}
        for k, (i, _) in enumerate(rows):
            buckets.setdefault(i, []).append(k)
        for i, idxs in buckets.items():
            vals = site_mean[idxs]
            sds = site_sd[idxs]
            j = int(np.argmax(vals))
            out_max[i] = float(vals[j])
            out_sd[i] = float(sds[j])
            out_n[i] = len(idxs)
            out_avg[i] = float(vals.mean())
        return {
            "mean": out_max,
            "sd": out_sd,
            "n_sites": out_n,
            "avg": out_avg,
            "ref_mean": mean_p[:n],
        }

    @torch.no_grad()
    def _ensemble_measure(
        self, task: str, ref_mean, ref_max, var_mean, var_max, nuc,
        site_ref=None, site_var=None, site_chem=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        measures = []
        for net in self.ensembles[task]:
            raw = net.forward_from_cache(
                ref_mean, ref_max, var_mean, var_max, nuc,
                site_ref=site_ref, site_var=site_var, site_chem=site_chem,
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
            if self.domains is not None:
                self.domains.ensure(refs)
            if pair:
                wreck_flags = []
                for rec, ref in zip(batch, refs):
                    spans = self.domains.spans(ref) if self.domains is not None else ()
                    wreck_flags.append(
                        wreck_grade(rec.ref_protein, rec.var_protein, rec.ref_dna, rec.var_dna, spans)
                    )
            else:
                wreck_flags = [(False, "none", None)] * len(batch)

            pooled: dict[str, tuple] = {}
            gof_tasks = [t for t in tasks if t not in {"lof", "mlof"}]
            mlof_pack: dict[str, np.ndarray] | None = None
            if "lof" in tasks:
                lof_esm = self.task_esm["lof"]
                var_mean, var_max = self._embed_seqs(vars_, lof_esm)
                zeros = torch.zeros(len(batch), 12, device=self.device)
                pooled["lof"] = (var_mean, var_max, var_mean, var_max, zeros, None, None)
            if gof_tasks:
                pair_esm = self.task_esm[gof_tasks[0]]
                pooled["_pair"] = self._embed_pairs(refs, vars_, pair_esm)
            if "mlof" in tasks:
                mlof_pack = self._mlof_ensemble(refs, vars_, self.task_esm["mlof"])

            measures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            family: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
            for task in tasks:
                if task == "mlof":
                    measures[task] = (mlof_pack["mean"], mlof_pack["sd"])
                    query = mlof_pack["ref_mean"]
                    if task in self.galleries and query is not None:
                        family[task] = in_family_flags(query.cpu(), self.galleries[task], self.in_family_threshold)
                    continue
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
                    "mlof_mean": float(mlof_pack["avg"][i]) if mlof_pack is not None else float("nan"),
                    "mlof_n_sites": int(mlof_pack["n_sites"][i]) if mlof_pack is not None else 0,
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
