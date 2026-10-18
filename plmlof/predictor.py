"""Inference: comparative LoF (ref↔var ESM2; 35M teacher or 8M distilled)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from plmlof.applicability import in_family_flags, load_gallery
from plmlof.constants import IN_FAMILY_COSINE
from plmlof.data.features import extract_nucleotide_features
from plmlof.domains import DomainIndex
from plmlof.embed import _pool, forward_hidden_and_logits
from plmlof.encoders import esm2_for_task
from plmlof.inference.vcf_handler import VariantRecord, parse_fasta_pairs, parse_protein_fasta
from plmlof.model import TaskNet
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


def _load_net(ckpt_path: Path, device: torch.device) -> TaskNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("task") != "lof":
        raise RuntimeError(f"{ckpt_path} is not a LoF checkpoint (task={ckpt.get('task')!r})")
    cfg = ckpt["model_config"]
    head_hidden, dropout = _head_hparams(ckpt)
    sd = ckpt["state_dict"]
    if any(k.startswith("seq_norm.") for k in sd):
        raise RuntimeError(
            f"{ckpt_path} is an alignment-free LoF checkpoint. "
            "Retrain the LoF head (ref↔var ComparisonModule) and point --model at the new ensemble."
        )
    net = TaskNet(
        hidden_size=cfg["hidden_size"],
        task="lof",
        pool_strategy=cfg.get("pool_strategy", "mean_max"),
        use_cross_attention=cfg.get("use_cross_attention", False),
        head_hidden=head_hidden,
        dropout=dropout,
    )
    net.load_state_dict(sd)
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
    """Comparative LoF. Encoder (35M or distilled 8M) loads lazily from checkpoint meta."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 1024,
        in_family_threshold: float = IN_FAMILY_COSINE,
        domains: str | Path | None = None,
        hmm: str | Path | None = None,
        hmm_cpus: int = 1,
        tasks: tuple[str, ...] | list[str] | None = None,
        lof_only: bool = True,
    ):
        del lof_only  # always LoF-only; kept for call-site compatibility
        if tasks is not None and tuple(tasks) != ("lof",):
            raise ValueError("Predictor only supports tasks=('lof',)")
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.in_family_threshold = in_family_threshold
        self.domains = None
        if domains or hmm:
            self.domains = DomainIndex(
                parquet=Path(domains) if domains else None,
                hmm=Path(hmm) if hmm else None,
                cpus=hmm_cpus,
            )
        self._encoders: dict[str, tuple[object, object]] = {}
        self.ensembles: list[TaskNet] = []
        self.nulls: np.ndarray | None = None
        self.gallery: dict | None = None
        self.esm_name = esm2_for_task("lof")

        tdir = self.model_dir / "lof"
        members = discover_ensemble(tdir) if tdir.exists() else []
        if not members:
            # Also accept a bare model dir that already is the lof/ folder.
            members = discover_ensemble(self.model_dir)
            tdir = self.model_dir
        if not members:
            raise FileNotFoundError(f"No LoF ensemble under {self.model_dir / 'lof'}")
        self.ensembles = [_load_net(p, self.device) for p in members]
        self.esm_name = self._task_encoder_name(tdir, members[0])
        null_path = tdir / "null_scores.pt"
        if null_path.exists():
            self.nulls = torch.load(null_path, map_location="cpu", weights_only=False)["scores"].numpy()
        gal_path = tdir / "train_gallery.pt"
        if gal_path.exists():
            self.gallery = load_gallery(gal_path)
        logger.info(
            "lof: %s members, encoder=%s, null=%s",
            len(self.ensembles), self.esm_name, self.nulls is not None,
        )

    def _task_encoder_name(self, tdir: Path, ckpt_path: Path) -> str:
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
        return esm2_for_task("lof")

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
    def _embed_pairs(
        self, refs: list[str], vars_: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer, encoder = self._get_encoder(self.esm_name)
        nuc = torch.stack([
            extract_nucleotide_features(r, v) for r, v in zip(refs, vars_)
        ]).to(self.device)
        enc = tokenizer(
            refs + vars_, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        hidden, _ = forward_hidden_and_logits(encoder, ids, mask)
        mean_p, max_p = _pool(hidden, mask)
        n = len(refs)
        return mean_p[:n], max_p[:n], mean_p[n:], max_p[n:], nuc

    @torch.no_grad()
    def _ensemble_measure(
        self, ref_mean, ref_max, var_mean, var_max, nuc,
    ) -> tuple[np.ndarray, np.ndarray]:
        measures = []
        for net in self.ensembles:
            raw = net.forward_from_cache(ref_mean, ref_max, var_mean, var_max, nuc)
            measures.append(net.probability(raw).float().cpu())
        stacked = torch.stack(measures, dim=0)
        return stacked.mean(0).numpy(), stacked.std(0).numpy()

    def predict_records(self, records: list[VariantRecord], *, pair: bool = True) -> list[dict]:
        results: list[dict] = []
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

            identical = np.array(
                [r.replace("*", "") == v.replace("*", "") for r, v in zip(refs, vars_)],
                dtype=bool,
            )
            ref_mean, ref_max, var_mean, var_max, nuc = self._embed_pairs(refs, vars_)
            mean, sd = self._ensemble_measure(ref_mean, ref_max, var_mean, var_max, nuc)
            if identical.any():
                mean = mean.copy()
                sd = sd.copy()
                mean[identical] = 0.0
                sd[identical] = 0.0

            if self.gallery is not None:
                in_fam, nearest, sim = in_family_flags(
                    ref_mean.cpu(), self.gallery, self.in_family_threshold,
                )
            else:
                in_fam = np.ones(len(batch), dtype=bool)
                nearest = [""] * len(batch)
                sim = np.ones(len(batch))

            if self.nulls is not None:
                p = empirical_p(mean, self.nulls)
                p = np.where(in_fam, p, np.nan)
                if identical.any():
                    p = np.where(identical, 1.0, p)
            else:
                p = np.full(len(batch), np.nan)
            q = np.full_like(p, np.nan)
            finite = np.isfinite(p)
            if finite.any():
                q[finite] = benjamini_hochberg(p[finite])

            for i, rec in enumerate(batch):
                structural, kind, prior = wreck_flags[i]
                results.append({
                    "gene": rec.gene,
                    "wreck": structural,
                    "wreck_kind": kind,
                    "wreck_prior": float(prior) if prior is not None else float("nan"),
                    "lof_score": float(mean[i]),
                    "lof_sd": float(sd[i]),
                    "lof_p": float(p[i]),
                    "lof_q": float(q[i]),
                    "in_family": bool(in_fam[i]),
                    "nearest_train_gene": nearest[i],
                    "ref_cosine": float(sim[i]),
                })
        return results

    def predict_fasta(self, reference: str | Path, variants: str | Path) -> list[dict]:
        records = parse_fasta_pairs(reference, variants)
        return self.predict_records(records, pair=True)

    def predict_proteins(self, proteins: str | Path) -> list[dict]:
        """Alignment-free LoF: one protein FASTA, no reference pair."""
        records = parse_protein_fasta(proteins)
        return self.predict_records(records, pair=False)
