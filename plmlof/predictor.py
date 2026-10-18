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
from plmlof.domains import SURE_MIN, DomainIndex
from plmlof.embed import embed_unique_sequences, resolve_amp_dtype
from plmlof.encoders import esm2_for_task
from plmlof.inference.vcf_handler import VariantRecord, parse_fasta_pairs, parse_protein_fasta
from plmlof.model import TaskNet
from plmlof.quantize_encoder import apply_weight_only_quantize, load_encoder_int8_bnb
from plmlof.stats import benjamini_hochberg, empirical_p
from plmlof.wreck import wreck_grade

logger = logging.getLogger(__name__)


def unique_in_order(sequences: list[str]) -> list[str]:
    """Deduplicate while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out


def _head_hparams(ckpt: dict) -> tuple[int, float]:
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
    """Comparative LoF with production speed options.

    * Unique-sequence embed cache (scatter into pairs)
    * Length-sorted ESM2 batches
    * Optional ``torch.compile`` + larger batches
    * Optional INT8 (bitsandbytes) / FP8 autocast
    * Optional rule short-circuit (identity / sure wrecks skip the head)
    * ``max_ensemble_members`` to ship 1–2 seeds in production
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        batch_size: int = 128,
        max_seq_length: int = 1024,
        in_family_threshold: float = IN_FAMILY_COSINE,
        domains: str | Path | None = None,
        hmm: str | Path | None = None,
        hmm_cpus: int = 1,
        tasks: tuple[str, ...] | list[str] | None = None,
        lof_only: bool = True,
        max_ensemble_members: int | None = None,
        compile_encoder: bool = False,
        quantize: str | None = None,
        amp: str = "auto",
        rule_short_circuit: bool = False,
        sure_prior_min: float = SURE_MIN,
    ):
        del lof_only
        if tasks is not None and tuple(tasks) != ("lof",):
            raise ValueError("Predictor only supports tasks=('lof',)")
        if quantize not in (None, "int8", "fp8"):
            raise ValueError("quantize must be None, 'int8', or 'fp8'")
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_seq_length = max_seq_length
        self.in_family_threshold = in_family_threshold
        self.compile_encoder = bool(compile_encoder)
        self.quantize = quantize
        self.amp = amp  # activation autocast; FP8/INT8 are weight-only on the encoder
        self.rule_short_circuit = bool(rule_short_circuit)
        self.sure_prior_min = float(sure_prior_min)
        self.domains = None
        if domains or hmm:
            self.domains = DomainIndex(
                parquet=Path(domains) if domains else None,
                hmm=Path(hmm) if hmm else None,
                cpus=hmm_cpus,
            )
        self._encoders: dict[str, tuple[object, object]] = {}
        self._seq_pool: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._compile_warmed = False
        self.ensembles: list[TaskNet] = []
        self.nulls: np.ndarray | None = None
        self.gallery: dict | None = None
        self.esm_name = esm2_for_task("lof")

        tdir = self.model_dir / "lof"
        members = discover_ensemble(tdir) if tdir.exists() else []
        if not members:
            members = discover_ensemble(self.model_dir)
            tdir = self.model_dir
        if not members:
            raise FileNotFoundError(f"No LoF ensemble under {self.model_dir / 'lof'}")
        if max_ensemble_members is not None:
            if max_ensemble_members < 1:
                raise ValueError("max_ensemble_members must be >= 1")
            members = members[: int(max_ensemble_members)]
        self.ensembles = [_load_net(p, self.device) for p in members]
        self.esm_name = self._task_encoder_name(tdir, members[0])
        null_path = tdir / "null_scores.pt"
        if null_path.exists():
            self.nulls = torch.load(null_path, map_location="cpu", weights_only=False)["scores"].numpy()
        gal_path = tdir / "train_gallery.pt"
        if gal_path.exists():
            self.gallery = load_gallery(gal_path)
        logger.info(
            "lof: %s members, encoder=%s, null=%s, batch=%s, compile=%s, quantize=%s, short_circuit=%s",
            len(self.ensembles), self.esm_name, self.nulls is not None,
            self.batch_size, self.compile_encoder, self.quantize, self.rule_short_circuit,
        )

    def clear_embed_cache(self) -> None:
        self._seq_pool.clear()

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
        if esm_name in self._encoders:
            return self._encoders[esm_name]

        logger.info("Loading ESM2 %s (quantize=%s compile=%s)", esm_name, self.quantize, self.compile_encoder)
        tokenizer = AutoTokenizer.from_pretrained(esm_name)
        encoder = None
        int8_via_bnb = False

        if self.quantize == "int8":
            try:
                encoder = load_encoder_int8_bnb(esm_name, self.device)
                int8_via_bnb = True
                logger.info("Loaded ESM2 with bitsandbytes INT8")
            except Exception as exc:
                logger.warning(
                    "bitsandbytes INT8 load failed (%s); trying torchao / full precision",
                    exc,
                )

        if encoder is None:
            encoder = AutoModel.from_pretrained(esm_name).to(self.device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            if self.quantize in ("int8", "fp8") and not int8_via_bnb:
                encoder = apply_weight_only_quantize(encoder, self.quantize)

        if self.compile_encoder and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                encoder = torch.compile(encoder)
                logger.info("torch.compile enabled on ESM2 encoder")
            except Exception as exc:
                logger.warning("torch.compile failed (%s); continuing uncompiled", exc)

        self._encoders[esm_name] = (tokenizer, encoder)
        return self._encoders[esm_name]

    @torch.no_grad()
    def _ensure_embedded(self, sequences: list[str]) -> int:
        missing = [s for s in unique_in_order(sequences) if s not in self._seq_pool]
        if not missing:
            return 0
        tokenizer, encoder = self._get_encoder(self.esm_name)
        amp = resolve_amp_dtype(self.device, prefer=self.amp)
        ordered, mean_t, max_t = embed_unique_sequences(
            missing,
            encoder,
            tokenizer,
            self.device,
            batch_size=self.batch_size,
            max_length=self.max_seq_length,
            num_workers=0,
            amp_dtype=amp if amp is not None else "fp32",
        )
        for i, seq in enumerate(ordered):
            self._seq_pool[seq] = (mean_t[i].contiguous(), max_t[i].contiguous())
        logger.info("Embedded %s new sequences (pool=%s)", len(missing), len(self._seq_pool))
        return len(missing)

    def _gather_pooled(self, seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            means = torch.stack([self._seq_pool[s][0] for s in seqs], dim=0)
            maxes = torch.stack([self._seq_pool[s][1] for s in seqs], dim=0)
        except KeyError as exc:
            raise RuntimeError(
                f"sequence missing from embed cache: {exc}. Call _ensure_embedded first."
            ) from exc
        return means.to(self.device), maxes.to(self.device)

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

    def _normalize_pair(self, rec: VariantRecord) -> tuple[str, str]:
        ref = rec.ref_protein.replace("*", "")[: self.max_seq_length]
        var = rec.var_protein.replace("*", "")[: self.max_seq_length]
        return ref, var

    def _classify_pair(
        self,
        rec: VariantRecord,
        ref: str,
        var: str,
        *,
        pair: bool,
    ) -> tuple[tuple[bool, str, float | None], str]:
        """Return (wreck_flags, route) where route is 'identity'|'sure_wreck'|'model'."""
        if not pair:
            return (False, "none", None), "model"
        spans = self.domains.spans(ref) if self.domains is not None else ()
        flags = wreck_grade(rec.ref_protein, rec.var_protein, rec.ref_dna, rec.var_dna, spans)
        structural, _kind, prior = flags
        if ref == var:
            return flags, "identity"
        if (
            self.rule_short_circuit
            and structural
            and prior is not None
            and float(prior) >= self.sure_prior_min
        ):
            return flags, "sure_wreck"
        return flags, "model"

    def predict_records(self, records: list[VariantRecord], *, pair: bool = True) -> list[dict]:
        if not records:
            return []

        refs_all: list[str] = []
        vars_all: list[str] = []
        wreck_all: list[tuple[bool, str, float | None]] = []
        routes: list[str] = []
        for rec in records:
            ref, var = self._normalize_pair(rec)
            refs_all.append(ref)
            vars_all.append(var)

        if self.domains is not None:
            self.domains.ensure(unique_in_order(refs_all))

        for rec, ref, var in zip(records, refs_all, vars_all):
            flags, route = self._classify_pair(rec, ref, var, pair=pair)
            wreck_all.append(flags)
            # Identity short-circuit is always on (score 0); sure wreck needs the flag.
            if route == "identity":
                routes.append("identity")
            elif route == "sure_wreck":
                routes.append("sure_wreck")
            else:
                routes.append("model")

        model_idx = [i for i, r in enumerate(routes) if r == "model"]
        need_seqs = unique_in_order(
            [refs_all[i] for i in model_idx] + [vars_all[i] for i in model_idx]
        )
        # Gallery / in-family for non-model rows still wants ref pooled if gallery exists.
        if self.gallery is not None:
            need_seqs = unique_in_order(
                need_seqs + [refs_all[i] for i, r in enumerate(routes) if r != "model"]
            )

        n_pair_slots = len(records) * 2
        logger.info(
            "Predict %s pairs (%s model / %s identity / %s sure-wreck) from %s unique embeds "
            "(%.1fx vs naive pair ESM2)",
            len(records),
            len(model_idx),
            routes.count("identity"),
            routes.count("sure_wreck"),
            len(need_seqs),
            n_pair_slots / max(len(need_seqs), 1),
        )
        if need_seqs:
            self._ensure_embedded(need_seqs)

        # Warmup compile once so the first timed batch is not cold.
        if (
            self.compile_encoder
            and not self._compile_warmed
            and model_idx
            and self.device.type == "cuda"
        ):
            self._score_indices(records, refs_all, vars_all, wreck_all, [model_idx[0]])
            self._compile_warmed = True

        results = [None] * len(records)  # type: list[dict | None]

        # Fill short-circuit rows.
        for i, route in enumerate(routes):
            if route == "model":
                continue
            structural, kind, prior = wreck_all[i]
            if route == "identity":
                score, sd, p_emp = 0.0, 0.0, 1.0
            else:
                score = float(prior) if prior is not None else 1.0
                sd, p_emp = 0.0, 0.0
            in_fam, nearest, sim = False, "", float("nan")
            if self.gallery is not None and refs_all[i] in self._seq_pool:
                rm, _ = self._gather_pooled([refs_all[i]])
                in_fam_a, nearest_a, sim_a = in_family_flags(
                    rm.cpu(), self.gallery, self.in_family_threshold,
                )
                in_fam, nearest, sim = bool(in_fam_a[0]), nearest_a[0], float(sim_a[0])
            results[i] = {
                "gene": records[i].gene,
                "wreck": structural,
                "wreck_kind": kind,
                "wreck_prior": float(prior) if prior is not None else float("nan"),
                "lof_score": score,
                "lof_sd": sd,
                "lof_p": p_emp,
                "lof_q": p_emp,
                "in_family": in_fam,
                "nearest_train_gene": nearest,
                "ref_cosine": sim,
                "route": route,
            }

        # Score neural rows in length-sorted batches (less host-side scatter churn).
        if model_idx:
            model_idx_sorted = sorted(
                model_idx,
                key=lambda i: max(len(refs_all[i]), len(vars_all[i])),
            )
            for start in range(0, len(model_idx_sorted), self.batch_size):
                chunk = model_idx_sorted[start:start + self.batch_size]
                scored = self._score_indices(records, refs_all, vars_all, wreck_all, chunk)
                for i, row in zip(chunk, scored):
                    row["route"] = "model"
                    results[i] = row

        # BH over finite neural p-values only; short-circuit keeps its fixed p.
        neural_p = np.array([
            results[i]["lof_p"] for i in model_idx  # type: ignore[index]
        ], dtype=np.float64)
        if model_idx and np.isfinite(neural_p).any():
            q = np.full_like(neural_p, np.nan)
            finite = np.isfinite(neural_p)
            q[finite] = benjamini_hochberg(neural_p[finite])
            for j, i in enumerate(model_idx):
                results[i]["lof_q"] = float(q[j])  # type: ignore[index]

        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    def _score_indices(
        self,
        records: list[VariantRecord],
        refs_all: list[str],
        vars_all: list[str],
        wreck_all: list[tuple[bool, str, float | None]],
        indices: list[int],
    ) -> list[dict]:
        refs = [refs_all[i] for i in indices]
        vars_ = [vars_all[i] for i in indices]
        nuc = torch.stack([
            extract_nucleotide_features(r, v) for r, v in zip(refs, vars_)
        ]).to(self.device)
        ref_mean, ref_max = self._gather_pooled(refs)
        var_mean, var_max = self._gather_pooled(vars_)
        mean, sd = self._ensemble_measure(ref_mean, ref_max, var_mean, var_max, nuc)

        if self.gallery is not None:
            in_fam, nearest, sim = in_family_flags(
                ref_mean.cpu(), self.gallery, self.in_family_threshold,
            )
        else:
            in_fam = np.ones(len(indices), dtype=bool)
            nearest = [""] * len(indices)
            sim = np.ones(len(indices))

        if self.nulls is not None:
            p = empirical_p(mean, self.nulls)
            p = np.where(in_fam, p, np.nan)
        else:
            p = np.full(len(indices), np.nan)

        out = []
        for j, i in enumerate(indices):
            structural, kind, prior = wreck_all[i]
            out.append({
                "gene": records[i].gene,
                "wreck": structural,
                "wreck_kind": kind,
                "wreck_prior": float(prior) if prior is not None else float("nan"),
                "lof_score": float(mean[j]),
                "lof_sd": float(sd[j]),
                "lof_p": float(p[j]),
                "lof_q": float("nan"),
                "in_family": bool(in_fam[j]),
                "nearest_train_gene": nearest[j],
                "ref_cosine": float(sim[j]),
            })
        return out

    def predict_fasta(self, reference: str | Path, variants: str | Path) -> list[dict]:
        records = parse_fasta_pairs(reference, variants)
        return self.predict_records(records, pair=True)

    def predict_proteins(self, proteins: str | Path) -> list[dict]:
        records = parse_protein_fasta(proteins)
        return self.predict_records(records, pair=False)
