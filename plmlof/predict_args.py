"""Shared argparse / Predictor kwargs for predict CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_predictor_speed_args(p: argparse.ArgumentParser) -> None:
    """Flags for production LoF predict (ensemble trim, compile, quantize, rules)."""
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="ESM2 unique-seq batch size (raise until GPU memory is tight; default 256)",
    )
    p.add_argument(
        "--max-ensemble-members", type=int, default=1,
        help="Use first N ensemble seeds (1–2 for production; keep 5 for teacher eval). Default 1.",
    )
    p.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="torch.compile ESM2 encoder once at load (default: on for CUDA predict)",
    )
    p.add_argument(
        "--quantize", choices=["none", "int8", "fp8"], default="none",
        help="Weight-only quantize ESM2 (validate vs FP16 before production use)",
    )
    p.add_argument(
        "--amp", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
        help="Autocast dtype for ESM2 activations",
    )
    p.add_argument(
        "--rule-short-circuit", action=argparse.BooleanOptionalAction, default=True,
        help="Skip ESM2/head for identity (score 0) and sure wrecks (use wreck_prior). "
        "Tail/mid-confidence alleles still run the model. Default: on.",
    )
    p.add_argument(
        "--sure-prior-min", type=float, default=None,
        help="Minimum wreck_prior to short-circuit sure wrecks (default: domains.SURE_MIN=0.80)",
    )


def predictor_kwargs_from_args(args: argparse.Namespace) -> dict:
    """Build Predictor(...) kwargs from parsed CLI args (speed-related only)."""
    quant = None if getattr(args, "quantize", "none") in (None, "none") else args.quantize
    kwargs: dict = {
        "batch_size": int(args.batch_size),
        "max_ensemble_members": int(args.max_ensemble_members),
        "compile_encoder": bool(getattr(args, "compile", True)),
        "quantize": quant,
        "amp": getattr(args, "amp", "auto"),
        "rule_short_circuit": bool(getattr(args, "rule_short_circuit", True)),
        "lof_only": True,
    }
    if getattr(args, "sure_prior_min", None) is not None:
        kwargs["sure_prior_min"] = float(args.sure_prior_min)
    if getattr(args, "domains", None):
        kwargs["domains"] = args.domains
    if getattr(args, "hmm", None):
        kwargs["hmm"] = args.hmm
    if getattr(args, "hmm_cpus", None) is not None:
        kwargs["hmm_cpus"] = int(args.hmm_cpus)
    return kwargs


def resolve_model_dir(model: str | Path) -> Path:
    """Accept a checkpoint file or an ensemble / best-models directory."""
    path = Path(model)
    if path.is_dir():
        return path
    # .../lof/seed0/checkpoints/model_best.pt → ensemble root
    return path.parent.parent.parent
