"""Optional weight-only INT8 / FP8 for the ESM2 encoder at predict time."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def apply_weight_only_quantize(encoder: torch.nn.Module, scheme: str) -> torch.nn.Module:
    """Quantize ESM2 weights in-place when backends are available.

    ``int8``: torchao int8_weight_only, else bitsandbytes load path (caller).
    ``fp8``: torchao float8_weight_only when present.
    """
    if scheme not in ("int8", "fp8"):
        raise ValueError(f"unknown quantize scheme: {scheme!r}")

    try:
        from torchao.quantization import quantize_  # type: ignore
    except ImportError:
        logger.warning("torchao not installed; cannot apply %s weight-only quantize", scheme)
        return encoder

    try:
        if scheme == "int8":
            from torchao.quantization import int8_weight_only  # type: ignore

            quantize_(encoder, int8_weight_only())
            logger.info("Applied torchao int8_weight_only to ESM2")
            return encoder
        from torchao.quantization import float8_weight_only  # type: ignore

        quantize_(encoder, float8_weight_only())
        logger.info("Applied torchao float8_weight_only to ESM2")
        return encoder
    except Exception as exc:
        logger.warning("%s weight-only quantize failed (%s); using full precision", scheme, exc)
        return encoder


def load_encoder_int8_bnb(esm_name: str, device: torch.device):
    """Load ESM2 with bitsandbytes 8-bit weights (optional dependency)."""
    from transformers import AutoModel, BitsAndBytesConfig

    qcfg = BitsAndBytesConfig(load_in_8bit=True)
    if device.type == "cuda":
        device_map = {"": int(device.index or 0)}
    else:
        device_map = "cpu"
    return AutoModel.from_pretrained(esm_name, quantization_config=qcfg, device_map=device_map)
