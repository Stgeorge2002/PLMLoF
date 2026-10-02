"""Which frozen ESM2 each head is embedded with."""

from __future__ import annotations

import json
from pathlib import Path

ESM2_LOF = "facebook/esm2_t12_35M_UR50D"
ESM2_PAIR = "facebook/esm2_t33_650M_UR50D"

HIDDEN_SIZE = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
    "facebook/esm2_t48_15B_UR50D": 5120,
}


def esm2_for_task(task: str, model_cfg: dict | None = None) -> str:
    cfg = model_cfg or {}
    if task == "lof":
        return str(cfg.get("esm2_model_name_lof") or ESM2_LOF)
    return str(cfg.get("esm2_model_name") or ESM2_PAIR)


def write_encoder_meta(task_dir: Path, esm2_model_name: str, hidden_size: int, **extra) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {"esm2_model_name": esm2_model_name, "hidden_size": int(hidden_size), **extra}
    (task_dir / "encoder.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_encoder_meta(task_dir: Path) -> dict | None:
    path = Path(task_dir) / "encoder.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
