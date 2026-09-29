#!/usr/bin/env bash
# Download ESM2 weights into $HF_HOME (project cache). Never ~/.cache.
#
#   bash isambard/download_models.sh           # 8M + 650M
#   bash isambard/download_models.sh --tiny    # 8M only
#   bash isambard/download_models.sh --all     # 8M, 35M, 150M, 650M

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ ! -x "${PLMLOF_VENV}/bin/python" ]]; then
    echo "ERROR: venv not found at $PLMLOF_VENV. Run setup first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${PLMLOF_VENV}/bin/activate"

MODE="${1:---production}"

echo "Downloading ESM2 snapshots to $HF_HOME ..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "  HF_TOKEN is set (authenticated Hub downloads)."
else
    echo "  HF_TOKEN is unset. Shared-IP Hub rate limits may apply; set HF_TOKEN if downloads 429."
fi

python - <<PY
import os
import sys
from huggingface_hub import snapshot_download

mode = "$MODE"
models = {
    "--tiny": ["facebook/esm2_t6_8M_UR50D"],
    "--all": [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
    ],
}.get(mode, ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t33_650M_UR50D"])

token = os.environ.get("HF_TOKEN") or None
cache = os.environ["HF_HUB_CACHE"]

for name in models:
    print(f"  {name}")
    snapshot_download(repo_id=name, cache_dir=cache, token=token)
    print(f"  {name}: OK")
print("Done.")
PY
