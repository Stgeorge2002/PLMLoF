#!/usr/bin/env bash
# One-time Isambard-AI setup: aarch64 CUDA PyTorch venv, ESM2 weights, smoke tests.
# Must run on a compute node (sbatch isambard/jobs/setup.sbatch). Login nodes have
# no GPU, 4 GiB RAM, and 1 core — pip/torch/ESM2 will fail or get the session killed.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: setup.sh must run under Slurm on a compute node." >&2
    echo "  From the repo root:  bash isambard/submit.sh setup" >&2
    exit 1
fi

TINY=false
if [[ "${1:-}" == "--tiny" ]]; then
    TINY=true
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: Isambard-AI GH200 nodes are aarch64; this host is $(uname -m)." >&2
    echo "       An x86_64 venv or Docker image from WSL/RunPod will not run here." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
cd "$PLMLOF_ROOT"

if [[ ! -f plmlof/model.py ]]; then
    echo "ERROR: plmlof/model.py is missing. Copy the full working tree." >&2
    exit 1
fi

echo "=============================================="
echo " PLMLoF Isambard-AI setup"
echo "=============================================="
echo "Job:         ${SLURM_JOB_ID}  node=$(hostname)"
echo "Repo:        $PLMLOF_ROOT"
echo "Venv:        $PLMLOF_VENV"
echo "HF_HOME:     $HF_HOME"
echo "TMPDIR:      $TMPDIR"
echo ""

export PATH="${UV_INSTALL_DIR}:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv into $UV_INSTALL_DIR ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi

if [[ -x "${PLMLOF_VENV}/bin/python" ]]; then
    _venv_arch="$("${PLMLOF_VENV}/bin/python" -c "import platform; print(platform.machine())")"
    if [[ "$_venv_arch" != "aarch64" ]]; then
        echo "ERROR: existing venv is ${_venv_arch}, not aarch64. Delete $PLMLOF_VENV and re-run setup." >&2
        exit 1
    fi
    echo "Reusing venv at $PLMLOF_VENV"
else
    echo "Creating Python 3.12 venv ..."
    uv venv --seed --python 3.12 "$PLMLOF_VENV"
fi
unset _venv_arch

# shellcheck disable=SC1091
source "${PLMLOF_VENV}/bin/activate"

# GH200 nodes need the official aarch64 CUDA 12.8 wheel. Mixing PyPI / cu130
# can install 2.14+cu130 which reports cuda=13.0 but is_available() is False.
echo "Installing PyTorch (CUDA 12.8 aarch64, cu128 index only)..."
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch

echo "Installing PLMLoF..."
# uv has no --upgrade-strategy (that is pip-only; unknown flags exit 2).
uv pip install -e ".[dev]" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo "Re-pinning PyTorch to cu128 in case the project install replaced it..."
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch

python - <<'PY'
import torch, sys
print(f"  torch {torch.__version__}  built_cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit(
        "CUDA is not available after the cu128 install. "
        "Delete .venv and resubmit smoke; do not run pipeline."
    )
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  capability: {torch.cuda.get_device_capability(0)}")
print(f"  arch list: {torch.cuda.get_arch_list()}")
PY

echo ""
if [[ "$TINY" == true ]]; then
    echo "Pre-downloading ESM2-8M only (no 650M)..."
    bash "${SCRIPT_DIR}/download_models.sh" --tiny
else
    echo "Pre-downloading ESM2 weights into $HF_HOME ..."
    bash "${SCRIPT_DIR}/download_models.sh"
fi

echo ""
echo "GPU tensor smoke test ..."
python - <<'PY'
import torch
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
y = x @ x.T
print(f"  matmul: OK  ({y.shape})")
PY

echo ""
echo "ESM2-8M forward pass + TaskNet ..."
python - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer
from plmlof.model import TaskNet

device = "cuda"
name = "facebook/esm2_t6_8M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
enc = AutoModel.from_pretrained(name).to(device).eval()
batch = tok(["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERTFTSLPFF"], return_tensors="pt").to(device)
with torch.no_grad():
    hidden = enc(**batch).last_hidden_state
print(f"  ESM2-8M hidden {tuple(hidden.shape)}")
d = hidden.shape[-1]
net = TaskNet(hidden_size=d, task="lof").to(device).eval()
b = 2
mean = torch.randn(b, d, device=device)
mx = torch.randn(b, d, device=device)
nuc = torch.zeros(b, 12, device=device)
with torch.no_grad():
    score = net.forward_from_pooled(mean, mx, mean, mx, nuc)
print(f"  TaskNet LoF {tuple(score.shape)}")
PY

if [[ "$TINY" != true ]]; then
echo ""
echo "torch.compile (precompute uses this by default) ..."
python - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).eval()
try:
    compiled = torch.compile(model)
    enc = tokenizer(
        ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERTFTSLPFF"],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        compiled(**enc)
    print("  torch.compile: OK")
except Exception as e:
    print(f"  WARNING: torch.compile failed ({e})")
    print("  Precompute with --no-compile if this persists.")
PY
fi

echo ""
echo "=============================================="
echo " Setup complete."
if [[ "$TINY" == true ]]; then
    echo " Tiny setup only (ESM2-8M). Full weights: bash isambard/submit.sh setup"
else
    echo " Next:  bash isambard/submit.sh smoke   # cheap GPU check"
    echo " Then:  bash isambard/submit.sh pipeline"
fi
echo "=============================================="
