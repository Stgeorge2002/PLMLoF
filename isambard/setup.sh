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

if [[ ! -f plmlof/data/dataset.py ]]; then
    echo "ERROR: plmlof/data/dataset.py is missing. Copy the full working tree, not a clone that omitted gitignored data/ paths." >&2
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

echo "Installing PyTorch + PLMLoF (CUDA 12.8 aarch64 wheels) ..."
# Single resolve so -e ".[dev]" cannot replace the CUDA wheel with a CPU build from PyPI.
uv pip install \
    torch \
    -e ".[dev]" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match

python - <<'PY'
import torch, sys
print(f"  torch {torch.__version__}  cuda={torch.version.cuda}  arch={torch.cuda.get_arch_list() if torch.cuda.is_available() else 'no-gpu'}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available. Setup must run on a GH200 compute node, not a login node.")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  capability: {torch.cuda.get_device_capability(0)}")
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
echo "ESM2-8M forward pass ..."
python - <<'PY'
import torch
from plmlof.models.plmlof_model import PLMLoFModel
from plmlof.data.dataset import SyntheticPLMLoFDataset
from plmlof.data.collator import PLMLoFCollator
from torch.utils.data import DataLoader

device = "cuda"
model = PLMLoFModel(esm2_model_name="facebook/esm2_t6_8M_UR50D", freeze_esm2=True).to(device)
dataset = SyntheticPLMLoFDataset(num_samples=4)
collator = PLMLoFCollator(tokenizer_name="facebook/esm2_t6_8M_UR50D")
loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
batch = next(iter(loader))
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
with torch.no_grad():
    logits = model(
        ref_input_ids=batch["ref_input_ids"],
        ref_attention_mask=batch["ref_attention_mask"],
        var_input_ids=batch["var_input_ids"],
        var_attention_mask=batch["var_attention_mask"],
        nucleotide_features=batch["nucleotide_features"],
    )
print(f"  logits {tuple(logits.shape)}  preds={logits.argmax(-1).tolist()}")
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
