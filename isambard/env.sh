# Isambard-AI Phase 2 environment. Source from every job script and from submit.sh.
#
# Storage rules (BriCS):
#   $HOME       100 GiB  — config only. Filling it can block SSH login.
#   $PROJECTDIR 200 TiB  — code, venv, HuggingFace weights, curated data, checkpoints.
#   $SCRATCHDIR   5 TiB  — embeddings, logs, compiler caches, TMPDIR fallback.
#   $LOCALDIR    48 GiB  — node-local tmpfs, wiped at job end. Prefer for TMPDIR.
#
# Do not `set -euo` here — this file is sourced.

if [[ -z "${PROJECTDIR:-}" || -z "${SCRATCHDIR:-}" ]]; then
    if [[ "${USER:-}" == *.* ]]; then
        _proj="${USER##*.}"
        export PROJECTDIR="${PROJECTDIR:-/projects/${_proj}}"
        export SCRATCHDIR="${SCRATCHDIR:-/scratch/${_proj}/${USER}}"
    fi
fi

: "${PROJECTDIR:?PROJECTDIR is not set. This environment is for Isambard-AI.}"
: "${SCRATCHDIR:?SCRATCHDIR is not set. This environment is for Isambard-AI.}"

if [[ ! -d "$PROJECTDIR" ]]; then
    echo "ERROR: PROJECTDIR does not exist: $PROJECTDIR" >&2
    exit 1
fi
if [[ ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR does not exist: $SCRATCHDIR" >&2
    exit 1
fi

# Repo lives on project space — never under $HOME.
export PLMLOF_ROOT="${PLMLOF_ROOT:-${PROJECTDIR}/${USER}/PLMLoF}"

# If this file is sourced from a clone that is already the repo, prefer that
# over the default path (lets you name the directory differently).
_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_detected_root="$(cd "${_env_dir}/.." && pwd)"
if [[ -f "${_detected_root}/pyproject.toml" && -d "${_detected_root}/plmlof" ]]; then
    export PLMLOF_ROOT="${_detected_root}"
fi

case "${PLMLOF_ROOT}" in
    "${HOME}"|"${HOME}"/*)
        echo "ERROR: PLMLOF_ROOT is under \$HOME (${PLMLOF_ROOT})." >&2
        echo "       Clone the repo to \$PROJECTDIR/\$USER/PLMLoF so HuggingFace" >&2
        echo "       weights, the venv, and data cannot fill the 100 GiB home quota." >&2
        exit 1
        ;;
esac

export PLMLOF_VENV="${PLMLOF_VENV:-${PLMLOF_ROOT}/.venv}"
export PLMLOF_CACHE="${PLMLOF_CACHE:-${PROJECTDIR}/${USER}/plmlof-cache}"
export PLMLOF_SCRATCH="${PLMLOF_SCRATCH:-${SCRATCHDIR}/plmlof}"

export PLMLOF_DATA_DIR="${PLMLOF_DATA_DIR:-${PLMLOF_ROOT}/data/processed}"
export PLMLOF_EMB_DIR="${PLMLOF_EMB_DIR:-${PLMLOF_SCRATCH}/embeddings}"
export PLMLOF_OUTPUT_DIR="${PLMLOF_OUTPUT_DIR:-${PLMLOF_ROOT}/outputs/production}"
export PLMLOF_LOG_DIR="${PLMLOF_LOG_DIR:-${PLMLOF_SCRATCH}/logs}"

export PLMLOF_TRAIN_CFG="${PLMLOF_TRAIN_CFG:-${PLMLOF_ROOT}/configs/production_training.yaml}"
export PLMLOF_MODEL_CFG="${PLMLOF_MODEL_CFG:-${PLMLOF_ROOT}/configs/production_model.yaml}"

# HuggingFace / Torch / pip / uv / inductor — never ~/.cache
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PLMLOF_CACHE}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PLMLOF_CACHE}/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PLMLOF_CACHE}/uv}"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-${PLMLOF_CACHE}/uv-bin}"
export HF_HOME="${HF_HOME:-${PLMLOF_CACHE}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${PLMLOF_CACHE}/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PLMLOF_SCRATCH}/torch-inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PLMLOF_SCRATCH}/triton}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${PLMLOF_SCRATCH}/pycache}"

# Node-local tmpfs is wiped at job end; fall back to scratch on login nodes.
if [[ -n "${LOCALDIR:-}" && -d "${LOCALDIR}" && -w "${LOCALDIR}" ]]; then
    export TMPDIR="${TMPDIR:-${LOCALDIR}}"
else
    export TMPDIR="${TMPDIR:-${PLMLOF_SCRATCH}/tmp}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# Optional: export HF_TOKEN before submitting to avoid shared-IP Hub rate limits.
# export HF_TOKEN="hf_..."

mkdir -p \
    "$PLMLOF_CACHE" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$TORCH_HOME" \
    "$XDG_CACHE_HOME" \
    "$PIP_CACHE_DIR" \
    "$UV_CACHE_DIR" \
    "$UV_INSTALL_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$PYTHONPYCACHEPREFIX" \
    "$PLMLOF_EMB_DIR" \
    "$PLMLOF_OUTPUT_DIR" \
    "$PLMLOF_LOG_DIR" \
    "$TMPDIR"

# Refuse to proceed if any cache resolved under $HOME (mis-set override).
for _dir in "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"; do
    case "$_dir" in
        "${HOME}"|"${HOME}"/*)
            echo "ERROR: cache path $_dir is under \$HOME. Unset the override." >&2
            exit 1
            ;;
    esac
done

unset _env_dir _detected_root _proj _dir
