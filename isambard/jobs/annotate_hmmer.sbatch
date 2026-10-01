#!/bin/bash
# Isambard 3 ONLY — CPU Grace node. Do not submit this on Isambard-AI (that bills GPUs).
# Do not run on a login node. Do not add --exclusive.
#
# 20 hmmscan processes × 2 threads = 40 cores on one Grace node.
#
# Submit from the clone on $PROJECTDIR:
#   sbatch isambard/jobs/annotate_hmmer.sbatch

#SBATCH --job-name=plmlof-hmmer
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit with sbatch on Isambard 3. Do not run this on a login node." >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:?}"

if [[ -z "${PROJECTDIR:-}" ]]; then
    echo "ERROR: PROJECTDIR is unset. Are you on Isambard 3?" >&2
    exit 1
fi

case "${PWD}" in
    "${HOME}"|"${HOME}"/*)
        echo "ERROR: submit from a clone under \$PROJECTDIR, not \$HOME." >&2
        exit 1
        ;;
esac

WORKDIR="${SCRATCHDIR:-/tmp}/plmlof-hmmer/${SLURM_JOB_ID}"
mkdir -p "$WORKDIR" data/processed data/raw/pfam
export TMPDIR="$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

echo "Host=$(hostname)  job=${SLURM_JOB_ID}  cpus=${SLURM_CPUS_PER_TASK:-}"
echo "Repo=$PWD"
echo "Work=$WORKDIR"

if ! command -v hmmscan >/dev/null 2>&1; then
    module spider HMMER 2>/dev/null || true
    module load HMMER 2>/dev/null || module load hmmer 2>/dev/null || true
fi
if ! command -v hmmscan >/dev/null 2>&1; then
    echo "ERROR: hmmscan not on PATH. On the compute node run:  module spider HMMER" >&2
    echo "       then add the matching  module load ...  line to this script." >&2
    exit 1
fi
echo "hmmscan=$(command -v hmmscan)"
hmmscan -h | head -n 3 || true

VENV="${PROJECTDIR}/${USER}/plmlof-cpu-venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "Creating CPU venv at $VENV (pandas/pyarrow only, no torch)"
    python3 -m venv "$VENV"
    "${VENV}/bin/pip" install --upgrade pip
    "${VENV}/bin/pip" install pandas pyarrow biopython
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

FASTA="${PLMLOF_HMMER_FASTA:-data/processed/unique_wt.faa}"
HMM="${PLMLOF_HMMER_HMM:-data/raw/pfam/Pfam-A.hmm.gz}"
OUT="${PLMLOF_HMMER_OUT:-data/processed/pfam_domains.parquet}"

if [[ ! -f "$FASTA" ]]; then
    echo "ERROR: missing $FASTA — copy unique_wt.faa from the laptop (do not parse GBFFs here)." >&2
    exit 1
fi
if [[ ! -f "$HMM" ]]; then
    echo "ERROR: missing $HMM — copy Pfam-A.hmm.gz from the laptop. Do not download on the login node." >&2
    exit 1
fi

python data/scripts/annotate_domains.py \
    --fasta "$FASTA" \
    --hmm "$HMM" \
    --out "$OUT" \
    --shards 20 \
    --cpus-per-shard 2 \
    --work-dir "$WORKDIR"

echo "Done. $OUT"
ls -lh "$OUT"
