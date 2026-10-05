#!/usr/bin/env bash
# Submit PLMLoF jobs from an Isambard-AI login node.
# Run this from the clone on $PROJECTDIR — never from $HOME.
#
#   bash isambard/submit.sh smoke            # cheap GPU check (ESM2-8M only)
#   bash isambard/submit.sh setup
#   bash isambard/submit.sh pipeline         # embed + train + eval (tables must already be on disk)
#   bash isambard/submit.sh pipeline --train-only
#   bash isambard/submit.sh pipeline --eval-only
#   bash isambard/submit.sh pipeline --sweep     # LoF+MDG head ablations, no GoF
#   bash isambard/submit.sh embed
#   bash isambard/submit.sh all              # setup, then pipeline after setup succeeds
#   bash isambard/submit.sh test             # smoke if venv already exists
#
# Training data is NOT downloaded here. On a laptop:
#   bash scripts/prepare_local.sh
#   rsync -avP data/processed/{lof,mlof,growth_gof,amr_gof} HOST:$PROJECTDIR/$USER/PLMLoF/data/processed/

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit.sh is for the login node. Inside a job, call setup.sh / pipeline.sh directly." >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=env.sh
source "${ROOT}/isambard/env.sh"

mkdir -p "$PLMLOF_LOG_DIR"

ACTION="${1:-pipeline}"
shift || true

SBATCH_OUT=(--output="${PLMLOF_LOG_DIR}/%x-%j.out" --error="${PLMLOF_LOG_DIR}/%x-%j.err")

submit_one() {
    local script="$1"
    shift
    sbatch "${SBATCH_OUT[@]}" "${ROOT}/isambard/jobs/${script}" "$@"
}

case "$ACTION" in
    smoke)
        submit_one smoke.sbatch
        echo "Submitted smoke (max 20 min, 1 GPU). Env check only — not training."
        ;;
    setup)
        submit_one setup.sbatch
        ;;
    pipeline)
        submit_one pipeline.sbatch "$@"
        ;;
    sweep)
        submit_one pipeline.sbatch --sweep
        ;;
    embed)
        submit_one pipeline.sbatch --embed-only
        ;;
    data)
        echo "Training data is prepared on a laptop, not on Isambard." >&2
        echo "  bash scripts/prepare_local.sh" >&2
        echo "  rsync -avP data/processed/{lof,mlof,growth_gof,amr_gof} HOST:\$PROJECTDIR/\$USER/PLMLoF/data/processed/" >&2
        exit 1
        ;;
    test)
        submit_one test.sbatch
        ;;
    all)
        setup_id="$(sbatch "${SBATCH_OUT[@]}" --parsable "${ROOT}/isambard/jobs/setup.sbatch")"
        echo "Submitted setup as ${setup_id}"
        sbatch "${SBATCH_OUT[@]}" --dependency=afterok:"${setup_id}" \
            "${ROOT}/isambard/jobs/pipeline.sbatch" "$@"
        ;;
    help|-h|--help)
        sed -n '2,20p' "$0"
        ;;
    *)
        echo "Unknown action: $ACTION (expected smoke|setup|pipeline|sweep|embed|test|all)" >&2
        exit 1
        ;;
esac
