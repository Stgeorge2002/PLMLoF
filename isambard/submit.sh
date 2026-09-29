#!/usr/bin/env bash
# Submit PLMLoF jobs from an Isambard-AI login node.
# Run this from the clone on $PROJECTDIR — never from $HOME.
#
#   bash isambard/submit.sh smoke            # cheap GPU check (~minutes, ESM2-8M only)
#   bash isambard/submit.sh setup
#   bash isambard/submit.sh pipeline
#   bash isambard/submit.sh all              # setup, then pipeline after setup succeeds
#   bash isambard/submit.sh test             # smoke train if venv already exists
#   bash isambard/submit.sh pipeline --train-only
#   bash isambard/submit.sh data             # pipeline --data-only

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
        echo "Submitted smoke (max 20 min, 1 GPU). This is the cheap check — not the full pipeline."
        ;;
    setup)
        submit_one setup.sbatch
        ;;
    pipeline)
        submit_one pipeline.sbatch "$@"
        ;;
    data)
        submit_one pipeline.sbatch --data-only
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
        sed -n '2,14p' "$0"
        ;;
    *)
        echo "Unknown action: $ACTION (expected smoke|setup|pipeline|data|test|all)" >&2
        exit 1
        ;;
esac
