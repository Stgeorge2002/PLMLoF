#!/usr/bin/env bash
# Prepare PLMLoF training tables on a LAPTOP (or any machine with network).
# Do not run this on an Isambard login node. Do not run it inside the GPU pipeline.
#
# Usage (from repo root):
#   bash scripts/prepare_local.sh
#   bash scripts/prepare_local.sh --skip-genomes   # if GBFFs / synthetic already exist
#   bash scripts/prepare_local.sh --gbff-dir /path/to/gbff
#   bash scripts/prepare_local.sh --with-pfam      # download Pfam-A and grade wrecks by domain
#   bash scripts/prepare_local.sh --skip-genomes --tasks growth_gof
#
# Then copy task tables to the cluster clone:
#   rsync -avP data/processed/{lof,mlof,growth_gof,amr_gof} HOST:/projects/b6bh/$USER/PLMLoF/data/processed/

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_GENOMES=0
WITH_PFAM=0
GBFF_DIR="${ROOT}/data/raw/refseq_gbff"
HMM_PATH="${ROOT}/data/raw/pfam/Pfam-A.hmm.gz"
DOMAINS_PATH="${ROOT}/data/processed/pfam_domains.parquet"
N_GENOMES=150
CPUS="${CPUS:-20}"
TABLE_TASKS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-genomes) SKIP_GENOMES=1; shift ;;
        --with-pfam) WITH_PFAM=1; shift ;;
        --gbff-dir) GBFF_DIR="$2"; shift 2 ;;
        --hmm) HMM_PATH="$2"; shift 2 ;;
        --n-genomes) N_GENOMES="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --tasks) TABLE_TASKS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import pandas, pyarrow, Bio" 2>/dev/null; then
    echo "Need pandas, pyarrow, biopython. e.g.  pip install pandas pyarrow biopython" >&2
    exit 1
fi

mkdir -p data/raw/proteingym data/processed data/raw/refseq_gbff data/raw/pfam

echo "=== 1/5 ProteinGym substitutions parquet (all taxa, laptop download) ==="
if [[ -f data/processed/proteingym_substitutions.parquet ]]; then
    echo "  exists, skipping download_proteingym.py"
else
    "$PYTHON" data/scripts/download_proteingym.py
fi

echo "=== 2/5 CARD / OF GoF source table ==="
if [[ -f data/processed/gof_growth_amr.parquet ]]; then
    echo "  exists, skipping curate_gof_table.py"
else
    "$PYTHON" data/scripts/curate_gof_table.py
fi

echo "=== 3/5 Bacterial GBFFs ==="
if [[ "$SKIP_GENOMES" -eq 1 ]]; then
    echo "  --skip-genomes set"
elif [[ -z "$(ls -A "$GBFF_DIR" 2>/dev/null | head -1)" ]]; then
    echo "  Downloading ~${N_GENOMES} complete bacterial GBFFs → $GBFF_DIR"
    "$PYTHON" data/scripts/download_bacterial_genomes.py --out "$GBFF_DIR" --n "$N_GENOMES"
else
    echo "  GBFFs already in $GBFF_DIR"
fi

echo "=== 4/5 Pfam domains (optional) + synthetic LoF ==="
if [[ -f data/processed/synthetic_lof.parquet ]]; then
    echo "  synthetic_lof.parquet exists, skipping generation"
    echo "  delete it to rebuild with domain-graded priors"
elif [[ "$SKIP_GENOMES" -eq 1 && -z "$(ls -A "$GBFF_DIR" 2>/dev/null | head -1)" ]]; then
    echo "  --skip-genomes set and no GBFFs — LoF wreck channel will be empty"
else
    if [[ "$WITH_PFAM" -eq 1 && ! -f "$DOMAINS_PATH" ]]; then
        if [[ ! -f "$HMM_PATH" ]]; then
            echo "  Downloading Pfam-A.hmm.gz (~1 GB) → $HMM_PATH"
            "$PYTHON" data/scripts/download_pfam.py --out "$HMM_PATH"
        fi
        if "$PYTHON" -c "import pyhmmer" 2>/dev/null || command -v hmmscan >/dev/null 2>&1; then
            echo "  Annotating unique CDS with HMMER/Pfam"
            "$PYTHON" data/scripts/annotate_domains.py \
                --gbff-dir "$GBFF_DIR" \
                --hmm "$HMM_PATH" \
                --out "$DOMAINS_PATH" \
                --cpus "$CPUS" \
                --max-proteins 25000
        else
            echo "  pyhmmer/hmmscan missing — last 10% of each ORF is the tail proxy" >&2
            echo "  pip install pyhmmer   # or install HMMER 3" >&2
        fi
    fi
    GEN_ARGS=(--gbff-dir "$GBFF_DIR" --out data/processed/synthetic_lof.parquet)
    if [[ -f "$DOMAINS_PATH" ]]; then
        GEN_ARGS+=(--domains "$DOMAINS_PATH")
        echo "  Grading wrecks with $DOMAINS_PATH"
    else
        echo "  No Pfam map; last 10% of each ORF is the tail proxy (not every stop = 1.0)"
    fi
    "$PYTHON" data/scripts/generate_synthetic_lof.py "${GEN_ARGS[@]}"
fi

echo "=== 5/5 Build train/val/test/null tables ==="
TABLE_ARGS=(--processed data/processed --out data/processed)
if [[ -n "$TABLE_TASKS" ]]; then
    # shellcheck disable=SC2206
    TABLE_ARGS+=(--tasks $TABLE_TASKS)
fi
"$PYTHON" data/scripts/build_tables.py "${TABLE_ARGS[@]}"

echo ""
echo "Local tables are in data/processed/{lof,mlof,growth_gof,amr_gof}/"
echo "Copy those directories to Isambard, then:  bash isambard/submit.sh pipeline"
