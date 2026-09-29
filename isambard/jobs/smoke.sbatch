#!/bin/bash
# Cheap GPU check: ESM2-8M + a few synthetic samples. No ProteinGym, no 650M.
# Hard cap 20 min so this cannot hold a GH200 for a long allocation.
# Max bill: 20 min × 0.25 NHR/h ≈ 0.083 NHR (usually much less if it finishes early).

#SBATCH --job-name=plmlof-smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source isambard/env.sh

# A venv that only has torch (failed mid-setup) must not skip the rest of install.
if [[ ! -x "${PLMLOF_VENV}/bin/python" ]] \
    || ! "${PLMLOF_VENV}/bin/python" -c "import yaml, numpy, plmlof, torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Tiny setup (PyTorch cu128 + PLMLoF deps + ESM2-8M)..."
    bash isambard/setup.sh --tiny
fi

bash isambard/pipeline.sh --smoke
