# PLMLoF

Bacterial variant classifier: **LoF / WT / GoF**. Runs on **Isambard-AI Phase 2** (1 GH200 per job).

Do not run setup, pip, ProteinGym, or training on a login node. Do not put the repo or caches in `$HOME` (100 GiB; filling it can block SSH).

---

## 1. WSL — copy the repo onto project space

Replace `HOST` with the Clifton SSH host you already use.

```bash
clifton auth
clifton ssh-config write

ssh tbea20.b6bh@HOST 'mkdir -p /projects/b6bh/tbea20.b6bh/PLMLoF'

rsync -avz --progress \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'data/raw' \
  --exclude 'data/processed' \
  --exclude 'data/embeddings' \
  --exclude 'data/scripts/data' \
  --exclude 'outputs' \
  --exclude 'wandb' \
  --exclude '*.log' \
  --exclude '*.pt' \
  --exclude 'runpod' \
  /home/theoa/PLMLoF/ \
  tbea20.b6bh@HOST:/projects/b6bh/tbea20.b6bh/PLMLoF/
```

Then log in:

```bash
ssh tbea20.b6bh@HOST
```

---

## 2. Login node — check, then submit

```bash
echo "$HOME"
echo "$PROJECTDIR"
echo "$SCRATCHDIR"
# expect:
# /home/b6bh/tbea20.b6bh
# /projects/b6bh
# /scratch/b6bh/tbea20.b6bh

cd /projects/b6bh/tbea20.b6bh/PLMLoF
ls plmlof/data/dataset.py data/scripts/download_proteingym.py isambard/submit.sh
```

Optional (avoids HuggingFace 429s):

```bash
export HF_TOKEN=hf_YOUR_TOKEN
```

First time — venv + ESM2 download (2 h, 1 GPU):

```bash
bash isambard/submit.sh setup
squeue --me
tail -f "$SCRATCHDIR/plmlof/logs/"plmlof-setup-*.out
```

Wait until the log says `Setup complete.`

Smoke test (30 min):

```bash
bash isambard/submit.sh test
squeue --me
tail -f "$SCRATCHDIR/plmlof/logs/"plmlof-test-*.out
```

Full pipeline after setup has succeeded (24 h, 1 GPU). Check remaining **b6bh** NHR on https://portal.isambard.ac.uk first.

```bash
bash isambard/submit.sh pipeline
squeue --me
tail -f "$SCRATCHDIR/plmlof/logs/"plmlof-pipeline-*.out
```

Checkpoint when finished:

```text
/projects/b6bh/tbea20.b6bh/PLMLoF/outputs/production/checkpoints/model_best.pt
```

---

## 3. Later runs (login node)

Setup is already done. From `/projects/b6bh/tbea20.b6bh/PLMLoF`:

```bash
bash isambard/submit.sh pipeline                  # data → embeddings → train → eval
bash isambard/submit.sh data                      # download + curate only
bash isambard/submit.sh pipeline --train-only     # embeddings already exist
bash isambard/submit.sh pipeline --eval-only
bash isambard/submit.sh pipeline --quick          # 30K samples, short epochs
bash isambard/submit.sh pipeline --scale 900      # 900K samples
```

Cancel your jobs:

```bash
squeue --me
scancel JOBID
```

---

## 4. Predict (compute node)

After a checkpoint exists:

```bash
cd /projects/b6bh/tbea20.b6bh/PLMLoF
srun --nodes=1 --gpus=1 --time=01:00:00 --pty bash --login
source isambard/env.sh
source "$PLMLOF_VENV/bin/activate"
python scripts/predict.py \
  --reference ref_genes.fasta \
  --variants var_genes.fasta \
  --model "$PLMLOF_OUTPUT_DIR/checkpoints/model_best.pt" \
  --output "$PLMLOF_SCRATCH/predictions.tsv" \
  --device cuda
```

---

## Do not

- Copy into `$HOME` or copy a WSL `.venv`
- Run `isambard/setup.sh` or `isambard/pipeline.sh` on the login node (use `submit.sh`)
- Add `--exclusive` to jobs (bills 4 GPUs)

One GPU-hour = **0.25 NHR**. Mix nodes can still take `--gpus=1`.

---

## Local tests (WSL only)

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
