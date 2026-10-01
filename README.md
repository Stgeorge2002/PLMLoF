# PLMLoF v2

Bacterial **LoF score** for protein alleles (FASTA in), plus two rare flags (growth-fitness, resistance SNP). Not a 3-class LoF/WT/GoF softmax. Not Snippy. Not genome-wide GoF.

Clear wrecks (stop / frameshift / large indel) are a **rule**. HMMER grades how hard that rule is: early or in-domain = sure LoF (~1.0); a stop/frameshift strictly after the last Pfam hit is a C-terminal tail (~0.4), not a synthetic null. The network scores **missense** (in-domain vs extra-domain is the other contrast). Each head also emits ensemble ± and an empirical p-value vs a WT-like null.

**Data is downloaded on your laptop. Isambard only embeds, trains, and evaluates.**

---

## 1. Laptop — build training tables (no GPU)

From the repo root (needs `pandas`, `pyarrow`, `biopython`; GPU is not required):

```bash
bash scripts/prepare_v2_local.sh
```

This:

1. Downloads ProteinGym Prokaryote → `data/processed/proteingym_bacterial.parquet`
2. Builds CARD/OF GoF source → `data/processed/gof_growth_amr.parquet`
3. Downloads ~150 complete bacterial GBFFs and writes synthetic wrecks → `data/processed/synthetic_lof.parquet`
4. Writes **protein-held-out / species-held-out / family-held-out** tables under `data/processed/v2/`

To grade synthetic wrecks with Pfam coordinates (~1 GB HMM download, needs `pyhmmer` or `hmmscan`):

```bash
pip install pyhmmer
bash scripts/prepare_v2_local.sh --with-pfam
```

Without Pfam, the last 10% of each ORF is the tail proxy so C-terminal nonsense is not labelled `lof_score = 1.0`. Delete `data/processed/synthetic_lof.parquet` to rebuild after adding a domain map.

If GBFFs already exist:

```bash
bash scripts/prepare_v2_local.sh --gbff-dir /path/to/gbff
# or skip genome download+synthetic:
bash scripts/prepare_v2_local.sh --skip-genomes
```

Copy tables to the cluster (parquet is gitignored):

```bash
rsync -avP data/processed/v2/ HOST:/projects/b6bh/tbea20.b6bh/PLMLoF/data/processed/v2/
```

Do **not** run `prepare_v2_local.sh` on an Isambard login node.

---

## 2. Isambard — setup once, then train

Clone on **project space**, never `$HOME`. Connecting and cloning cost 0 NHR.

```bash
cd /projects/b6bh/tbea20.b6bh
git clone https://github.com/Stgeorge2002/PLMLoF.git
cd PLMLoF
```

Cheap GPU check (ESM2-8M only, no v2 data, ≤ ~0.08 NHR):

```bash
bash isambard/submit.sh smoke
```

Full venv + ESM2-650M (after smoke):

```bash
bash isambard/submit.sh setup
```

v2 pipeline (1 GH200, up to 24 h). **Fails immediately if `data/processed/v2/` is missing** — it will not download ProteinGym.

```bash
bash isambard/submit.sh pipeline
squeue --me
tail -f "$SCRATCHDIR/plmlof/logs/"plmlof-pipeline-*.out
```

Checkpoints:

```text
$PLMLOF_ROOT/outputs/v2/{lof,growth_gof,amr_gof}/seed*/checkpoints/model_best.pt
```

Other job flags:

```bash
bash isambard/submit.sh pipeline --train-only
bash isambard/submit.sh pipeline --eval-only
bash isambard/submit.sh pipeline --task lof
bash isambard/submit.sh embed
```

One GPU-hour = **0.25 NHR**. Do not add `--exclusive`.

---

## 3. Predict (compute node)

Paired FASTA (protein or CDS). Panaroo alleles are the intended input.

```bash
srun --nodes=1 --gpus=1 --time=01:00:00 --pty bash --login
source isambard/env.sh
source "$PLMLOF_VENV/bin/activate"
python scripts/predict.py \
  --model "$PLMLOF_V2_OUTPUT_DIR" \
  --reference ref.fasta \
  --variants var.fasta \
  --output predictions.tsv \
  --device cuda
```

Columns include `lof_score`, `lof_sd`, `lof_p`, `lof_q`, `in_family`, wreck flags, and growth/AMR GoF probabilities + calls.

Held-out Dewachter exam (never in training):

```bash
python scripts/evaluate_dewachter.py \
  --model-dir "$PLMLOF_V2_OUTPUT_DIR" \
  --reference /path/to/dewachter/ref.fasta \
  --variants  /path/to/dewachter/var.fasta \
  --scores    /path/to/dewachter/labels.tsv \
  --device cuda
```

---

## 4. What the three heads are

| Head | Target | Call |
|------|--------|------|
| LoF | `lof_score` ∈ [0, 1] (early/in-domain wrecks=1, in-domain missense=0.7, tail wreck≈0.4, WT=0) | daily score |
| Growth GoF | P(z ≥ +2 on OrganismalFitness growth) | only if p≥0.90 **and** in-family **and** empirical p<0.05 |
| AMR GoF | P(CARD-like resistance SNP) | same conservative rule |

GB1 binding and Tsuboyama stability are **not** training labels.

---

## Local tests (WSL)

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Do not

- Download ProteinGym / CARD / RefSeq inside `isambard/pipeline.sh`
- Copy the repo or HuggingFace cache into `$HOME`
- Run `setup.sh` / `pipeline.sh` on a login node
- Train on Dewachter
- Treat empirical p as P(the protein is dead)
- Score Panaroo serotype swaps or missing plasmids as missense LoF
