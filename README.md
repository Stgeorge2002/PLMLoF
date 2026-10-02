# PLMLoF

Bacterial **LoF score** for protein alleles (FASTA in), plus two rare flags (growth-fitness, resistance SNP). Not a 3-class LoF/WT/GoF softmax. Not Snippy. Not genome-wide GoF.

Clear wrecks (stop / frameshift / large indel) are a **rule**. HMMER grades how hard that rule is: early or in-domain = sure LoF (~1.0); a stop/frameshift strictly after the last Pfam hit is a C-terminal tail (~0.4), not a synthetic null. The network scores **missense** (in-domain vs extra-domain is the other contrast). Each head also emits ensemble ± and an empirical p-value vs a WT-like null.

**Data is downloaded on your laptop. Isambard only embeds, trains, and evaluates.**

---

## 1. Laptop — build training tables (no GPU)

From the repo root (needs `pandas`, `pyarrow`, `biopython`; GPU is not required):

```bash
bash scripts/prepare_local.sh
```

This:

1. Downloads ProteinGym substitutions → `data/processed/proteingym_substitutions.parquet`
2. Builds CARD/OF GoF source → `data/processed/gof_growth_amr.parquet`
3. Downloads ~150 complete bacterial GBFFs and writes synthetic wrecks → `data/processed/synthetic_lof.parquet`
4. Writes **protein-held-out / residue-held-out / family-held-out** tables under `data/processed/{lof,mlof,growth_gof,amr_gof}/`

To grade synthetic wrecks with Pfam coordinates (~1 GB HMM download, needs `pyhmmer` or `hmmscan`):

```bash
pip install pyhmmer
bash scripts/prepare_local.sh --with-pfam
```

Without Pfam, the last 10% of each ORF is the tail proxy so C-terminal nonsense is not labelled `lof_score = 1.0`. Delete `data/processed/synthetic_lof.parquet` to rebuild after adding a domain map.

If GBFFs already exist:

```bash
bash scripts/prepare_local.sh --gbff-dir /path/to/gbff
# or skip genome download+synthetic:
bash scripts/prepare_local.sh --skip-genomes
```

Copy tables to the cluster (parquet is gitignored):

```bash
rsync -avP data/processed/{lof,mlof,growth_gof,amr_gof} HOST:/projects/b6bh/tbea20.b6bh/PLMLoF/data/processed/
```

Do **not** run `prepare_local.sh` on an Isambard login node.

---

## 2. Isambard — setup once, then train

Clone on **project space**, never `$HOME`. Connecting and cloning cost 0 NHR.

```bash
cd /projects/b6bh/tbea20.b6bh
git clone https://github.com/Stgeorge2002/PLMLoF.git
cd PLMLoF
```

Cheap GPU check (ESM2-8M + TaskNet, no training data, ≤ ~0.08 NHR):

```bash
bash isambard/submit.sh smoke
```

Full venv + ESM2-35M (LoF) + ESM2-650M (MLoF/GoF):

```bash
bash isambard/submit.sh setup
```

Pipeline (1 GH200, up to 24 h). **Fails immediately if `data/processed/lof/` is missing** — it will not download ProteinGym.

```bash
bash isambard/submit.sh pipeline
squeue --me
tail -f "$SCRATCHDIR/plmlof/logs/"plmlof-pipeline-*.out
```

Checkpoints:

```text
$PLMLOF_ROOT/outputs/{lof,mlof,growth_gof,amr_gof}/seed*/checkpoints/model_best.pt
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
  --model "$PLMLOF_OUTPUT_DIR" \
  --reference ref.fasta \
  --variants var.fasta \
  --output predictions.tsv \
  --device cuda
```

Alignment-free LoF (no reference pair; loads ESM2-35M only):

```bash
python scripts/predict.py --model "$PLMLOF_OUTPUT_DIR" --proteins alleles.faa --device cuda
```

Columns include `lof_score`, `lof_sd`, `lof_p`, `lof_q`, `mlof_score`, `in_family`, wreck flags, and growth/AMR GoF probabilities + calls.

Held-out Dewachter exam (never in training):

```bash
python scripts/evaluate_dewachter.py \
  --model-dir "$PLMLOF_OUTPUT_DIR" \
  --reference /path/to/dewachter/ref.fasta \
  --variants  /path/to/dewachter/var.fasta \
  --scores    /path/to/dewachter/labels.tsv \
  --device cuda
```

---

## 4. Heads

| Head | Encoder | Target | Call |
|------|---------|--------|------|
| LoF | ESM2-35M | `lof_score` ∈ [0, 1] (early/in-domain wrecks=1, in-domain missense=0.7, tail wreck≈0.4, WT=0) | daily score |
| MLoF | ESM2-650M | missense-damage rank on all ProteinGym substitution genes | score, not a wreck caller |
| Growth GoF | ESM2-650M | P(z ≥ +2 on OrganismalFitness growth) | only if p≥0.90 **and** in-family **and** empirical p<0.05 |
| AMR GoF | ESM2-650M | P(CARD-like resistance SNP) | same conservative rule |

GB1 binding and Tsuboyama stability are **not** GoF training labels; they are MLoF missense-damage signal.

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
