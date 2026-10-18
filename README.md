# PLMLoF

Bacterial **LoF score** for protein alleles (FASTA in). Comparative: the head
contrasts **reference vs allele** pooled ESM2-35M (identity → ~0). Clear wrecks
(stop / frameshift / large indel) are a **rule**; MMseqs×Pfam (or a last-10%
tail proxy) grades how hard that rule is by **domain impact**. Each score also
carries ensemble ± and an empirical p-value vs a WT-like null.

**Data is downloaded on your laptop. Isambard only embeds, trains, and evaluates.**

---

## 1. Laptop — build LoF training tables (no GPU)

From the repo root (needs `pandas`, `pyarrow`, `biopython`; GPU is not required):

```bash
bash scripts/prepare_local.sh
```

This:

1. Downloads ~150 complete bacterial GBFFs and writes synthetic wrecks → `data/processed/synthetic_lof.parquet`
2. Writes residue/species-held-out tables under `data/processed/lof/`

To grade synthetic wrecks with domain coordinates (MMseqs2 × Pfam-A seeds):

```bash
conda install -c bioconda mmseqs2
bash scripts/prepare_local.sh --with-pfam
```

Uses `mmseqs easy-search` against Pfam seed FASTA, then keeps high-confidence
non-overlapping query spans. Exact HMMER/Pfam-GA remains available via
`annotate_domains.py --backend hmmer` if you ever need it.

Without Pfam, the last 10% of each ORF is the tail proxy so C-terminal nonsense
is not labelled `lof_score = 1.0`. Delete `data/processed/synthetic_lof.parquet`
to rebuild after adding a domain map.

If GBFFs already exist:

```bash
bash scripts/prepare_local.sh --gbff-dir /path/to/gbff
# or skip genome download+synthetic:
bash scripts/prepare_local.sh --skip-genomes
```

Copy tables to the cluster (parquet is gitignored):

```bash
rsync -avP data/processed/lof HOST:/projects/b6bh/tbea20.b6bh/PLMLoF/data/processed/
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

Full venv + ESM2-35M (LoF):

```bash
bash isambard/submit.sh setup
```

Pipeline (1 GH200, up to 24 h). **Fails immediately if `data/processed/lof/` is missing**.

```bash
bash isambard/submit.sh pipeline
squeue --me
tail -f "$SCRATCHDIR/plmlof/logs/"plmlof-pipeline-*.out
```

Checkpoints:

```text
$PLMLOF_ROOT/outputs/lof/seed*/checkpoints/model_best.pt
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

### Import (MutCode + Panaroo → FASTA pairs)

Main outbreak import decodes MutCode ``Mutation_Codes`` against Panaroo
``pan_genome_reference.fa`` (``[]`` = exact FASTA copy). LoF-matrix
``Reference_Length`` is annotation metadata only and is not used as the
MutCode reference length.

```bash
python scripts/build_from_lof_matrix.py \
  --lof-matrix N44_ST258/lof_matrix/lof_matrix.tsv \
  --results-dir N44_ST258 \
  --out-dir N44_ST258/plmlof_input
```

Native MutCode tables (with SHA-256 columns) are also accepted:

```bash
python scripts/build_from_lof_matrix.py \
  --mutcode-table mutation_codes.tsv.gz \
  --reference pan_genome_reference.fa \
  --out-dir plmlof_input
```

Then score with ``scripts/run_predict_batch.py --input-dir .../plmlof_input``.

Paired FASTA (protein or CDS) can also be passed directly to ``predict.py``.

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

Alignment-free LoF (no reference pair; still loads ESM2-35M):

```bash
python scripts/predict.py --model "$PLMLOF_OUTPUT_DIR" --proteins alleles.faa --device cuda
```

For the N44 ST258 frozen-head batch, submit `bash isambard/submit.sh predict`
from the login node after setup succeeds. The submission helper sets the repo
as Slurm's submission directory; the job loads its environment from there,
not from the location of Slurm's copied batch script. Git keeps the analysis scripts,
PLMLoF package, runtime configuration, and job scripts, not the outbreak data
or inference assets. The entire `N44_ST258_results/` and `best-models/` trees,
their `plmlof_input.tar*` / `best-models.tar*` transfer archives, and generated
`examples/blaOXA1/` data are ignored. Transfer data and the complete model bundle
(including JSON metadata) separately with rsync; example Python scripts remain
trackable.

Output columns: `lof_score`, `lof_sd`, `lof_p`, `lof_q`, wreck flags, and
`in_family` / nearest-train cosine.

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

## 4. LoF head

| Head | Encoder | Target | Call |
|------|---------|--------|------|
| LoF | ESM2-35M pooled **ref↔var** | wreck `lof_score` ∈ [0, 1] from domain impact; identity ~0 | daily score; `wreck_grade` is the overlay |

After changing tables or the embedder, delete `$PLMLOF_EMB_DIR/lof` so precompute
does not reuse stale caches.

---

## Local tests (WSL)

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Do not

- Download RefSeq / Pfam inside `isambard/pipeline.sh`
- Copy the repo or HuggingFace cache into `$HOME`
- Run `setup.sh` / `pipeline.sh` on a login node
- Train on Dewachter
- Treat empirical p as P(the protein is dead)
- Score Panaroo serotype swaps or missing plasmids as LoF
