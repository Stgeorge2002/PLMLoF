# Plan: PLMLoF — PLM-based LoF/GoF Variant Classifier for Bacteria

## TL;DR
Build a PyTorch-based variant effect classifier that replaces SNPEff, using a **dual-level architecture**: ESM2 (protein embeddings) as the primary encoder with engineered nucleotide-level features, optionally augmented by a Nucleotide Transformer encoder. Fine-tuned on bacterial LoF/GoF data curated from CARD (AMR/GoF), DEG (LoF), ProteinGym DMS, and synthetic variants. The model takes **reference + variant gene sequences per strain**, classifies each gene as **LoF / Wildtype / GoF** with confidence scores, and provides **position-level attribution** explaining the prediction. Designed for high-throughput inference on massive datasets with pre-cached reference embeddings.

---

## Phase 1: Project Scaffold & Data Pipeline

### Step 1.1 — Project structure
Create the following layout:

```
PLMLoF/
├── README.md
├── pyproject.toml
├── configs/
│   ├── model.yaml              # Model architecture config
│   ├── training.yaml           # Training hyperparameters
│   └── inference.yaml          # Inference settings
├── plmlof/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── esm2_encoder.py     # ESM2 wrapper (frozen + LoRA)
│   │   ├── nt_encoder.py       # Nucleotide Transformer wrapper (Phase 2)
│   │   ├── comparison.py       # Reference vs Variant comparison module
│   │   ├── classifier.py       # Classification head (LoF/WT/GoF)
│   │   └── plmlof_model.py     # Full model combining all components
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Dataset for ref+var pairs
│   │   ├── collator.py         # Dynamic padding collator
│   │   ├── preprocessing.py    # Sequence extraction, translation
│   │   └── features.py         # Engineered nucleotide features
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop with mixed precision
│   │   └── metrics.py          # F1, AUROC, confusion matrix
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Main prediction interface
│   │   ├── vcf_handler.py      # VCF/FASTA input parsing
│   │   ├── reference_cache.py  # Pre-compute & cache ref embeddings
│   │   └── attribution.py      # Integrated gradients explanations
│   └── utils/
│       ├── __init__.py
│       └── sequence_utils.py   # DNA→protein translation, alignment
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Training-ready data
│   └── scripts/
│       ├── download_card.py    # Download CARD AMR mutations
│       ├── download_deg.py     # Download Database of Essential Genes
│       ├── download_proteingym.py # Download ProteinGym DMS data
│       ├── curate_dataset.py   # Merge, label, split datasets
│       └── generate_synthetic.py # Generate synthetic LoF variants
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation & benchmarking
│   ├── predict.py              # Inference entry point (VCF/FASTA)
│   └── benchmark_vs_snpeff.py  # Compare predictions vs SNPEff
└── tests/
    ├── conftest.py             # Fixtures (tiny models, synthetic data)
    ├── test_model.py           # Model forward pass tests
    ├── test_dataset.py         # Data loading tests
    ├── test_features.py        # Nucleotide feature extraction tests
    ├── test_predictor.py       # End-to-end prediction tests
    └── test_attribution.py     # Attribution module tests
```

### Step 1.2 — Dependencies (`pyproject.toml`)
Core dependencies:
- `torch>=2.0` — framework
- `transformers>=4.30` — ESM2 + NT model loading from HuggingFace
- `peft>=0.7` — LoRA adapters for fine-tuning
- `datasets` — HuggingFace datasets
- `biopython` — sequence handling, VCF parsing, translation
- `pysam` — VCF reading
- `PyYAML` — config loading
- `scikit-learn` — metrics
- `captum` — attribution (integrated gradients)
- `accelerate` — mixed precision, multi-GPU
- `wandb` — experiment tracking (optional)

Dev dependencies: `pytest`, `pytest-cov`

### Step 1.3 — Data curation pipeline (*parallel with 1.1*)
Curate training data from 4 sources + synthetic augmentation:

**Source 1: CARD (GoF labels)**
- Download CARD protein variant data (SNP-mediated AMR)
- Extract: reference protein sequence + variant sequence + gene name + species
- Label: **GoF** (resistance-conferring mutations)
- Expected yield: ~5,000+ curated GoF variants across pan-bacterial species
- Script: `data/scripts/download_card.py`

**Source 2: DEG — Database of Essential Genes (LoF labels)**
- Download DEG bacterial essential genes
- For each essential gene: generate disruptive variants (premature stops, frameshifts at random positions)
- Label: **LoF** (disruption of essential gene = loss of function)  
- Expected yield: ~10,000+ synthetic LoF variants from ~3,500 essential genes
- Script: `data/scripts/download_deg.py`

**Source 3: ProteinGym DMS (LoF/WT/GoF by fitness threshold)**
- Download ProteinGym substitution assays for bacterial proteins
- Key bacterial DMS datasets: TEM-1 beta-lactamase (E. coli), DHFR, InfA, and others
- Threshold fitness scores: LoF (< -1σ), WT (within ±1σ), GoF (> +1σ)
- Expected yield: varies per assay, ~50K-200K labeled variants total (subset bacterial)
- Script: `data/scripts/download_proteingym.py`

**Source 4: Synthetic LoF generation**
- Take any bacterial reference protein from UniProt/DEG
- Introduce: premature stop codons, frameshift-simulated scrambled C-terminal, large truncations
- Label: **LoF** (deterministic)
- Purpose: ensure model learns structural disruption patterns
- Script: `data/scripts/generate_synthetic.py`

**Source 5: Wildtype (WT) labels**
- Reference sequences with synonymous mutations or no mutations → WT
- Ensures balanced classes
- Drawn from same genes as LoF/GoF sources

**Curation script** (`data/scripts/curate_dataset.py`):
- Merge all sources into unified format: `(ref_protein_seq, var_protein_seq, ref_dna_seq, var_dna_seq, label, gene, species, source)`
- Split: 80% train / 10% val / 10% test (stratified by species + gene family to avoid leakage)
- Cross-species holdout: hold out 2-3 species entirely for generalization testing
- Save as HuggingFace Dataset or parquet files

---

## Phase 2: Model Architecture

### Step 2.1 — ESM2 Encoder (`plmlof/models/esm2_encoder.py`)
- Load ESM2 from HuggingFace (`facebook/esm2_t33_650M_UR50D` for production, `facebook/esm2_t6_8M_UR50D` for testing)
- Freeze all parameters initially
- Add LoRA adapters (rank=16, target: query/value projection layers) via `peft`
- Extract per-residue embeddings (last hidden state) + [CLS]-equivalent pooled embedding
- Method: `encode(protein_sequence) → (per_residue_emb: [L, D], pooled_emb: [D])`
- Handle variable-length sequences with padding

### Step 2.2 — Nucleotide Feature Extractor (`plmlof/data/features.py`)
Engineered features computed from DNA-level comparison (no neural network needed):
- `is_frameshift`: bool — insertion/deletion length not divisible by 3
- `has_premature_stop`: bool — new stop codon before original stop
- `start_codon_lost`: bool — ATG→non-ATG at start
- `num_missense`: int — count of amino acid changes
- `num_nonsense`: int — count of new stop codons
- `num_synonymous`: int — count of silent mutations
- `truncation_fraction`: float — fraction of protein truncated (0.0 = none, 1.0 = complete)
- `mutation_density`: float — mutations per 100 residues
- `affected_region`: categorical — N-terminal / middle / C-terminal
- Output: feature vector of ~12 dimensions

### Step 2.3 — Comparison Module (`plmlof/models/comparison.py`)
Takes reference and variant embeddings, produces comparison features:
- **Element-wise difference**: `ref_emb - var_emb` (per-residue when aligned, pooled otherwise)
- **Element-wise product**: `ref_emb * var_emb` (interaction features)
- **Cosine similarity per position**: local similarity scores
- **Cross-attention** (optional): variant attends to reference, captures long-range dependency changes
- Pool comparison features → fixed-size vector (mean pool + max pool + [CLS] diff)
- Output: comparison vector `[4*D]` where D is ESM2 hidden dim

### Step 2.4 — Classification Head (`plmlof/models/classifier.py`)
- Input: concatenation of [comparison_vector, nucleotide_features]
- Architecture: `Linear(4*D+12, 256) → ReLU → Dropout(0.3) → Linear(256, 64) → ReLU → Dropout(0.2) → Linear(64, 3)`
- Output: logits for [LoF, WT, GoF]
- Loss: `CrossEntropyLoss` with class weights (handle imbalance) + optional label smoothing

### Step 2.5 — Full Model (`plmlof/models/plmlof_model.py`)
Combines all components:
```
PLMLoFModel:
  __init__(esm2_model_name, freeze_esm2, lora_config, use_nt_encoder=False)
  
  forward(ref_protein_ids, var_protein_ids, nucleotide_features, attention_mask_ref, attention_mask_var):
    ref_emb = esm2_encoder(ref_protein_ids)
    var_emb = esm2_encoder(var_protein_ids)
    comparison = comparison_module(ref_emb, var_emb)
    features = concat(comparison, nucleotide_features)
    logits = classifier(features)
    return logits  # [B, 3]
```

---

## Phase 3: Training Pipeline

### Step 3.1 — Dataset & DataLoader (`plmlof/data/dataset.py`, `collator.py`)
- `PLMLoFDataset`: loads curated parquet, returns `(ref_seq, var_seq, nuc_features, label)`
- `PLMLoFCollator`: tokenizes protein sequences with ESM2 tokenizer, dynamic padding, stacks nucleotide features
- Efficient: tokenize on-the-fly (not pre-tokenized) to save disk
- Support for stratified sampling in DataLoader

### Step 3.2 — Training loop (`plmlof/training/trainer.py`, `scripts/train.py`)
**Stage 1: Classification head only** (ESM2 fully frozen, LoRA inactive)
- Train for 10-20 epochs on full dataset
- LR: 1e-3, AdamW, cosine scheduler with warmup
- Purpose: learn comparison → classification mapping

**Stage 2: LoRA fine-tuning** (unfreeze LoRA adapters)
- Train for 5-10 more epochs
- LR: 1e-4 (lower for pre-trained params)
- Purpose: adapt ESM2 embeddings for bacterial variant domain

**Training features:**
- Mixed precision (fp16/bf16) via `accelerate`
- Gradient accumulation for large effective batch sizes
- Early stopping on validation F1
- Save best checkpoint by macro-F1
- Log to wandb (optional)

### Step 3.3 — Metrics (`plmlof/training/metrics.py`)
- Per-class precision, recall, F1
- Macro F1 (primary metric)
- AUROC (one-vs-rest)
- Confusion matrix
- Cross-species generalization metrics (on held-out species)

---

## Phase 4: Inference Pipeline

### Step 4.1 — Reference Caching (`plmlof/inference/reference_cache.py`)
- **Key optimization**: pre-compute ESM2 embeddings for all reference genes of a strain
- Cache to disk (torch tensors in a strain-specific directory)
- On new run: load cached embeddings, skip re-encoding
- Invalidate cache when reference changes
- This makes inference **~2x faster** since only variant sequences need encoding

### Step 4.2 — VCF/FASTA Handler (`plmlof/inference/vcf_handler.py`)
- Parse VCF + reference FASTA → extract per-gene variant sequences
- For each gene with variant(s): reconstruct variant protein sequence
- Handle multi-variant genes (apply all variants to the gene)
- Also support direct FASTA input (pre-extracted gene sequences)
- Output: list of `(gene_name, ref_protein, var_protein, nuc_features, variant_details)`

### Step 4.3 — Predictor (`plmlof/inference/predictor.py`)
Main user-facing interface:
```python
predictor = PLMLoFPredictor(model_path, device="cuda")
predictor.load_reference("strain_reference.fasta")  # Pre-caches embeddings

# From VCF:
results = predictor.predict_vcf("variants.vcf")

# From FASTA:
results = predictor.predict_fasta("variant_genes.fasta")

# Each result: {gene, prediction (LoF/WT/GoF), confidence, attribution_map, variant_details}
```
- Batched inference for throughput
- CPU and GPU support
- Returns structured results with attribution

### Step 4.4 — Attribution (`plmlof/inference/attribution.py`)
- Uses `captum` integrated gradients to attribute prediction to input positions
- Maps attribution scores back to amino acid positions → nucleotide positions
- Identifies top-contributing variants for the LoF/GoF call
- Supplements with rule-based evidence (e.g., "frameshift at position 142", "premature stop at position 87")
- Output: per-position importance scores + human-readable summary

---

## Phase 5 (Future): Nucleotide Transformer Integration

### Step 5.1 — NT Encoder (`plmlof/models/nt_encoder.py`)
- Load `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` from HuggingFace
- Encode reference + variant DNA sequences
- LoRA fine-tuning for bacterial domain

### Step 5.2 — Dual-encoder fusion
- Cross-attention between ESM2 protein embeddings and NT DNA embeddings
- Learned gating: model decides how much weight to give protein vs nucleotide signals
- Retrain classification head on fused features

### Step 5.3 — Evo integration (optional, for advanced prokaryotic analysis)
- Evo (7B, trained on prokaryotic genomes) would be the ideal nucleotide encoder for bacteria
- Requires significant GPU resources (RunPod)
- Could replace NT encoder if resources permit

---

## Testing Strategy

### Unit tests (CPU, local)
- `test_model.py`: Forward pass with ESM2-8M (tiny model), verify output shapes [B, 3], gradient flow
- `test_dataset.py`: Load synthetic mini-dataset, verify tokenization, padding, feature extraction
- `test_features.py`: Verify nucleotide feature extraction (frameshift detection, premature stop, etc.)
- `test_predictor.py`: End-to-end predict on 5 synthetic sequences, verify output format
- `test_attribution.py`: Verify attribution produces per-position scores

### Fixtures (`conftest.py`)
- Tiny ESM2 model (`esm2_t6_8M_UR50D`)
- 20-sequence synthetic dataset (5 LoF, 10 WT, 5 GoF)
- Mock reference FASTA, mock VCF

### All tests runnable on CPU without GPU, using ESM2-8M (~32MB)

---

## Relevant Files to Create/Modify

- `plmlof/models/esm2_encoder.py` — ESM2 wrapper with LoRA, uses `transformers.EsmModel` + `peft.LoraConfig`
- `plmlof/models/comparison.py` — Element-wise diff/product + pooling, inspired by siamese network patterns
- `plmlof/models/classifier.py` — Simple MLP head, standard PyTorch `nn.Module`
- `plmlof/models/plmlof_model.py` — Full model, orchestrates encoder → comparison → classifier
- `plmlof/data/features.py` — Rule-based nucleotide feature extraction using `biopython`
- `plmlof/data/dataset.py` — PyTorch `Dataset`, loads from parquet/HF format
- `plmlof/inference/predictor.py` — High-level API, loads model + reference, runs batched inference
- `plmlof/inference/reference_cache.py` — Embedding caching with `torch.save`/`torch.load`
- `plmlof/inference/attribution.py` — `captum.attr.IntegratedGradients` wrapper
- `data/scripts/curate_dataset.py` — Main data curation: merges CARD + DEG + ProteinGym + synthetic
- `scripts/train.py` — Training entry with argparse, 2-stage training loop
- `tests/conftest.py` — Shared fixtures with tiny model and synthetic data

---

## Verification

1. **Unit tests**: Run `pytest tests/ -v` — all pass on CPU with ESM2-8M, no GPU needed
2. **Smoke training**: Run `python scripts/train.py --config configs/model.yaml --max-epochs 2 --tiny` — verify training loop completes on CPU with synthetic data
3. **Inference test**: Run `python scripts/predict.py --reference test_ref.fasta --variants test_var.fasta --model-size 8M` — verify predictions output with correct format
4. **Attribution test**: Verify attribution maps are non-zero and positions near mutations have higher scores
5. **Benchmark vs SNPEff**: On shared test set, compare PLMLoF predictions vs SNPEff annotations for concordance
6. **Cross-species generalization**: Evaluate on held-out species (not seen during training) — metric: macro-F1 > 0.7

---

## Decisions

- **Primary encoder: ESM2** — best protein embeddings, MIT license, well-supported, captures missense/nonsense effects. Protein-level covers ~95% of LoF/GoF mechanisms.
- **Nucleotide effects via engineered features** (Phase 1) — frameshifts, premature stops, start codon loss are deterministic and better captured by rules than learned models. A nucleotide model (Phase 5) adds value for regulatory mutations and subtle effects.
- **Siamese architecture** (shared ESM2 weights for ref + var) — parameter efficient, natural for comparison tasks
- **3-class output** — LoF / Wildtype / GoF with softmax confidence
- **LoRA over full fine-tuning** — memory efficient, faster training, proven effective for domain adaptation
- **Pre-cached reference embeddings** — critical for throughput on massive datasets: encode reference once, reuse for all variants
- **ESM2-8M for testing, ESM2-650M for production** — tests must run locally without GPU; production runs on RunPod
- **Held-out species for validation** — prevents overfitting to well-studied organisms, tests pan-bacterial generalization
- **CARD for GoF, DEG for LoF** — most reliable bacterial-specific sources. ProteinGym DMS provides additional continuous functional data.
- **Phase 5 (NT/Evo) is deferred** — adds complexity, requires GPU. ESM2 + nucleotide features is the pragmatic first model. NT or Evo can be added later for marginal gains on regulatory variants.

---

## Further Considerations

1. **Class imbalance handling**: GoF variants (CARD) may be underrepresented vs LoF. Recommend class-weighted loss + focal loss option depending on final dataset composition.
2. **Multi-variant genes**: When a gene has 5+ mutations, the comparison module needs to handle many simultaneous changes. Current design handles this naturally (encode full variant protein), but attribution becomes harder. May need variant-level decomposition for interpretability.
3. **Regulatory/intergenic variants**: Phase 1 only handles coding variants (protein-level). Non-coding variants (promoter mutations affecting expression) require the nucleotide encoder (Phase 5). These should be flagged as "outside model scope" in Phase 1.
