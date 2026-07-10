# Stage 1 Optimization Guide

## 🔍 Current Stage 1 Pipeline (How Everything Works)

### Data Flow
```
Pre-computed Embeddings (disk)
    ↓
CachedEmbeddingDataset loads:
  • ref_mean [N, 1280]  ← ESM2-650M mean pooling over reference protein
  • ref_max  [N, 1280]  ← ESM2-650M max pooling over reference protein
  • var_mean [N, 1280]  ← Mean pooling over variant protein
  • var_max  [N, 1280]  ← Max pooling over variant protein
  • nuc_features [N, 12] ← Engineered features (length change, stops, etc.)
  • labels [N]          ← 0=LoF, 1=WT, 2=GoF
  • dms_scores [N]      ← DMS fitness z-scores (optional)
    ↓
DataLoader (batch_size=64, grad_accum=4, effective=256)
    ↓
[Optional] Cross-Attention on 4 pooled vectors
    ↓
ComparisonModule:
  1. Concatenate: ref_pool = [ref_mean || ref_max]  [B, 2560]
                  var_pool = [var_mean || var_max]  [B, 2560]
  2. Compute: diff = ref_pool - var_pool            [B, 2560]
              prod = ref_pool * var_pool            [B, 2560]
  3. Concat: [diff || prod || ref || var]           [B, 10240]
  4. LayerNorm(10240)
  5. Project: Linear(10240 → 2560) → GELU → Linear(2560 → 5120)
    ↓
Feature Fusion:
  • LayerNorm on nuc_features [B, 12]
  • Concatenate [comparison || nuc_features] → [B, 5132]
    ↓
ClassifierHead MLP:
  • Linear(5132 → 512) → ReLU → Dropout(0.2)
  • Linear(512 → 128)  → ReLU → Dropout(0.14)
  • Linear(128 → 3)    → Logits
    ↓
[Optional] RegressionHead: Linear(5132 → 128) → ReLU → Linear(128 → 1)
    ↓
Loss Functions:
  • Classification: FocalLoss(gamma=2.0, label_smoothing=0.05)
  • Regression: SmoothL1Loss (weight=0.1, 3-epoch warmup)
    ↓
Optimization:
  • Optimizer: AdamW(lr=5e-5, weight_decay=0.01)
  • Scheduler: 10% linear warmup → ReduceLROnPlateau (factor=0.5, patience=5)
  • Mixed Precision: bf16 on Ampere GPU
  • Gradient Clipping: max_norm=1.0
  • Early Stopping: patience=15 epochs (with EMA smoothing, α=0.3)
```

### Current Performance (Epoch 193)
- **Macro F1**: 0.7929 (+2.8% vs baseline 0.771)
- **Per-Class Recall**: LoF=83.3%, WT=70.7%, GoF=83.8%
- **Spearman**: 0.4 (DMS correlation)
- **Training Time**: ~3-4 hours for 200 epochs with cached embeddings

---

## 🎯 Stage 1 Optimization Opportunities

### 🔥 **Priority 1: Embedding-Level Improvements**

#### A. Multi-Layer Extraction (HIGH IMPACT)
**Current**: Using only ESM2's final layer (layer 33)  
**Problem**: Different layers capture different semantic levels  
**Solution**: Extract from multiple layers and aggregate

**Expected Gain**: +1-2% macro_f1

**Implementation**:
```python
# Modify scripts/precompute_embeddings.py, line ~95
outputs = model(ids, attention_mask=mask, output_hidden_states=True)

# Option 1: Average layers 20-30
layer_stack = torch.stack(outputs.hidden_states[20:31])  # [11, B, L, D]
weighted_hidden = layer_stack.mean(dim=0)  # [B, L, D]

# Option 2: Learnable layer weights (requires Stage 1 code change)
# Add to ComparisonModule: self.layer_weights = nn.Parameter(torch.ones(11) / 11)
# weighted_hidden = (layer_stack * layer_weights.view(11, 1, 1, 1)).sum(0)

mean_p, max_p = _pool(weighted_hidden, mask)
```

**Why this works**:
- ESM2 layers 1-10: Local patterns, amino acid properties
- Layers 11-22: Secondary structure, local motifs  
- Layers 23-33: Global context, functional properties
- Averaging captures richer representations than final layer alone

---

#### B. Attention-Weighted Pooling (MEDIUM IMPACT)
**Current**: Uniform mean+max pooling  
**Problem**: All residues weighted equally, even uninformative padding  
**Solution**: Weight residues by their self-attention scores

**Expected Gain**: +0.5-1% macro_f1

**Implementation**:
```python
# Add to scripts/precompute_embeddings.py
outputs = model(ids, attention_mask=mask, output_attentions=True)
last_hidden = outputs.last_hidden_state  # [B, L, D]

# Average attention across heads in final layer
attn_weights = outputs.attentions[-1].mean(dim=1)  # [B, L, L] → [B, L]
attn_weights = attn_weights.mean(dim=1)  # [B, L]

# Normalize and apply mask
attn_weights = attn_weights * mask.float()
attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

# Attention-weighted pooling
attn_pool = (last_hidden * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

# Store attn_pool in addition to mean/max
```

---

#### C. [CLS] Token Baseline (LOW EFFORT)
**Current**: Not using [CLS] token (position 0)  
**Quick test**: Compare [CLS] vs mean+max pooling

**Implementation**:
```python
# In precompute_embeddings.py, add:
cls_emb = last_hidden[:, 0, :]  # [B, D]
# Save as "ref_cls" and "var_cls" in cached embeddings
```

---

### ⚡ **Priority 2: Architecture Enhancements**

#### D. Gated Comparison Features (MEDIUM IMPACT)
**Current**: Simple concatenation [diff || prod || ref || var]  
**Problem**: All features treated equally, some may be noisy  
**Solution**: Learn to gate which features are important

**Expected Gain**: +0.5-1% macro_f1

**Implementation** (modify `plmlof/models/comparison.py`):
```python
class ComparisonModule(nn.Module):
    def __init__(self, hidden_size: int, pool_strategy: str = "mean_max"):
        super().__init__()
        # ... existing code ...
        
        # Add gating mechanism
        raw_size = 4 * hidden_size * 2  # 10240 for mean_max
        self.gate_proj = nn.Sequential(
            nn.Linear(raw_size, raw_size // 4),
            nn.ReLU(),
            nn.Linear(raw_size // 4, raw_size),
            nn.Sigmoid(),  # Output [0, 1] weights
        )
    
    def project(self, raw_comparison: torch.Tensor) -> torch.Tensor:
        # Compute gates
        gates = self.gate_proj(raw_comparison)  # [B, 10240]
        gated_features = raw_comparison * gates  # Element-wise gating
        
        # Project gated features
        return self._proj(self._pre_norm(gated_features))
```

---

#### E. Residual Connections in Classifier (LOW EFFORT)
**Current**: Linear → ReLU → Dropout (no skip connections)  
**Solution**: Add residual connections for better gradient flow

**Expected Gain**: +0.3-0.5% macro_f1

**Implementation** (modify `plmlof/models/classifier.py`):
```python
class ClassifierHead(nn.Module):
    def __init__(self, input_size: int, hidden_dims: list[int], ...):
        super().__init__()
        # Add input projection to match first hidden dim
        self.input_proj = nn.Linear(input_size, hidden_dims[0])
        
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        self.output = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)  # [B, 512]
        
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            x = F.relu(x)
            if x.shape == residual.shape:  # Residual connection
                x = x + residual
            x = F.dropout(x, p=dropout, training=self.training)
        
        return self.output(x)
```

---

### 🔬 **Priority 3: Loss Function Tuning**

#### F. Class-Balanced Focal Loss (MEDIUM IMPACT)
**Current**: Uniform focal_gamma=2.0 across all classes  
**Problem**: WT class (70.7% recall) underperforming vs LoF/GoF (83%+)  
**Solution**: Use per-class focal gamma or class weights

**Expected Gain**: +0.5-1% macro_f1 (by lifting WT recall to 75%+)

**Implementation**:
```python
# Option 1: Per-class focal gamma (requires custom loss)
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, class_gammas: list[float] = [2.0, 3.0, 2.0]):
        # Higher gamma for WT (middle class) → focus more on hard WT examples
        ...

# Option 2: Class weights (already supported)
# In runpod_training.yaml:
class_weights: [1.0, 1.3, 1.0]  # Boost WT class weight
```

---

#### G. Triplet Loss for Embedding Separation (ADVANCED)
**Current**: Only classification loss  
**Problem**: Embeddings not explicitly optimized for class separability  
**Solution**: Add triplet loss to push LoF/WT/GoF embeddings apart

**Expected Gain**: +0.5-1.5% macro_f1

**Implementation**:
```python
# Add to CachedTrainer._compute_loss()
from torch.nn import TripletMarginLoss

triplet_criterion = TripletMarginLoss(margin=1.0)

# Use comparison features as embeddings
anchor = comparison[labels == 1]  # WT
positive = comparison[labels == 1]  # Other WT
negative = comparison[labels != 1]  # LoF or GoF

triplet_loss = triplet_criterion(anchor, positive, negative)
total_loss = cls_loss + 0.1 * triplet_loss  # Small weight
```

---

### 📊 **Priority 4: Data Augmentation**

#### H. Embedding Mixup (HIGH IMPACT, LOW EFFORT)
**Current**: No data augmentation  
**Solution**: Mix embeddings between samples of same class

**Expected Gain**: +1-2% macro_f1

**Implementation** (add to `CachedTrainer._train_epoch()`):
```python
def mixup_embeddings(batch, alpha=0.2):
    """Apply mixup to cached embeddings."""
    lam = np.random.beta(alpha, alpha)
    batch_size = batch['labels'].size(0)
    index = torch.randperm(batch_size).to(batch['labels'].device)
    
    # Only mix within same class
    same_class = batch['labels'] == batch['labels'][index]
    lam_vec = torch.where(same_class, torch.tensor(lam), torch.tensor(1.0))
    lam_vec = lam_vec.view(-1, 1)
    
    # Mix embeddings
    batch['ref_mean'] = lam_vec * batch['ref_mean'] + (1 - lam_vec) * batch['ref_mean'][index]
    batch['ref_max'] = lam_vec * batch['ref_max'] + (1 - lam_vec) * batch['ref_max'][index]
    batch['var_mean'] = lam_vec * batch['var_mean'] + (1 - lam_vec) * batch['var_mean'][index]
    batch['var_max'] = lam_vec * batch['var_max'] + (1 - lam_vec) * batch['var_max'][index]
    
    return batch

# In training loop:
with torch.amp.autocast(...):
    if self.training and np.random.random() < 0.5:  # 50% chance
        batch = mixup_embeddings(batch)
    logits, reg_pred = self._forward(batch)
```

---

#### I. Feature Noise Injection (LOW EFFORT)
**Current**: No noise augmentation  
**Solution**: Add small Gaussian noise during training

**Expected Gain**: +0.3-0.5% macro_f1

**Implementation**:
```python
# In CachedTrainer._forward()
if self.training:
    noise_scale = 0.01
    ref_mean = ref_mean + torch.randn_like(ref_mean) * noise_scale
    ref_max = ref_max + torch.randn_like(ref_max) * noise_scale
    var_mean = var_mean + torch.randn_like(var_mean) * noise_scale
    var_max = var_max + torch.randn_like(var_max) * noise_scale
```

---

### 🛡️ **Priority 5: Regularization & Stability**

#### J. Stochastic Weight Averaging (SWA) (MEDIUM IMPACT)
**Current**: Best single checkpoint saved  
**Solution**: Average weights from final 20 epochs

**Expected Gain**: +0.5-1% macro_f1

**Implementation** (add to `CachedTrainer.train()`):
```python
from torch.optim.swa_utils import AveragedModel, update_bn

# After optimizer creation:
swa_model = AveragedModel(self.comparison)  # Wrap comparison module
swa_start_epoch = max_epochs - 20

for epoch in range(start_epoch, max_epochs + 1):
    # ... training ...
    
    if epoch >= swa_start_epoch:
        swa_model.update_parameters(self.comparison)

# After training:
torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
self.comparison = swa_model.module  # Replace with averaged model
```

---

#### K. Label Smoothing Tuning (LOW EFFORT)
**Current**: label_smoothing=0.05  
**Experiment**: Try 0.1 or 0.15 for better calibration

**Expected Gain**: +0.2-0.5% macro_f1

---

#### L. Warmup Ratio Tuning (LOW EFFORT)
**Current**: warmup_ratio=0.10 (20 epochs for 200-epoch run)  
**Experiment**: Try 0.05 (10 epochs) or 0.15 (30 epochs)

**Expected Gain**: +0.2-0.4% macro_f1

---

### 🔧 **Priority 6: Training Dynamics**

#### M. Cosine Annealing with Warm Restarts (MEDIUM IMPACT)
**Current**: Plateau scheduler (reduces LR on validation plateau)  
**Alternative**: CosineAnnealingWarmRestarts for periodic LR resets

**Expected Gain**: +0.5-1% macro_f1

**Implementation**:
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,  # Initial restart period (20 epochs)
    T_mult=2,  # Double period after each restart: 20 → 40 → 80
    eta_min=1e-6,
)
```

---

#### N. Gradient Accumulation Tuning (LOW EFFORT)
**Current**: grad_accum=4 → effective_batch=256  
**Experiment**: Try grad_accum=8 → effective_batch=512

**Expected Gain**: +0.2-0.5% macro_f1 (larger batches = more stable gradients)

---

#### O. Learning Rate Warmup Scheduling (MEDIUM IMPACT)
**Current**: Linear warmup  
**Alternative**: Exponential or cosine warmup

**Implementation**:
```python
# Cosine warmup (smoother transition)
warmup_scheduler = CosineAnnealingLR(
    optimizer, T_max=warmup_epochs, eta_min=1e-6
)
```

---

## 🎖️ **Recommended Implementation Order**

### Phase 1: Quick Wins (1-2 days)
1. **Embedding Mixup** (Priority H) - Easiest, high impact
2. **Feature Noise Injection** (Priority I) - Trivial to add
3. **Class-Balanced Focal Loss** (Priority F) - Target WT weakness
4. **Label Smoothing Tuning** (Priority K) - One-line config change

**Expected Total Gain**: +2-4% macro_f1 → **0.81-0.83**

### Phase 2: Architecture Improvements (3-5 days)
5. **Multi-Layer Extraction** (Priority A) - Requires re-computing embeddings (~2 hours)
6. **Gated Comparison Features** (Priority D) - Moderate code change
7. **Residual Connections in Classifier** (Priority E) - Simple refactor

**Expected Total Gain**: +1.5-3% macro_f1 → **0.825-0.86**

### Phase 3: Advanced Techniques (1 week)
8. **Stochastic Weight Averaging** (Priority J) - Straightforward implementation
9. **Attention-Weighted Pooling** (Priority B) - Requires re-computing embeddings
10. **Triplet Loss** (Priority G) - Complex but powerful

**Expected Total Gain**: +1-2.5% macro_f1 → **0.835-0.885**

---

## 📈 **Estimated Performance Trajectory**

| Phase | Techniques | Estimated Macro F1 | Gain from Current |
|-------|-----------|-------------------|------------------|
| Current | Plateau scheduler | 0.7929 | Baseline |
| Phase 1 | Quick wins | 0.81-0.83 | +1.7-3.7% |
| Phase 2 | Architecture | 0.825-0.86 | +3.2-6.7% |
| Phase 3 | Advanced | 0.835-0.885 | +4.2-9.2% |
| **Stage 2** | LoRA fine-tuning | **0.85-0.90** | **+5.7-10.7%** |

---

## 🚀 **Getting Started**

### Step 1: Analyze Current Embeddings
```bash
python scripts/analyze_embeddings.py
```
This will show if there are any normalization or distribution issues.

### Step 2: Implement Quick Wins (Phase 1)
Start with embedding mixup - add to `plmlof/training/trainer.py` after line 585.

### Step 3: Re-train and Compare
```bash
python scripts/train.py \
    --config configs/runpod_training.yaml \
    --model-config configs/runpod_model.yaml \
    --train-data data/processed/train.parquet \
    --val-data data/processed/val.parquet \
    --precomputed data/embeddings
```

Track results in a spreadsheet to identify which techniques work best.

---

## 💡 **Key Insights**

1. **Embeddings are the foundation** - Multi-layer extraction likely has the highest ceiling
2. **WT class is the bottleneck** - Target it specifically with class weights/focal loss
3. **Data augmentation is free performance** - Mixup requires no new data or embeddings
4. **Stage 2 is still the biggest lever** - Don't forget LoRA fine-tuning after optimizing Stage 1

---

## ⚠️ **Pitfalls to Avoid**

1. **Don't re-compute embeddings unnecessarily** - Only for multi-layer/attention pooling
2. **Test one change at a time** - Avoid confounding effects
3. **Watch for overfitting** - Validation loss should track training loss
4. **Don't over-regularize** - Too much dropout/noise hurts performance
5. **Profile GPU memory** - Some techniques (SWA, triplet loss) increase memory usage

---

**Good luck! You're already at 0.7929 - these optimizations could push you to 0.85+ before even touching Stage 2! 🎯**
