"""Analyze cached embedding statistics to verify preprocessing quality."""

import torch
from pathlib import Path

def analyze_embeddings(emb_path: Path):
    """Check embedding statistics for potential issues."""
    data = torch.load(emb_path, map_location="cpu")
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {emb_path.name}")
    print(f"{'='*60}")
    
    for key in ["ref_mean", "ref_max", "var_mean", "var_max"]:
        if key in data:
            tensor = data[key]  # [N, D]
            print(f"\n{key}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Mean: {tensor.mean().item():.6f}")
            print(f"  Std:  {tensor.std().item():.6f}")
            print(f"  Min:  {tensor.min().item():.6f}")
            print(f"  Max:  {tensor.max().item():.6f}")
            
            # Check for NaN/Inf
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️  NaNs: {nan_count}, Infs: {inf_count}")
            
            # Check distribution skewness
            median = tensor.median().item()
            print(f"  Median: {median:.6f} (mean-median diff: {abs(tensor.mean().item() - median):.6f})")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    emb_dir = Path("data/embeddings")
    
    for split in ["train", "val", "test"]:
        emb_file = emb_dir / f"{split}_embeddings.pt"
        if emb_file.exists():
            analyze_embeddings(emb_file)
