# PLMLoF RunPod Dockerfile
# Base: PyTorch with CUDA support (matches RunPod GPU pods)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

LABEL maintainer="PLMLoF" \
      description="PLM-based LoF/GoF variant classifier for bacterial genomes"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub

WORKDIR /workspace/PLMLoF

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e ".[tracking,dev]"

# Expose port for optional tensorboard/wandb
EXPOSE 6006

# Default entrypoint — can be overridden
ENTRYPOINT ["/bin/bash"]
