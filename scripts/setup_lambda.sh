#!/usr/bin/env bash
# scripts/setup_lambda.sh
#
# One-time environment setup on a fresh Lambda Labs A100 instance.
# Run this once after SSH-ing in and cloning the repo.
#
# Usage:
#   git clone https://github.com/Jules-Canada/interpretability-clin-trials.git ignis
#   cd ignis
#   bash scripts/setup_lambda.sh

set -euo pipefail

echo "=== ignis Lambda Labs setup ==="
echo "Python: $(python3 --version)"
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
echo

# ---------------------------------------------------------------------------
# 1. Virtual environment
# ---------------------------------------------------------------------------
echo "--- Creating virtual environment ---"
python3 -m venv .venv
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 2. Install PyTorch with CUDA support
# ---------------------------------------------------------------------------
# A100 supports CUDA 12.x natively — use cu121 build.
echo "--- Installing PyTorch (CUDA 12.1) ---"
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
# Note: torchvision and torchaudio are NOT installed — they pin an older torch version
# and conflict with the torch version pulled in by project dependencies.

# ---------------------------------------------------------------------------
# 3. Install project dependencies
# ---------------------------------------------------------------------------
echo "--- Installing ignis and dependencies ---"
pip install --quiet -e ".[train]"

# ---------------------------------------------------------------------------
# 4. Create data directories
# ---------------------------------------------------------------------------
echo "--- Creating data directories ---"
mkdir -p data/activations
mkdir -p checkpoints/pythia-410m-4096
mkdir -p frontend/graph_data

# ---------------------------------------------------------------------------
# 5. Verify GPU is visible to PyTorch
# ---------------------------------------------------------------------------
echo "--- Verifying GPU ---"
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check driver/torch install'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo
echo "=== Setup complete ==="
echo "Run the pipeline with:"
echo "  bash scripts/run_pipeline.sh"
