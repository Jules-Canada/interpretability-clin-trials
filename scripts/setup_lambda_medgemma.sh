#!/usr/bin/env bash
# scripts/setup_lambda_medgemma.sh
#
# One-time environment setup on a fresh Lambda Labs H100 instance for MedGemma-4B-pt.
# Run this once after SSH-ing in and cloning the repo.
#
# Prerequisites:
#   - Accept MedGemma terms at https://huggingface.co/google/medgemma-4b-pt before running
#   - Have your HuggingFace token with read access ready
#
# Usage:
#   git clone https://github.com/Jules-Canada/interpretability-clin-trials.git ignis
#   cd ignis
#   bash scripts/setup_lambda_medgemma.sh

set -euo pipefail

echo "=== ignis Lambda Labs setup (MedGemma-4B-pt) ==="
echo "Python: $(python3 --version)"
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
echo "Disk:   $(df -h . | awk 'NR==2{print $2" total, "$4" free"}')"
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
echo "--- Installing PyTorch (CUDA 12.1) ---"
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
# torchvision/torchaudio are NOT installed — they pin an older torch and conflict.

# ---------------------------------------------------------------------------
# 3. Install project dependencies
# ---------------------------------------------------------------------------
echo "--- Installing ignis and dependencies ---"
pip install --quiet -e ".[train]"

# ---------------------------------------------------------------------------
# 4. HuggingFace login (required for gated MedGemma model)
# ---------------------------------------------------------------------------
echo "--- HuggingFace login ---"
echo "You must accept MedGemma terms at:"
echo "  https://huggingface.co/google/medgemma-4b-pt"
echo "before this will work."
echo
huggingface-cli login --token "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

# ---------------------------------------------------------------------------
# 5. Create data directories
# ---------------------------------------------------------------------------
echo "--- Creating data directories ---"
mkdir -p data/activations
mkdir -p checkpoints/medgemma-4b-1024
mkdir -p frontend/graph_data

# ---------------------------------------------------------------------------
# 6. Upload clinical trial protocol corpus
# ---------------------------------------------------------------------------
echo "--- Corpus upload ---"
echo "Upload protocols_backup.jsonl from your Mac before running the pipeline:"
echo "  scp protocols_backup.jsonl ubuntu@<ip>:ignis/data/protocols.jsonl"

# ---------------------------------------------------------------------------
# 7. Verify GPU is visible to PyTorch
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
echo "Upload corpus, then run:"
echo "  bash scripts/run_pipeline_medgemma.sh"
