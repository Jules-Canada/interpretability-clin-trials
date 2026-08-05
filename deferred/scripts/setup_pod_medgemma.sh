#!/usr/bin/env bash
# scripts/setup_pod_medgemma.sh
#
# One-time environment setup on a fresh GPU pod (RunPod H100) for MedGemma-4B-pt.
# Run this once after SSH-ing in and cloning the repo.
#
# Prerequisites:
#   - Accept MedGemma terms at https://huggingface.co/google/medgemma-4b-pt before running
#   - Have your HuggingFace token with read access ready
#
# Usage:
#   git clone https://github.com/Jules-Canada/interpretability-clin-trials.git ignis
#   cd ignis
#   bash scripts/setup_pod_medgemma.sh

set -euo pipefail

echo "=== ignis pod setup (MedGemma-4B-pt) ==="
echo "Python: $(python3 --version)"
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
echo "Disk:   $(df -h . | awk 'NR==2{print $2" total, "$4" free"}')"
echo

# ---------------------------------------------------------------------------
# 0. Relocate heavy dirs to /workspace + persist HF_HOME
# ---------------------------------------------------------------------------
# RunPod's root disk is ~20GB; the corpus (6GB) + CLT checkpoint (12GB) + HF
# model weights (~7GB for MedGemma) blow past that. /workspace is the large
# attached volume. Symlink data/ and checkpoints/ into /workspace, and force
# HF_HOME there too so model downloads land on the volume.
# Idempotent — re-runs leave existing symlinks alone.
echo "--- Relocating heavy dirs to /workspace ---"
if [ -d /workspace ]; then
    mkdir -p /workspace/data/activations \
             /workspace/checkpoints/medgemma-4b-1024 \
             /workspace/.cache/huggingface
    [ ! -e data ]        && ln -s /workspace/data data
    [ ! -e checkpoints ] && ln -s /workspace/checkpoints checkpoints
    export HF_HOME=/workspace/.cache/huggingface
    grep -q 'HF_HOME=/workspace' ~/.bashrc 2>/dev/null \
        || echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    grep -q 'source.*\.venv/bin/activate' ~/.bashrc 2>/dev/null \
        || echo 'cd /interpretability-clin-trials 2>/dev/null && source .venv/bin/activate 2>/dev/null' >> ~/.bashrc
    echo "  data/        -> $(readlink data 2>/dev/null || echo '(real dir, not symlinked)')"
    echo "  checkpoints/ -> $(readlink checkpoints 2>/dev/null || echo '(real dir, not symlinked)')"
    echo "  HF_HOME=${HF_HOME}"
else
    echo "  WARNING: /workspace not found — heavy artifacts will go to root disk."
    echo "  Make sure root has 30GB+ free or this run will fail."
fi

# ---------------------------------------------------------------------------
# 0b. Install tmux (needed to keep extract/train/graphs alive across SSH drops)
# ---------------------------------------------------------------------------
if ! command -v tmux >/dev/null 2>&1; then
    echo "--- Installing tmux ---"
    apt-get update -qq && apt-get install -y -qq tmux
fi

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
pip install --quiet -e ".[dev]"

# ---------------------------------------------------------------------------
# 4. HuggingFace login (required for gated MedGemma model)
# ---------------------------------------------------------------------------
echo "--- HuggingFace login ---"
echo "You must accept MedGemma terms at:"
echo "  https://huggingface.co/google/medgemma-4b-pt"
echo "before this will work."
echo
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || hf auth login --token "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

# ---------------------------------------------------------------------------
# 5. Create remaining (small) data directories
# ---------------------------------------------------------------------------
# data/ and checkpoints/ already point at /workspace via section 0; just ensure
# expected subdirs exist. frontend/graph_data lives on root — graph JSONs are tiny.
echo "--- Creating data directories ---"
mkdir -p data/activations
mkdir -p checkpoints/medgemma-4b-1024
mkdir -p frontend/graph_data

# ---------------------------------------------------------------------------
# 6. Upload clinical trial protocol corpus
# ---------------------------------------------------------------------------
echo "--- Corpus upload ---"
echo "From your Mac, scp the corpus + CLT checkpoint up:"
echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 \\"
echo "      ~/Desktop/protocol_corpus/ct_corpus/protocols.jsonl \\"
echo "      root@<ip>:interpretability-clin-trials/data/protocols.jsonl"
echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 \\"
echo "      ~/Desktop/ignis/checkpoints/medgemma-4b-1024/clt_inference.pt \\"
echo "      root@<ip>:interpretability-clin-trials/checkpoints/medgemma-4b-1024/"
echo "Both will follow the /workspace symlinks set up in section 0."

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
echo "Activate the venv in every new shell (or tmux pane):"
echo "  source .venv/bin/activate"
echo
echo "Upload corpus, then run:"
echo "  bash scripts/run_pipeline_medgemma.sh"

