#!/usr/bin/env bash
# scripts/setup_pod_circuit_tracer.sh
#
# One-time environment setup on a fresh GPU pod (RunPod H100) for Stage 1:
# self-hosted circuit-tracer + nnsight + Gemma Scope 2 transcoders.
# No CLT training, no HDF5, no corpus — just attribution graphs on pretrained
# transcoders. See docs/ignis_approach_handoff.md and CLAUDE.md Phase 2.
#
# Prerequisites (accept terms while logged in as your HF user):
#   - https://huggingface.co/google/gemma-3-4b-it
#   - the MedGemma variant you intend to run (e.g. google/medgemma-4b-it / -pt)
#   - https://huggingface.co/google/gemma-2-2b   (used by the --smoke step)
#
# Usage:
#   git clone https://github.com/Jules-Canada/interpretability-clin-trials.git ignis
#   cd ignis
#   export HF_TOKEN=<your HF read token>
#   bash scripts/setup_pod_circuit_tracer.sh

set -euo pipefail

echo "=== ignis pod setup (Stage 1: circuit-tracer) ==="
echo "Python: $(python3 --version)"
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
echo "Disk:   $(df -h . | awk 'NR==2{print $2" total, "$4" free"}')"
echo

# ---------------------------------------------------------------------------
# 0. Force HF cache onto /workspace + persist venv activation
# ---------------------------------------------------------------------------
# RunPod's root disk is ~20GB. Model weights add up fast (gemma-2-2b +
# gemma-3-4b-it + a MedGemma variant + the transcoder sets = 30GB+), so push the
# HF cache onto the large /workspace volume. Idempotent.
echo "--- Relocating HF cache to /workspace ---"
if [ -d /workspace ]; then
    mkdir -p /workspace/.cache/huggingface
    export HF_HOME=/workspace/.cache/huggingface
    grep -q 'HF_HOME=/workspace' ~/.bashrc 2>/dev/null \
        || echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    # Absolute path captured now — a literal $PWD resolves to /root in a fresh
    # shell (e.g. a new tmux pane) and silently fails to activate the venv.
    VENV_ACT="$(pwd)/.venv/bin/activate"
    grep -qF "$VENV_ACT" ~/.bashrc 2>/dev/null \
        || echo "source '$VENV_ACT' 2>/dev/null" >> ~/.bashrc
    echo "  HF_HOME=${HF_HOME}"
else
    echo "  WARNING: /workspace not found — model weights will go to the root disk."
    echo "  Make sure root has 40GB+ free or downloads will fail."
fi

# ---------------------------------------------------------------------------
# 0b. tmux (keep attribution runs alive across SSH drops)
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
pip install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# 2. PyTorch (CUDA 12.1)
# ---------------------------------------------------------------------------
# torchvision/torchaudio are NOT installed — they pin an older torch and break
# the transformer_lens import that circuit-tracer pulls in (known gotcha).
echo "--- Installing PyTorch (CUDA 12.1) ---"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 3. circuit-tracer + Stage 1 deps
# ---------------------------------------------------------------------------
# circuit-tracer pulls transformer_lens + transformers; we add the nnsight
# backend (arbitrary HF models incl. Gemma 3) and jinja2 (apply_chat_template).
# We do NOT `pip install -e .` the ignis repo — Stage 1 needs none of the CLT
# deps; the scripts import the pure-python `prompts` module via sys.path.
echo "--- Installing circuit-tracer + nnsight ---"
pip install --quiet "git+https://github.com/safety-research/circuit-tracer.git"
# hf_transfer: RunPod images set HF_HUB_ENABLE_HF_TRANSFER=1 but don't ship the
# package, which hard-fails every model/transcoder download until installed.
pip install --quiet nnsight jinja2 hf_transfer

# ---------------------------------------------------------------------------
# 4. HuggingFace login (gated: gemma-3-4b-it, MedGemma, gemma-2-2b)
# ---------------------------------------------------------------------------
echo "--- HuggingFace login ---"
echo "Accept terms for every model you'll run (gemma-3-4b-it, your MedGemma"
echo "variant, gemma-2-2b for --smoke) before this will work."
echo
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null \
    || hf auth login --token "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

# ---------------------------------------------------------------------------
# 5. Output dirs for graphs + sweeps
# ---------------------------------------------------------------------------
echo "--- Creating output directories ---"
mkdir -p frontend/graph_data data

# ---------------------------------------------------------------------------
# 6. Verify GPU + circuit-tracer import
# ---------------------------------------------------------------------------
echo "--- Verifying GPU + imports ---"
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check driver/torch install'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import circuit_tracer
from circuit_tracer import attribute, ReplacementModel  # noqa: F401
print('circuit_tracer import OK')
"

echo
echo "=== Setup complete ==="
echo "Activate the venv in every new shell (or tmux pane):"
echo "  source .venv/bin/activate"
echo
echo "Then, in order:"
echo "  1. python scripts/run_graphs_ct.py --smoke           # gemma-2-2b API check"
echo "  2. python deferred/scripts/sweep_eligibility.py --model google/gemma-3-4b-it  # Stage 1 gate, closed"
echo "  3. python scripts/run_graphs_ct.py --model google/gemma-3-4b-it \\"
echo "         --transcoders mwhanna/gemma-scope-2-4b-it"
echo "  4. repeat 2-3 for your MedGemma variant (same --transcoders set)"
