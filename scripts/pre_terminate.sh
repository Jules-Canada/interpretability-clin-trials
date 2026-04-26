#!/usr/bin/env bash
# scripts/pre_terminate.sh
#
# Run this on the compute instance BEFORE terminating, if the pipeline was
# interrupted or run in parts. Handles everything needed before scp home.
#
# If you ran run_pipeline.sh to completion, this is already done — skip it.
#
# Usage (Pythia-410m):
#   bash scripts/pre_terminate.sh checkpoints/pythia-410m-2048 2048
#
# Usage (MedGemma-4B-pt):
#   bash scripts/pre_terminate.sh checkpoints/medgemma-4b-1024 1024 \
#       /medgemma-4b.h5 34 2560 10240

set -euo pipefail

CHECKPOINT_DIR=${1:-"checkpoints/pythia-410m-2048"}
N_FEATURES=${2:-2048}
ACTIVATION_PATH=${3:-"data/activations/pythia-410m.h5"}
N_LAYERS=${4:-24}
D_MODEL=${5:-1024}
D_MLP=${6:-4096}

echo "=== pre_terminate.sh ==="
echo "  CHECKPOINT_DIR : $CHECKPOINT_DIR"
echo "  N_FEATURES     : $N_FEATURES"
echo "  ACTIVATION_PATH: $ACTIVATION_PATH"
echo "  N_LAYERS       : $N_LAYERS"
echo "  D_MODEL        : $D_MODEL"
echo "  D_MLP          : $D_MLP"
echo

# Resolve checkpoint: prefer clt_final.pt, fall back to latest step checkpoint
CHECKPOINT="$CHECKPOINT_DIR/clt_final.pt"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/clt_step_*.pt 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: no checkpoint found in $CHECKPOINT_DIR"
        exit 1
    fi
    echo "Note: clt_final.pt not found, using $CHECKPOINT"
fi
echo "Checkpoint: $CHECKPOINT"

# Strip optimizer state
INFERENCE_CKPT="$CHECKPOINT_DIR/clt_inference.pt"
if [ -f "$INFERENCE_CKPT" ]; then
    echo "clt_inference.pt already exists — skipping strip"
else
    echo "Stripping optimizer state → $INFERENCE_CKPT ..."
    python3 -c "
import torch
ckpt = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
torch.save({'model_state_dict': ckpt['model_state_dict'], 'step': ckpt.get('step', 0)}, '$INFERENCE_CKPT')
print('Saved: $INFERENCE_CKPT')
"
fi

# Collect graph features
echo
echo "Collecting (layer, feature) pairs from all graph JSONs..."
python scripts/collect_graph_features.py \
    --graph_dir frontend/graph_data \
    --output data/graph_features.json

# Find top activations (needs HDF5)
if [ ! -f "$ACTIVATION_PATH" ]; then
    echo
    echo "ERROR: HDF5 not found at $ACTIVATION_PATH"
    echo "Re-extract activations first, or pass the correct path as argument 3."
    exit 1
fi

if [ -f "data/feature_activations.jsonl" ]; then
    echo
    echo "feature_activations.jsonl already exists — skipping find_top_activations"
    echo "(delete it and re-run if you want to regenerate)"
else
    echo
    echo "Finding top activating tokens per feature..."
    python scripts/find_top_activations.py \
        --checkpoint "$INFERENCE_CKPT" \
        --activation_path "$ACTIVATION_PATH" \
        --n_layers "$N_LAYERS" \
        --d_model "$D_MODEL" \
        --d_mlp "$D_MLP" \
        --n_features "$N_FEATURES" \
        --features_file data/graph_features.json \
        --output_path data/feature_activations.jsonl
fi

# Print scp checklist
echo
echo "=== Ready to terminate. Run these FROM YOUR MAC: ==="
echo
echo "  INSTANCE=root@<instance-ip>"
echo "  scp \"\$INSTANCE:ignis/frontend/graph_data/*.json\" frontend/graph_data/"
echo "  scp \"\$INSTANCE:ignis/${INFERENCE_CKPT}\" ${CHECKPOINT_DIR}/"
echo "  scp \"\$INSTANCE:ignis/data/feature_activations.jsonl\" data/"
echo "  scp \"\$INSTANCE:ignis/data/graph_features.json\" data/"
echo
echo "HDF5 (${ACTIVATION_PATH}) stays on instance — re-extract next time."
