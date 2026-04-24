#!/usr/bin/env bash
# scripts/run_pipeline.sh
#
# Full pipeline: extraction → training → graphs → feature labeling prep.
# Run on the Lambda instance. Steps 1–4 all happen here because steps 3–4
# need the HDF5 file (which stays on the instance and is not scp'd home).
#
# Expected runtime: ~1hr extraction, ~10hrs training, ~30min graphs, ~15min features = ~12hrs
# Expected cost:    ~$18 at Lambda Labs H100 rates (~$1.50/hr)
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/run_pipeline.sh
#
# To resume a failed training run:
#   bash scripts/run_pipeline.sh --resume

set -euo pipefail

RESUME=${1:-""}

# ---------------------------------------------------------------------------
# Config — change these if you change model or feature count
# ---------------------------------------------------------------------------
N_FEATURES=2048
N_LAYERS=24
D_MODEL=1024
D_MLP=4096
N_STEPS=50000
BATCH_SIZE=512

ACTIVATION_PATH="data/activations/pythia-410m.h5"
CHECKPOINT_DIR="checkpoints/pythia-410m-${N_FEATURES}"
PROMPTS_FILE="prompts/trial_prompts.json"

echo "=== ignis pipeline: Pythia-410m (n_features=${N_FEATURES}) ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

# ---------------------------------------------------------------------------
# Step 1: Activation extraction
# ---------------------------------------------------------------------------
if [ -f "$ACTIVATION_PATH" ]; then
    EXISTING=$(python3 -c "
import h5py
f = h5py.File('$ACTIVATION_PATH', 'r')
print(f['resid_pre_0'].shape[0])
f.close()
" 2>/dev/null || echo "0")
    echo "--- Activation file exists: $EXISTING tokens already extracted ---"
    if [ "$EXISTING" -ge "5000000" ]; then
        echo "    Skipping extraction (>= 5M tokens present)"
    else
        echo "    File incomplete — re-extracting"
        rm "$ACTIVATION_PATH"
    fi
fi

if [ ! -f "$ACTIVATION_PATH" ]; then
    echo "--- Step 1: Extracting activations from Pythia-410m ---"
    echo "    Target: 5M tokens | ETA: ~1hr"
    python scripts/extract_activations.py \
        --model_name EleutherAI/pythia-410m \
        --output_path "$ACTIVATION_PATH" \
        --max_tokens 5000000 \
        --batch_size 16 \
        --seq_len 128 \
        --flush_every 5
    echo "--- Extraction complete: $(date) ---"
fi

# ---------------------------------------------------------------------------
# Step 2: CLT training
# ---------------------------------------------------------------------------
echo
echo "--- Step 2: Training CLT on Pythia-410m activations ---"
echo "    n_features=${N_FEATURES} | Steps: ${N_STEPS} | ETA: ~10hrs"

RESUME_ARGS=()
if [ "$RESUME" = "--resume" ]; then
    RESUME_ARGS=(--resume)
    echo "    Resuming from latest checkpoint in $CHECKPOINT_DIR"
fi

python scripts/train_clt.py \
    --activation_path "$ACTIVATION_PATH" \
    --n_layers "$N_LAYERS" \
    --d_model "$D_MODEL" \
    --d_mlp "$D_MLP" \
    --n_features "$N_FEATURES" \
    --lr 2e-4 \
    --batch_size "$BATCH_SIZE" \
    --n_steps "$N_STEPS" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_every 5000 \
    --log_every 100 \
    --wandb_group pythia-410m-h100 \
    "${RESUME_ARGS[@]}"

echo
echo "--- Training complete: $(date) ---"

# ---------------------------------------------------------------------------
# Step 3: Attribution graphs
# ---------------------------------------------------------------------------
echo
echo "--- Step 3: Generating attribution graphs ---"
echo "    Prompts: $PROMPTS_FILE"

CHECKPOINT="$CHECKPOINT_DIR/clt_final.pt"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/clt_step_*.pt 2>/dev/null | head -1)
    echo "    clt_final.pt not found — using $CHECKPOINT"
fi

python scripts/run_graphs_batch.py \
    --checkpoint "$CHECKPOINT" \
    --n_layers "$N_LAYERS" \
    --d_model "$D_MODEL" \
    --d_mlp "$D_MLP" \
    --n_features "$N_FEATURES" \
    --prompts_file "$PROMPTS_FILE"

echo "--- Graphs complete: $(date) ---"

# ---------------------------------------------------------------------------
# Step 4: Feature labeling prep (must run here — needs HDF5)
# ---------------------------------------------------------------------------
echo
echo "--- Step 4: Feature labeling prep ---"
echo "    Stripping optimizer state → clt_inference.pt"

INFERENCE_CKPT="$CHECKPOINT_DIR/clt_inference.pt"
python3 -c "
import torch
ckpt = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
torch.save({'model_state_dict': ckpt['model_state_dict'], 'step': ckpt.get('step', 0)}, '$INFERENCE_CKPT')
print('Saved: $INFERENCE_CKPT')
"

echo "    Collecting features from all graph JSONs..."
python scripts/collect_graph_features.py \
    --graph_dir frontend/graph_data \
    --output data/graph_features.json

echo "    Finding top activating tokens per feature (scans HDF5)..."
python scripts/find_top_activations.py \
    --checkpoint "$INFERENCE_CKPT" \
    --activation_path "$ACTIVATION_PATH" \
    --n_layers "$N_LAYERS" \
    --d_model "$D_MODEL" \
    --d_mlp "$D_MLP" \
    --n_features "$N_FEATURES" \
    --features_file data/graph_features.json \
    --output_path data/feature_activations.jsonl

echo "--- Feature labeling prep complete: $(date) ---"

# ---------------------------------------------------------------------------
# Done — print scp checklist
# ---------------------------------------------------------------------------
echo
echo "=== Pipeline complete: $(date) ==="
echo
echo "Run these scp commands FROM YOUR MAC before terminating:"
echo
echo "  INSTANCE=ubuntu@<instance-ip>"
echo "  scp \"\$INSTANCE:ignis/frontend/graph_data/*.json\" frontend/graph_data/"
echo "  scp \"\$INSTANCE:ignis/${INFERENCE_CKPT}\" ${CHECKPOINT_DIR}/"
echo "  scp \"\$INSTANCE:ignis/data/feature_activations.jsonl\" data/"
echo "  scp \"\$INSTANCE:ignis/data/graph_features.json\" data/"
echo
echo "Then on your Mac run:"
echo "  python scripts/label_features.py --resume"
echo
echo "HDF5 (${ACTIVATION_PATH}) stays on the instance — re-extract cheaply next time."
