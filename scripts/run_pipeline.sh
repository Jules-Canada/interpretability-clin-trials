#!/usr/bin/env bash
# scripts/run_pipeline.sh
#
# Full extraction + training + graph generation pipeline for Pythia-410m on Lambda Labs A100.
# Expected runtime: ~1hr extraction, ~4hrs training, ~30min graphs = ~6hrs total.
# Expected cost:    ~$9 at Lambda Labs A100 rates (~$1.50/hr).
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/run_pipeline.sh
#
# To resume a failed training run:
#   bash scripts/run_pipeline.sh --resume

set -euo pipefail

RESUME=${1:-""}

ACTIVATION_PATH="data/activations/pythia-410m.h5"
CHECKPOINT_DIR="checkpoints/pythia-410m-4096"
PROMPTS_FILE="prompts/trial_prompts.json"

echo "=== ignis pipeline: Pythia-410m (n_features=4096) ==="
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
        --seq_len 128
    echo "--- Extraction complete: $(date) ---"
fi

# ---------------------------------------------------------------------------
# Step 2: CLT training
# ---------------------------------------------------------------------------
echo
echo "--- Step 2: Training CLT on Pythia-410m activations ---"
echo "    n_features=4096 | Steps: 50,000 | ETA: ~4hrs"

RESUME_ARGS=()
if [ "$RESUME" = "--resume" ]; then
    RESUME_ARGS=(--resume)
    echo "    Resuming from latest checkpoint in $CHECKPOINT_DIR"
fi

python scripts/train_clt.py \
    --activation_path "$ACTIVATION_PATH" \
    --n_layers 24 \
    --d_model 1024 \
    --d_mlp 4096 \
    --n_features 4096 \
    --lr 2e-4 \
    --batch_size 512 \
    --n_steps 50000 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_every 5000 \
    --log_every 100 \
    --wandb_group pythia-410m-a100 \
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
    # Fall back to latest checkpoint if clt_final.pt not present
    CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/clt_step_*.pt 2>/dev/null | head -1)
    echo "    clt_final.pt not found — using $CHECKPOINT"
fi

python scripts/run_graphs_batch.py \
    --checkpoint "$CHECKPOINT" \
    --n_layers 24 \
    --d_model 1024 \
    --d_mlp 4096 \
    --n_features 4096 \
    --prompts_file "$PROMPTS_FILE"

echo "--- Graphs complete: $(date) ---"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo "=== Pipeline complete: $(date) ==="
echo
echo "Download graphs to your Mac (run this locally, not on Lambda):"
echo "  scp ubuntu@<instance-ip>:ignis/frontend/graph_data/*.json frontend/graph_data/"
echo
echo "Then terminate the Lambda instance."
