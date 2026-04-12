#!/usr/bin/env bash
# scripts/run_pipeline.sh
#
# Full extraction + training pipeline for Pythia-410m on a Lambda Labs A100.
# Expected runtime: ~2hrs extraction, ~6hrs training = ~8hrs total.
# Expected cost:    ~$12 at Lambda Labs A100 rates (~$1.50/hr).
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
CHECKPOINT_DIR="checkpoints/pythia-410m"

echo "=== ignis pipeline: Pythia-410m ==="
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
    echo "    Target: 5M tokens | ETA: ~2hrs"
    python scripts/extract_activations.py \
        --model_name EleutherAI/pythia-410m \
        --output_path "$ACTIVATION_PATH" \
        --max_tokens 5_000_000 \
        --batch_size 8 \
        --seq_len 128
    echo "--- Extraction complete: $(date) ---"
fi

# ---------------------------------------------------------------------------
# Step 2: CLT training
# ---------------------------------------------------------------------------
echo
echo "--- Step 2: Training CLT on Pythia-410m activations ---"
echo "    Steps: 50,000 | Log every: 100 | Save every: 5,000 | ETA: ~6hrs"

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
    --sparsity_coeff 1e-2 \
    --lr 2e-4 \
    --batch_size 512 \
    --n_steps 50000 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_every 5000 \
    --log_every 100 \
    --wandb_group pythia-410m \
    "${RESUME_ARGS[@]}"

echo
echo "=== Pipeline complete: $(date) ==="
echo
echo "Next step — copy the checkpoint back to your Mac:"
echo "  scp -r <instance-ip>:~/ignis/$CHECKPOINT_DIR ./checkpoints/"
