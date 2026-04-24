#!/usr/bin/env bash
# scripts/run_pipeline_medgemma.sh
#
# End-to-end CLT training pipeline for MedGemma-4B-pt on a Lambda Labs H100.
#
# Model: google/medgemma-4b-pt (Gemma 3 4B, 34 layers, d_model=2560, d_mlp=10240 GeGLU)
# Corpus: clinical trial protocols (data/protocols.jsonl, --text_field full_text)
# Storage: float16 HDF5 — ~400GB for 2M tokens (resid + mlp_post, 34 layers)
#          Requires a 500GB+ disk (use 1TB Lambda volume to be safe).
#
# Usage (from ignis/ directory with .venv active):
#   source .venv/bin/activate
#   bash scripts/run_pipeline_medgemma.sh
#
# To resume after interruption: re-run from the failed step.
# Before terminating the instance: bash scripts/pre_terminate.sh medgemma-4b-1024 1024

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME="google/medgemma-4b-pt"
N_LAYERS=34
D_MODEL=2560
D_MLP=10240
N_FEATURES=1024
MAX_TOKENS=2_000_000
BATCH_SIZE=4
SEQ_LEN=128
CORPUS="data/protocols.jsonl"
TEXT_FIELD="full_text"
CHECKPOINT_DIR="checkpoints/medgemma-4b-1024"
HDF5_PATH="/workspace/medgemma-4b.h5"
GRAPH_DIR="frontend/graph_data"
PROMPT_FILE="prompts/trial_prompts.json"

echo "=== ignis MedGemma-4B-pt pipeline ==="
echo "Model:      ${MODEL_NAME}"
echo "Corpus:     ${CORPUS}"
echo "Max tokens: ${MAX_TOKENS}"
echo "n_features: ${N_FEATURES}"
echo "HDF5 path:  ${HDF5_PATH}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Disk:       $(df -h . | awk 'NR==2{print $4" free"}')"
echo

if [[ ! -f "${CORPUS}" ]]; then
    echo "ERROR: corpus not found at ${CORPUS}"
    echo "Upload it first: scp protocols_backup.jsonl ubuntu@<ip>:ignis/${CORPUS}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Extract activations
# ---------------------------------------------------------------------------
echo "--- Step 1: Extract activations (${MAX_TOKENS} tokens, float16) ---"
python scripts/extract_activations.py \
    --model_name "${MODEL_NAME}" \
    --output_path "${HDF5_PATH}" \
    --max_tokens "${MAX_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --local_dataset "${CORPUS}" \
    --text_field "${TEXT_FIELD}" \
    --dtype float16 \
    --flush_every 5

echo "Step 1 complete. HDF5: $(du -sh ${HDF5_PATH} | cut -f1)"

# ---------------------------------------------------------------------------
# Step 2: Train CLT
# ---------------------------------------------------------------------------
echo "--- Step 2: Train CLT (n_features=${N_FEATURES}, 50k steps) ---"
python scripts/train_clt.py \
    --model_name "${MODEL_NAME}" \
    --hdf5_path "${HDF5_PATH}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --n_features "${N_FEATURES}" \
    --n_steps 50000 \
    --batch_size 512 \
    --lr 2e-4 \
    --sparsity_coeff 1e-2

echo "Step 2 complete."

# ---------------------------------------------------------------------------
# Step 3: Build attribution graphs for all clinical prompts
# ---------------------------------------------------------------------------
echo "--- Step 3: Build attribution graphs ---"
python scripts/run_graphs_batch.py \
    --model_name "${MODEL_NAME}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${GRAPH_DIR}"

echo "Step 3 complete. Graphs: $(ls ${GRAPH_DIR}/*.json 2>/dev/null | wc -l)"

# ---------------------------------------------------------------------------
# Step 4: Strip optimizer state, collect graph features, find top activations
# ---------------------------------------------------------------------------
echo "--- Step 4: Post-processing ---"
bash scripts/pre_terminate.sh "${CHECKPOINT_DIR}" "${N_FEATURES}"

echo
echo "=== Pipeline complete ==="
echo "Run the scp commands printed above before terminating the instance."
