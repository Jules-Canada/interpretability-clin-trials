#!/usr/bin/env bash
set -euo pipefail
cd ~/interpretability-clin-trials

HDF5_PATH=/workspace/activations/medgemma-4b-pubmed-full.h5

echo "=== Step 1: Strip optimizer state ==="
for NAME in pubmed-1e2 pubmed-5e2; do
    SRC="checkpoints/${NAME}/clt_final.pt"
    DST="checkpoints/${NAME}/clt_inference.pt"
    if [ -f "$DST" ]; then
        echo "  $DST already exists, skipping"
    else
        python3 -c "
import torch, sys
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
torch.save({'model_state_dict': ckpt['model_state_dict'], 'step': ckpt.get('step', 0)}, sys.argv[2])
print(f'  {sys.argv[1]} -> {sys.argv[2]}')
" "$SRC" "$DST"
    fi
done

echo "=== Step 2: Compute RMS scales ==="
python scripts/compute_clt_scales.py --hdf5 "$HDF5_PATH" --checkpoint checkpoints/pubmed-1e2/clt_inference.pt --n_layers 34
python scripts/compute_clt_scales.py --hdf5 "$HDF5_PATH" --checkpoint checkpoints/pubmed-5e2/clt_inference.pt --n_layers 34

echo "=== Step 3: Generate Track B prompts ==="
python3 -c "
import json, sys
sys.path.insert(0, '.')
passing = ['pharm_warfarin_moa','pharm_metformin','path_insulin_organ','path_mi_artery','path_dvt_complication','anat_csf_production','anat_bile_storage']
from prompts.medical_knowledge import MEDICAL_KNOWLEDGE_PROMPTS
out = [{'id':p['id'],'prompt':p['text'],'target_token':p['target_token']} for p in MEDICAL_KNOWLEDGE_PROMPTS if p['id'] in passing]
json.dump(out, open('data/medical_knowledge_passing.json','w'), indent=2)
print(f'  {len(out)} prompts')

for suffix in ['pubmed_1e2', 'pubmed_5e2']:
    tagged = [dict(p, id=p['id'] + '_' + suffix) for p in out]
    path = f'/tmp/trackb_{suffix}.json'
    json.dump(tagged, open(path, 'w'))
    print(f'  wrote {path}')
"

echo "=== Step 4: Graphs with L0~42 (PubMed, 1e-2) ==="
python scripts/run_graphs_batch.py \
    --checkpoint checkpoints/pubmed-1e2/clt_inference.pt \
    --n_layers 34 --d_model 2560 --d_mlp 10240 --n_features 1024 \
    --model_name google/medgemma-4b-pt \
    --prompts_file /tmp/trackb_pubmed_1e2.json \
    --output_dir frontend/graph_data

echo "=== Step 5: Graphs with L0~16 (PubMed, 5e-2) ==="
python scripts/run_graphs_batch.py \
    --checkpoint checkpoints/pubmed-5e2/clt_inference.pt \
    --n_layers 34 --d_model 2560 --d_mlp 10240 --n_features 1024 \
    --model_name google/medgemma-4b-pt \
    --prompts_file /tmp/trackb_pubmed_5e2.json \
    --output_dir frontend/graph_data

echo "=== Done ==="
