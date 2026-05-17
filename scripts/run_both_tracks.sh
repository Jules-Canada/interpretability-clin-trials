#!/bin/bash
# scripts/run_both_tracks.sh — generate Track B + Track A graphs in one shot.
# Run on pod after checkpoint is uploaded.
set -e

python -c "
import json
passing = ['pharm_warfarin_moa','pharm_metformin','path_insulin_organ','path_mi_artery','path_dvt_complication','anat_csf_production','anat_bile_storage']
from prompts.medical_knowledge import MEDICAL_KNOWLEDGE_PROMPTS
out = [{'id':p['id'],'prompt':p['text'],'target_token':p['target_token']} for p in MEDICAL_KNOWLEDGE_PROMPTS if p['id'] in passing]
json.dump(out, open('data/medical_knowledge_passing.json','w'), indent=2)
print(f'Track B: {len(out)} prompts')
separating = ['cat_receptor_003','cat_histology_002','cat_site_001','cat_surgical_001','cat_allergy_003','cat_histology_001','cat_site_003','cat_allergy_001','cat_surgical_003','cat_surgical_002','cat_receptor_001','cat_allergy_002','cat_steroid_003']
from prompts.categorical_prompts import CATEGORICAL_PROMPTS
stems = set(separating)
out2 = [{'id':p['id'],'prompt':p['text'],'target_token':p['target_token']} for p in CATEGORICAL_PROMPTS if any(p['id'].startswith(s) for s in stems)]
json.dump(out2, open('data/contrastive_passing.json','w'), indent=2)
print(f'Track A: {len(out2)} prompts')
"

echo "=== Track B (standard readout) ==="
python scripts/run_graphs_batch.py --checkpoint_dir checkpoints/medgemma-4b-1024 --n_layers 34 --d_model 2560 --d_mlp 10240 --n_features 1024 --model_name google/medgemma-4b-pt --prompts_file data/medical_knowledge_passing.json --output_dir frontend/graph_data

echo "=== Track A (contrastive readout) ==="
python scripts/run_graphs_batch.py --checkpoint_dir checkpoints/medgemma-4b-1024 --n_layers 34 --d_model 2560 --d_mlp 10240 --n_features 1024 --model_name google/medgemma-4b-pt --prompts_file data/contrastive_passing.json --contrastive --output_dir frontend/graph_data

echo "=== Done ==="
