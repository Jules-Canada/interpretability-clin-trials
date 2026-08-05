# Corpus Experiment — L0 Feature Distribution

**Date:** 2026-06-01
**Status:** Planned

## Question

Why are 92/110 graph features at Layer 0 (structural/formatting)?
Is it the corpus, the sparsity (L0~91), or both?

## Hypothesis

The protocol corpus is formatting-heavy (TOC dots, section numbers, consent
boilerplate, whitespace patterns). If the CLT trains on this distribution, L0
encoders learn structural features that dominate the pruning budget, starving
late-layer medical features. A cleaner corpus should shift the layer distribution.

Alternatively, L0~91 sparsity floods the graph with weakly-contributing features
at every layer, and L0 just happens to have the most — in which case corpus won't
matter and only a sparsity reduction (L0~20-30) will fix it.

## Design

Three CLT training runs on the same architecture, differing only in corpus:

| Arm | Dataset | Text field | Medical? | In-distribution for MedGemma? |
|-----|---------|------------|----------|-------------------------------|
| A — Web control | `allenai/c4` (HF streaming) | `text` | No | Yes — C4 is in the Gemma training mix |
| B — Medical prose | `ccdv/pubmed-summarization` (HF streaming) | `abstract` | Yes | Closer — medical pretraining data |
| C — Medical formatting | `protocols.jsonl` (local) | `full_text` | Yes | Closer — medical pretraining data |

**Why not Pile as the non-medical control:** MedGemma's base model is Gemma 3 4B,
which was not trained on the Pile. Using Pile would confound "non-medical content"
with "OOD text the model handles poorly." C4 was part of the original Gemma training
mix, making it a clean in-distribution non-medical control.

**Held constant across all three arms:**
- Model: `google/medgemma-4b-pt` (34 layers, d_model=2560, d_mlp=10240)
- CLT: n_features=1024, sparsity_coeff=1e-2, 5k steps, batch_size=512
- Extraction: 2M tokens, seq_len=128, batch_size=4, float16, resid_only=False
- Evaluation: same 8 Track B medical-factual prompts
- Metrics: layer distribution of graph features, completeness, L0 norm

## Predictions

| Outcome | What it means | Next step |
|---------|---------------|-----------|
| C (protocols) much worse than A+B | Formatting is the problem | Retrain production CLT on PubMed |
| All three similar (~90% L0) | Sparsity is the problem, not corpus | Run sparsity experiment (L0~20-30) |
| B (PubMed) best, A (C4) middle | Medical content AND clean formatting both help | Use PubMed + higher sparsity |

## Compute Budget

Per arm: ~1hr extraction + ~30min training (5k steps) + ~20min graphs/features = ~2hrs.
Three arms: ~6hrs compute. With pod overhead (setup, SCP, debugging): budget 8-9hrs
wall time, or split across two pod sessions.

Estimated cost: ~$12-15 (A10 at ~$1.50/hr, sufficient for 5k-step diagnostic runs).

## Pod Commands

All commands assume the pod is set up per CLAUDE.md (setup_pod_medgemma.sh done,
.venv active, HF token set, protocols.jsonl scp'd to data/).

### Step 0 — Verify setup

```bash
source .venv/bin/activate
python -c "from transformer_lens import HookedTransformer; print('TL OK')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls data/protocols.jsonl  # only needed for Arm C
```

### Step 1 — Extract activations (run sequentially, ~1hr each)

```bash
# Arm A — C4 (web, non-medical)
python scripts/extract_activations.py \
    --model_name google/medgemma-4b-pt \
    --dataset allenai/c4 --dataset_split train \
    --text_field text \
    --output_path data/activations/medgemma-c4.h5 \
    --max_tokens 2000000 --batch_size 4 --seq_len 128 \
    --dtype float16

# Arm B — PubMed abstracts (medical prose)
python scripts/extract_activations.py \
    --model_name google/medgemma-4b-pt \
    --dataset ccdv/pubmed-summarization --dataset_split train \
    --text_field abstract \
    --output_path data/activations/medgemma-pubmed.h5 \
    --max_tokens 2000000 --batch_size 4 --seq_len 128 \
    --dtype float16

# Arm C — Clinical trial protocols (medical formatting)
python scripts/extract_activations.py \
    --model_name google/medgemma-4b-pt \
    --local_dataset data/protocols.jsonl \
    --text_field full_text \
    --output_path data/activations/medgemma-protocols.h5 \
    --max_tokens 2000000 --batch_size 4 --seq_len 128 \
    --dtype float16
```

### Step 2 — Train CLTs (run sequentially, ~30min each at 5k steps)

```bash
# Arm A — C4
python scripts/train_clt.py \
    --activation_path data/activations/medgemma-c4.h5 \
    --n_layers 34 --d_model 2560 --d_mlp 10240 --n_features 1024 \
    --sparsity_coeff 1e-2 --n_steps 5000 --batch_size 512 \
    --checkpoint_dir checkpoints/corpus-exp-c4 \
    --wandb_group corpus-experiment --save_every 1000

# Arm B — PubMed
python scripts/train_clt.py \
    --activation_path data/activations/medgemma-pubmed.h5 \
    --n_layers 34 --d_model 2560 --d_mlp 10240 --n_features 1024 \
    --sparsity_coeff 1e-2 --n_steps 5000 --batch_size 512 \
    --checkpoint_dir checkpoints/corpus-exp-pubmed \
    --wandb_group corpus-experiment --save_every 1000

# Arm C — Protocols
python scripts/train_clt.py \
    --activation_path data/activations/medgemma-protocols.h5 \
    --n_layers 34 --d_model 2560 --d_mlp 10240 --n_features 1024 \
    --sparsity_coeff 1e-2 --n_steps 5000 --batch_size 512 \
    --checkpoint_dir checkpoints/corpus-exp-protocols \
    --wandb_group corpus-experiment --save_every 1000
```

### Step 3 — Generate graphs + collect features (per arm)

Run `run_graph.py` on the same 8 Track B prompts with each CLT checkpoint,
then `collect_graph_features.py` + `find_top_activations.py` for each arm.
Exact commands depend on how run_graph.py is invoked for batch runs — adapt
from the Track B session (`docs/session_2026_05_30.md`).

### Step 4 — SCP results back to Mac

```bash
# From Mac — adjust INSTANCE to match pod IP/port
INSTANCE=root@<IP>
PORT=<PORT>

for arm in c4 pubmed protocols; do
    scp -P $PORT $INSTANCE:interpretability-clin-trials/checkpoints/corpus-exp-${arm}/clt_step_5000.pt \
        checkpoints/corpus-exp-${arm}/
    scp -P $PORT $INSTANCE:interpretability-clin-trials/data/feature_activations_${arm}.jsonl \
        data/
done
```

### Step 5 — Compare (local, after SCP)

For each arm, count features by layer from the graph JSONs:
```bash
python -c "
import json, glob, collections
for arm in ['c4', 'pubmed', 'protocols']:
    counts = collections.Counter()
    for f in glob.glob(f'frontend/graph_data/*_{arm}.json'):
        g = json.load(open(f))
        for node in g.get('nodes', []):
            if 'layer' in node:
                counts[node['layer']] += 1
    total = sum(counts.values())
    l0 = counts.get(0, 0)
    print(f'{arm:12s}: {l0}/{total} L0 ({100*l0/total:.0f}%)  layers={sorted(counts.keys())}')
"
```

## Results

*(to be filled after pod session)*

| Arm | L0 features | Total features | L0 % | L5+ features | Mean completeness |
|-----|-------------|----------------|-------|--------------|-------------------|
| A — C4 | | | | | |
| B — PubMed | | | | | |
| C — Protocols | | | | | |

## Interpretation

*(to be filled after results)*
