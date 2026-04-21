# ignis

**Attribution graphs for clinical trial reasoning in language models.**

Replicates the Cross-Layer Transcoder (CLT) methodology from
[*Circuit Tracing: Revealing Computational Graphs in Language Models*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
and applies it to clinical trial prompts — tracing which internal features drive model
predictions about eligibility, adverse events, and trial endpoints.

---

## What this does

Standard language models are opaque: you can see what they predict, but not why.
Attribution graphs open the black box by identifying the specific internal features
(learned concepts) that caused a particular prediction, and the paths through which
they interact.

This project applies that technique to clinical language, asking questions like:

- When a model predicts a patient is *eligible*, which features fired? Were they
  responding to ECOG status, prior therapy history, or something else?
- When a model flags *hepatotoxicity*, is it tracking enzyme levels, the word "toxic",
  or a combination?
- Do features generalise across trial types, or are they condition-specific?

---

## Model

**EleutherAI/Pythia-410m** — 24 layers, d_model=1024, d_mlp=4096.
Chosen for transparency and reproducibility; future work targets MedGemma-27B + MIMIC-IV.

---

## Pipeline

```
1. Extract activations
   scripts/extract_activations.py
   └─ Streams text from pile-uncopyrighted, runs Pythia, saves residual streams
      and MLP outputs to HDF5 (5M tokens, ~20GB)

2. Train CLT
   scripts/train_clt.py
   └─ Trains a Cross-Layer Transcoder on the cached activations
      (24 encoders + 300 decoder matrices, JumpReLU sparsity)
      ~10hrs on H100, n_features=2048

3. Build attribution graphs
   scripts/run_graphs_batch.py
   └─ For each clinical prompt: freeze attention + LayerNorm, compute
      feature→feature edge weights, prune to top-K, export to JSON

4. Label features
   scripts/collect_graph_features.py   # which features appear in graphs?
   scripts/find_top_activations.py     # what tokens activate each feature?
   scripts/label_features.py           # Claude API → natural-language labels

5. Visualise
   frontend/                           # anthropics/attribution-graphs-frontend
   notebooks/                          # training diagnostics + feature exploration
```

---

## Prompt categories

| Category | Example prompt | Target token |
|---|---|---|
| Eligibility | "The patient meets all inclusion criteria and is therefore" | ` eligible` |
| Eligibility | "With an ECOG performance status of 3, the patient is" | ` excluded` |
| Adverse events | "Grade 3 neutropenia was observed, so the dose was" | ` reduction` |
| Adverse events | "Liver enzyme levels were three times the upper limit of normal, suggesting the drug is" | ` toxic` |
| Endpoints | "The tumor decreased in size by 35%, indicating a partial" | ` response` |
| Endpoints | "The primary endpoint of progression-free survival was" | ` met` |

14 prompts total across eligibility, adverse events, and endpoints.
See `prompts/trial_prompts.json` and `prompts/eligibility.py`, `adverse_events.py`, `endpoints.py`.

---

## Repo layout

```
clt/
  model.py          CrossLayerTranscoder — encoders, JumpReLU, decoder matrices
  train.py          Training loop (model-agnostic via ActivationLoader protocol)
  config.py         CLTConfig and TrainConfig dataclasses
  loader.py         HDF5ActivationLoader, LiveActivationLoader

graphs/
  build.py          Attribution graph construction (frozen attention + LayerNorm)
  prune.py          Top-K node/edge pruning by logit contribution
  export.py         Serialize to frontend JSON schema

prompts/
  eligibility.py    4 eligibility prompts (NSCLC, ECOG, prior therapy)
  adverse_events.py 4 adverse event prompts (hematologic, hepatotoxicity)
  endpoints.py      4 endpoint prompts (PFS, OS, ORR)

scripts/
  extract_activations.py   Dump Pythia residual streams to HDF5
  train_clt.py             Train CLT from cached activations
  run_graphs_batch.py      Build + export graphs for all prompts
  collect_graph_features.py  Collect (layer, feature) pairs from graph JSONs
  find_top_activations.py    Find top activating token contexts per feature
  label_features.py          Call Claude API to label each feature

viz/
  features.py       Activation heatmaps, top-K bar charts, decoder norm plots
  graphs.py         Node contribution charts, influence score plots

notebooks/
  01_training_diagnostics.ipynb   Loss curves, L0 sparsity — pipeline validation
  02_feature_exploration.ipynb    Prompt → features → labels — clinical readout
  03_attribution_graphs.ipynb     Graph construction and completeness analysis

frontend/                         anthropics/attribution-graphs-frontend (cloned)
```

---

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Dev run: extract 50k tokens from pythia-70m (CPU, ~2 min)
python scripts/extract_activations.py \
    --model_name EleutherAI/pythia-70m \
    --output_path data/activations/pythia-70m.h5 \
    --max_tokens 50000

# Train CLT on toy data (CPU, ~1 min)
python scripts/train_clt.py \
    --activation_path data/activations/pythia-70m.h5 \
    --n_layers 6 --d_model 512 --d_mlp 2048 --n_features 512 \
    --n_steps 500 --no_wandb

# Run tests
python -m pytest tests/

# Full pipeline (requires GPU — see scripts/run_pipeline.sh)
bash scripts/run_pipeline.sh
```

---

## Current status

- [x] CLT training loop implemented and tested on toy model
- [x] Activations extracted from Pythia-410m (5M tokens)
- [ ] CLT trained on Pythia-410m — in progress (~12k/50k steps)
- [x] Attribution graph construction implemented
- [x] Frontend rendering graphs locally
- [x] Clinical trial prompts written and verified
- [ ] Batch graphs generated for all 14 trial prompts
- [ ] Feature labeling pipeline run end-to-end

---

## Key findings so far

- **Sparsity coefficient matters early:** `sparsity_coeff=2e-4` lets reconstruction dominate
  and L0 saturates near `n_features`. Updated default to `1e-2`.
- **HDF5 access pattern is critical:** random token sampling caused 0% GPU utilisation.
  Fixed by sampling contiguous blocks (matching the HDF5 chunk size of 1024 tokens).
- **Graph completeness with untrained CLT:** ~0.5–0.8 — reconstruction errors dominate.
  Expect 0.85–0.99 after training converges.
- **L0 at 12k steps:** ~55–75 active features per token out of 2048 — healthy sparsity,
  still tightening as JumpReLU thresholds calibrate.

---

## Long-term goals

### 1. Complete Pythia-410m proof-of-concept
Finish the current training run, generate attribution graphs for all 14 clinical prompts,
label features, and produce a notebook-based readout showing which internal concepts drive
eligibility, adverse event, and endpoint predictions.

### 2. Scale to MedGemma-27B + MIMIC-IV
Apply the same CLT methodology to a medically-trained model using real clinical notes.
MedGemma features are expected to be more clinically meaningful — tracking conditions,
lab values, and eligibility criteria rather than generic linguistic patterns.
Requires PhysioNet credentialing for MIMIC-IV access.

### 3. Find cross-trial generalisable features
Identify which features fire consistently across trial types (e.g. a general "adverse event
severity" feature) vs. which are condition-specific (e.g. NSCLC-specific eligibility
features). This is the core scientific question.

### 4. Publish and communicate
- Submit findings to a clinical ML venue (ML4H, CHIL, or similar workshop)
- Write at least one public technical post explaining the methodology accessibly
- Maintain the GitHub repo as a public reference implementation

---

## References

- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) — Anthropic, 2025
- [anthropics/attribution-graphs-frontend](https://github.com/anthropics/attribution-graphs-frontend) — frontend visualiser
- [EleutherAI/pythia](https://github.com/EleutherAI/pythia) — model family
