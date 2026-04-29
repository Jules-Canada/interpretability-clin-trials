# Attribution Graph Replication — Clinical Trials

## Project Goal

Replicate the Cross-Layer Transcoder (CLT) methodology from:
> *Circuit Tracing: Revealing Computational Graphs in Language Models*
> https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Then apply attribution graphs to clinical trial prompts to identify features involved
in reasoning about eligibility criteria, adverse events, and endpoint inference.

## Long-Term Goals

1. **Complete Pythia-410m proof-of-concept** — finish current training run, generate
   graphs for all 14 clinical prompts, label features, produce notebook readout.

2. **Scale to MedGemma + MIMIC-IV** — apply CLT to a medically-trained model on real clinical
   notes. Requires PhysioNet credentialing (long-lead item — start early).
   **Active path: fix `graphs/build.py` with frozen autograd backward pass (handles Gemma 3
   post-norms), then use our trained CLT on MedGemma-4B-pt. Gemma Scope 2 CLTs only cover
   270M and 1B — no 4B CLT available. See Phase 2 Models section for full plan.**

3. **Find cross-trial generalisable features** — which features fire consistently across
   trial types vs. which are condition-specific? This is the core scientific question.

4. **Publish and communicate** — target ML4H or CHIL workshop; write at least one public
   technical post. Keep the GitHub repo as a clean public reference implementation.

These goals serve two purposes: contributing to mechanistic interpretability in the
clinical domain, and building a public portfolio for a career pivot into clinical AI.

---

## Models

### Phase 1 — Proof of concept (complete)
**EleutherAI/pythia-410m** (24 layers, d_model=1024, d_mlp=4096)
- Corpus: The Pile (monology/pile-uncopyrighted), 5M tokens
- CLT: n_features=2048, 50k steps, H100
- Key finding: syntactic features, not clinical — expected for a general model

### Phase 2 — Medical domain (active path: fix attribution code + use our trained CLT)

**Decision (2026-04-28): Fix `graphs/build.py` to use frozen autograd backward passes, then use
our existing trained CLT on MedGemma. Do NOT wait for Gemma Scope 2 CLTs.**

**Why not Gemma Scope 2 CLTs:**
Gemma Scope 2 (Sep 2025 technical report, McDougall et al.) only trains CLTs for Gemma 3 270M
and 1B. Gemma 3 4B, 12B, and 27B have single-layer transcoders and crosscoders only — no CLT.
Single-layer transcoders cannot model cross-layer feature→feature edges, which Anthropic's paper
shows are essential for sparse, interpretable graphs. We need a CLT.

**Why not Gemma Scope 2 single-layer transcoders:**
They reconstruct MLP outputs AFTER the post-MLP RMSNorm (i.e. what actually enters the residual),
whereas our CLT reconstructs `mlp_post` (pre-W_out). Different target space, different decoder
semantics — incompatible with our attribution pipeline without a significant rewrite. Also no
cross-layer edges.

**Root cause of completeness failure (diagnosed 2026-04-28):**
Gemma 3 uses post-norms — RMSNorm applied after attention output AND after MLP output before
adding to the residual. The circuit tracing paper used Claude 3 Sonnet (pre-norm only) and never
encountered this. Our manual `_compute_attention_propagated_v` + `_compute_corrected_logit_transfer`
correctly handles attention self-loops but silently skips the post-attention and post-MLP RMSNorm
Jacobians. Result: completeness ≈ −0.002.

**The fix — frozen autograd backward pass (paper's actual method):**
The circuit tracing paper never manually propagates v layer-by-layer. Their "backward Jacobian"
is a PyTorch autograd backward pass with cached values substituted for all nonlinearities:
- `detach(attn_pattern)` — frozen attention weights
- `detach(ln_scale)` — frozen denominator for every LN/RMSNorm including post-norms
- `detach(mlp_gate)` — frozen GeGLU gate values

This is architecture-agnostic: post-norms flow through automatically via frozen-denominator
Jacobians. No manual derivation. One backward pass per target node gives all source edge weights.

Implementation: delete `_compute_attention_propagated_v` and `_compute_corrected_logit_transfer`,
replace with a single function that uses TransformerLens hooks to freeze cached nonlinear values,
then calls `loss.backward()` where `loss = (v_detached · r_L)`. Read off `.grad` at each
layer's residual and mlp_post tensors. Also fix the logit node input vector:
  current (wrong for Gemma): `v = W_U[:,tok] * ln_w / hook_scale`
  correct:                   `v = (W_U[:,tok] - W_U.mean(dim=1)) * ln_w / hook_scale`
  (subtract mean gradient over vocabulary = ∇(logit_tok − mean_logit) as the paper specifies)

**Cost estimate:**
- Engineering: ~2–3 days to implement + test frozen backward pass
- Compute (minimum path, use existing CLT): ~$6 for graph generation on Lambda H100
- Compute (retrain CLT with better sparsity, L0~20–30 vs current L0~91): ~$50
- Existing CLT checkpoint at `checkpoints/medgemma-4b-1024/clt_inference.pt` is usable
  for initial validation; retrain only if graphs are too noisy for feature labeling.

**google/medgemma-4b-pt** (Gemma 3 4B, 34 layers, d_model=2560, d_mlp=10240 GeGLU)
- Confirmed base model: `google/gemma-3-4b-pt` (verified via HuggingFace model tree)
- Gated model — requires HuggingFace terms acceptance before downloading
- Our trained CLT: `checkpoints/medgemma-4b-1024/clt_inference.pt` (n_features=1024, L0~91)

Use TransformerLens for hooking into residual streams and MLP outputs.
Do NOT switch models mid-project without updating this file.

---

## Stack

| Component | Library |
|---|---|
| Model loading & hooks | `transformer_lens` |
| Tensor ops | `torch`, `einops` |
| Experiment tracking | `wandb` |
| Type safety | `jaxtyping`, `beartype` |
| Testing | `pytest` |
| Visualization frontend | `anthropics/attribution-graphs-frontend` (cloned to `./frontend/`) |

Python 3.11. All deps managed via `pyproject.toml`.

---

## Repo Layout

```
.
├── CLAUDE.md                  # This file — always read first
├── pyproject.toml
├── frontend/                  # Cloned: github.com/anthropics/attribution-graphs-frontend
│
├── clt/
│   ├── __init__.py
│   ├── model.py               # CrossLayerTranscoder class
│   ├── train.py               # train_step() and train() — model-agnostic training loop
│   ├── loader.py              # ActivationLoader protocol, LiveActivationLoader, HDF5ActivationLoader
│   └── config.py              # CLTConfig (architecture) and TrainConfig (training) dataclasses
│
├── graphs/
│   ├── __init__.py
│   ├── build.py               # Attribution graph construction
│   ├── prune.py               # Graph pruning (top-K nodes/edges)
│   └── export.py              # Serialize to frontend JSON schema
│
├── interventions/
│   ├── __init__.py
│   └── patch.py               # Feature steering & patching experiments
│
├── prompts/
│   ├── eligibility.py         # Eligibility criteria prompts
│   ├── adverse_events.py      # AE attribution prompts
│   └── endpoints.py           # Endpoint inference prompts
│
├── scripts/
│   ├── extract_activations.py # Dump residual stream + MLP outputs to disk
│   ├── train_clt.py           # Entry point: train CLT
│   ├── run_graph.py           # Entry point: build + export attribution graph
│   └── run_intervention.py    # Entry point: patching experiment
│
├── viz/
│   ├── features.py            # Feature activation plots (heatmap, top-k bar chart, L0 curves)
│   └── graphs.py              # Attribution graph plots (node contributions, layer flow)
│
├── notebooks/
│   ├── 01_training_diagnostics.ipynb   # Loss curves, L0 sparsity — developer use
│   └── 02_feature_exploration.ipynb    # Prompt → features → labels — non-technical readout
│
└── tests/
    ├── test_clt_toy.py        # 2-layer toy model end-to-end pipeline test
    ├── test_attribution.py    # Graph construction unit tests
    └── test_export.py         # Frontend JSON schema validation
```

---

## Key Architectural Facts (read before touching clt/model.py)

From the paper (§ Building an Interpretable Replacement Model):

- The CLT has **L encoder matrices**, one per layer. Each reads from the residual stream
  `x_l` at its layer using a linear encoder + **JumpReLU** nonlinearity.
- Each feature at layer `l'` contributes to **all subsequent MLP outputs** via separate
  decoder matrices `W_dec[l' → l]` for each `l >= l'`.
- MLP output reconstruction at layer `l`:
  `y_hat_l = sum_{l'=1}^{l} W_dec[l'→l] @ a_l'`
- Training loss = MSE reconstruction (summed over layers) + L1 sparsity penalty.
- JumpReLU: zero below threshold `θ`, linear above. Threshold is a learned parameter.

---

## Attribution Graph Facts (read before touching graphs/)

- **Freeze** attention patterns and LayerNorm denominators before computing attributions.
  This makes feature→feature interactions linear and attribution well-defined.
- Nodes: CLT features, token embeddings, reconstruction errors, output logits.
- Edges: linear effects between nodes. Feature pre-activation = sum of input edges.
- Pruning: keep top-K nodes/edges by contribution to target token logit. See §Appendix:
  Graph Pruning for the exact algorithm.
- Export format must match `anthropics/attribution-graphs-frontend` JSON schema.
  Check `frontend/README.md` for the schema spec before writing `graphs/export.py`.

### Attention Jacobians — REQUIRED for valid completeness

**The T matrix in `graphs/build.py` MUST include attention paths or completeness will be
~0.001 (MLP paths only). This is not optional — graphs without attention Jacobians cannot
answer "what caused the logit."**

The transfer matrix from feature `f` at layer `l_s` to the logit must account for the fact
that each layer's residual contribution propagates forward through **both** skip connections
and attention. The correct computation:

1. **Attention Jacobian per layer** (from frozen attention pattern `A_h[target, target]`):
   ```
   J_l = Σ_h  A_h[target, target] * W_V^h @ W_O^h    (d_model × d_model)
   ```
2. **Propagator** (skip connection + attention):
   ```
   P_l = I + J_l
   ```
3. **Effective readout vector at each layer** (backpropagate `v = W_U[:, target_token]`):
   ```
   v_L = v
   v_l = P_l^T @ v_{l+1}    (for l from L-1 down to 0)
   ```
4. **Full transfer matrix** (replace `v` with `v_{l_t+1}` in each layer's contribution):
   ```
   T_full[f, l_s] = Σ_{l_t ≥ l_s}  rms_{l_t} * W_dec[l_s→l_t][:, f] · (W_out[l_t] @ v_{l_t+1})
   ```
   vs. current broken version:
   ```
   T_MLP[f, l_s]  = Σ_{l_t ≥ l_s}  rms_{l_t} * W_dec[l_s→l_t][:, f] · (W_out[l_t] @ v)
   ```

**IMPLEMENTED (2026-04-27).** Verified pythia-70m completeness = 0.91 with random CLT.

Key implementation notes:
- `_compute_readout_vector` uses frozen-denominator LN gradient (NOT autograd — autograd
  gives v·r_L = 0 by Euler's theorem for degree-0 homogeneous LN/RMSNorm).
  Formula: `v = (W_U[:,tok] - mean(W_U[:,tok])) / hook_scale` (Pythia LayerNormPre)
           `v = W_U[:,tok] * ln_final.w / hook_scale` (Gemma RMSNorm)
- Completeness denominator = `logit - b_U[tok]` (Pythia has non-zero unembed bias b_U).
- Attention head cross-position paths captured as `attention` nodes → logit edges.
- All v_at_layer backprop uses float64 on CPU (MPS doesn't support float64; (1+σ)^L ≈
  8000× amplification would cause float32 precision loss in feat_sum/error_sum).

---

## Development Rules

1. **Always start with the toy model test** (`tests/test_clt_toy.py`) before running on
   Pythia. The toy model is a 2-layer, d_model=64 transformer. If the pipeline doesn't
   pass on the toy model, don't scale up.

2. **Log everything to wandb** during CLT training: per-layer reconstruction MSE,
   sparsity (L0 norm of activations), total loss. Group runs by model size.

3. **No magic numbers.** All hyperparameters (L1 coefficient λ, JumpReLU threshold θ,
   pruning K) live in `clt/config.py` as fields of `CLTConfig`.

4. **Shapes in comments.** Every tensor that passes between modules should have its
   shape annotated in the line above, e.g.:
   ```python
   # (batch, seq, d_model)
   x = hook_point.hook(resid_pre)
   ```

5. **Do not modify the frontend submodule** except to update the JSON it reads.

6. **Every milestone gets a notebook.** Each item in the Current Status checklist must have
   a corresponding notebook in `notebooks/` that renders its outputs visually before the
   next milestone begins. Notebooks are the primary format for non-technical readouts —
   use plain-English section headers and captions, not just code and plots.

7. **Use `viz/` for all figures.** Never call matplotlib directly in scripts or notebooks
   without going through a function in `viz/features.py` or `viz/graphs.py`. This keeps
   figures consistent and reusable. Add to `viz/` as new plot types are needed.

8. **Verify completeness before calling graphs valid.** After building any graph, check that
   `completeness >= 0.5`. If it is below that threshold, the T matrix is missing paths
   (almost certainly attention Jacobians) and the graph does not answer "what caused the logit."
   Rendering in the frontend is not sufficient — completeness must be checked numerically.

---

## Clinical Trial Prompt Guidelines

Prompts live in `prompts/`. Each file exports a list of `TrialPrompt` dicts:

```python
TrialPrompt = TypedDict('TrialPrompt', {
    'id': str,               # e.g. "eligibility_nsclc_001"
    'text': str,             # the prompt text
    'target_token': str,     # token whose logit we trace (e.g. "eligible", "yes")
    'domain_tags': list[str] # e.g. ["oncology", "NSCLC", "eligibility"]
})
```

Start with **10–15 prompts per category**. Prioritize:
- Eligibility: NSCLC, breast cancer, renal cell carcinoma (oncology focus)
- Adverse events: hematologic toxicity, hepatotoxicity
- Endpoints: PFS, OS, ORR definitions

When labeling features found in attribution graphs, record labels in
`prompts/feature_labels.jsonl` (one JSON object per line, keyed by feature index).

---

## Pod Setup (RunPod — do this in order on every fresh instance)

```bash
# 1. Clone repo
git clone https://YOUR_TOKEN@github.com/Jules-Canada/ignis.git
cd ignis

# 2. Fix torchvision conflict (breaks transformer_lens import)
pip uninstall torchvision torchaudio -y

# 3. Install project deps
pip install -e .

# 4. Point HuggingFace cache at the volume (avoids "no disk space" on root)
export HF_HOME=/workspace/.cache/huggingface

# 5. HuggingFace login (MedGemma is gated — token is at ~/.cache/huggingface/token on Mac)
python -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"

# 6. SCP checkpoint from Mac (run on Mac, create dir on pod first)
#    On pod:  mkdir -p ~/ignis/checkpoints/medgemma-4b-1024
#    On Mac:  scp -P <PORT> -i ~/.ssh/id_ed25519 checkpoints/medgemma-4b-1024/clt_inference.pt root@<IP>:~/ignis/checkpoints/medgemma-4b-1024/
```

**Known gotchas:**
- `huggingface-cli login` and `hf login` are both broken on this image — use the python one-liner above
- `torchvision` must be uninstalled before importing `transformer_lens` or you get a segfault
- Root disk is ~10GB; always set `HF_HOME=/workspace/.cache/huggingface` before downloading MedGemma (3.6GB weights)
- For graph generation only: A10 (24GB) is sufficient, no need for H100
- Repo clones as `interpretability-clin-trials` not `ignis` — adjust paths accordingly

---

## Compute Notes

- CLT training on Pythia-410m: expect ~4–8 GPU-hours on a single A100 for a reasonable
  run. Use gradient checkpointing if VRAM is tight.
- Activation extraction: extract and cache residual streams + MLP outputs to disk first
  (`scripts/extract_activations.py`), then train CLT offline. Don't re-forward Pythia
  on every CLT training step.
- For local dev/testing, use `pythia-70m` (6 layers) — fast enough to iterate on CPU.

### Pre-termination checklist (run before killing any Lambda instance)

**If `run_pipeline.sh` completed fully, Step 4 already ran — skip to scp.**
**If the pipeline was interrupted, run `bash scripts/pre_terminate.sh` first.** It handles
checkpoint stripping, `collect_graph_features`, and `find_top_activations` in one shot,
then prints the exact scp commands to run. See `docs/pipeline_lessons.md` for why.

scp commands (run from your Mac):
```
INSTANCE=ubuntu@<ip>
scp "$INSTANCE:ignis/frontend/graph_data/*.json" frontend/graph_data/
scp "$INSTANCE:ignis/checkpoints/pythia-410m-2048/clt_inference.pt" checkpoints/pythia-410m-2048/
scp "$INSTANCE:ignis/data/feature_activations.jsonl" data/
scp "$INSTANCE:ignis/data/graph_features.json" data/
```

**Rule: run all 4 scp commands before terminating. The HDF5 stays on the instance and is
re-extracted next time. Use `--resid_only` for find_top_activations runs (~491GB on disk);
use full extraction (resid + mlp_post, ~2.5TB) only for CLT training — needs a larger disk.
`feature_labels.jsonl` is generated locally after scp (run `python scripts/label_features.py --resume`).**

---

## Current Status

### Phase 1 — Pythia-410m (complete)
- [x] Repo scaffolded
- [x] Toy model test passing
- [x] CLT training loop implemented
- [x] Activations extracted from Pythia-410m (5M tokens, Lambda Labs H100)
- [x] CLT trained (50k steps, n_features=2048, H100)
- [x] Attribution graph construction implemented
- [x] Frontend rendering a graph (france_capital, water_boil working locally)
- [x] Clinical trial prompts loaded (14 prompts in prompts/trial_prompts.json)
- [x] Batch graphs generated for all 14 trial prompts
- [x] Feature labeling complete (feature_labels.jsonl, apply_labels.py patched graph JSONs)
- [x] Notebook 02 written — data-driven readout, no model loading required

### Phase 2 — MedGemma-4B-pt (active, 2026-04-28)
- [x] extract_activations.py updated: --local_dataset, --text_field, --dtype flags added
- [x] run_pipeline_medgemma.sh created (n_features=1024, float16, clinical corpus)
- [x] setup_lambda_medgemma.sh created (HuggingFace login, gated model)
- [x] CLT trained (50k steps, n_features=1024, H100, L0~91, mse_mean~0.44)
- [x] Attention Jacobians + frozen-LN v implemented — verified pythia-70m completeness 0.91
- [x] Diagnosed post-norm incompatibility — Gemma 3 post-norms break completeness (~−0.002)
- [x] Confirmed fix path: frozen autograd backward pass (paper's actual method), architecture-agnostic
- [x] Ruled out Gemma Scope 2 CLTs — only available for 270M and 1B, not 4B
- [ ] **Next**: Rewrite `graphs/build.py` — replace manual v_at_layer/effective_readout with frozen
      autograd backward pass. Delete `_compute_attention_propagated_v` and
      `_compute_corrected_logit_transfer`. Use TransformerLens hooks to detach cached LN scales,
      attn patterns, GeGLU gates. Fix logit v to `(W_U[:,tok] - W_U.mean(dim=1)) * ln_w / hook_scale`.
- [ ] Validate on Pythia-70m: completeness should remain ~0.91
- [ ] Validate on MedGemma: completeness should reach ≥ 0.5
- [ ] Rebuild 14 clinical graphs against MedGemma with existing CLT checkpoint
- [ ] If graphs too noisy (L0~91): retrain CLT targeting L0~20–30 (~$50 compute on Lambda H100)
- [ ] Feature labeling with correctly-pruned graphs
- [ ] Notebook 03 — MedGemma feature readout

### Long-lead items
- [ ] PhysioNet credentialing for MIMIC-IV (apply early — takes weeks)

## Findings So Far

- `sparsity_coeff=2e-4` is too weak — reconstruction dominates and L0 saturates near n_features.
  Updated default to `1e-2`. L0 still high at 500 steps on 50k tokens; expect improvement at scale.
- Activation extraction uses `monology/pile-uncopyrighted` streamed from HuggingFace.
  Requires `zstandard` for decompression. Default slice: 50k tokens for dev.
- Training loop is model-agnostic via `ActivationLoader` protocol — switching models
  only requires a new loader, not changes to `clt/train.py`.
- CLT must always be moved to the same device as the model it's paired with.
  Call `clt.to(next(model.parameters()).device)` at entry points (`build_attribution_graph`,
  test fixtures). Never scatter `.to(device)` calls on individual tensors inside helpers.
- Attribution graph completeness (sum of edges to logit / logit value) is ~0.001 with
  MLP-only T matrix — attention paths dominate logit prediction in both Pythia and MedGemma.
  Completeness was never verified for Phase 1 Pythia graphs; those graphs have the same bug.
  The paper's 0.85–0.99 figure requires attention Jacobians in the T matrix. See the
  "Attention Jacobians" section above — this must be implemented before graphs are valid.
- H100 training speed: ~1.37 steps/s with batch_size=512, n_features=2048, 24 layers. 50k steps ≈ 10hrs.
  n_features=4096 exceeded H100 VRAM (81GB needed vs 79GB available) — settled on 2048.
- steps/sec timing added to `_log()` in `clt/train.py` (elapsed, eta, steps/s).
- HDF5 random sampling caused 0% GPU utilization (512 random seeks per step). Fixed by sampling
  contiguous blocks instead — critical when chunk size is 1024 tokens.
- HDF5 now stores `token_ids` dataset (int32) for feature labeling context reconstruction.
  Old HDF5 files without this field need to be re-extracted before running label_features.py.
- flush_every default changed 500→5 to prevent ~200GB RAM accumulation before first disk write.
- torchvision/torchaudio conflict on Lambda: pins torch==2.5.1, incompatible with torch 2.11.0.
  Removed from setup_lambda.sh; uninstall manually on existing instances.
- HDF5 size for 5M tokens, 24 layers: ~2.5TB (resid + mlp_post, float32) or ~491GB (resid only).
  The "~20GB" estimate was wrong. A10 instances have 1.4TB disk — only fits resid_only. Use
  `--resid_only` flag for find_top_activations runs; full extraction needs H100 or dedicated storage.
- Gemma Scope 2 (released Dec 2025) includes cross-layer transcoders for all Gemma 3 sizes
  (270M–27B), SAEs for every layer, and chatbot-tuned model variants. Available on HuggingFace
  and Neuronpedia. Verify MedGemma base model version (likely Gemma 3) before training CLT
  from scratch — may be able to skip the training step entirely.
- `frontend/` is tracked as a gitlink (embedded repo), not a proper submodule. Contents won't
  clone with the outer repo. Convert with `git submodule add` if needed.
- Frontend util.js rewrites all absolute paths to transformer-circuits.pub — added localhost
  check to skip rewrite for local development.
- MedGemma-4B-pt CLT config: n_features=1024 chosen to fit H100 VRAM (34 layers, d_mlp=10240
  GeGLU, decoder matrix is O(L*(L+1)/2 × n_features × d_mlp)). float16 storage reduces HDF5
  to ~400GB for 2M tokens (resid + mlp_post, 34 layers). Use 1TB Lambda disk.
- Clinical trial protocol corpus: 49,002 docs from ClinicalTrials.gov, avg ~26k tokens/doc,
  JSONL with `full_text` field. Use --text_field full_text with extract_activations.py.
  2M tokens covers ~77 documents — sufficient for a proof-of-concept CLT run.
- **Gemma Scope 2 (McDougall et al., Sep 2025) CLTs only cover Gemma 3 270M and 1B** — not 4B,
  12B, or 27B. O(layers²) cost made larger CLTs impractical. 4B has single-layer skip transcoders
  on all layers, but these lack cross-layer edges and target a different space (post-MLP-norm output,
  not mlp_post). Cannot be dropped into our pipeline without significant rework. Our own trained
  CLT (`checkpoints/medgemma-4b-1024/`) remains the right artifact to use.
- **Gemma 3 post-norms break attribution completeness.** With our MedGemma CLT, v·r_L = 21.5
  (v is correct) but completeness ≈ −0.002. Root cause: Gemma 3 applies RMSNorm after each
  attention and MLP output before the residual addition. The method's effective_readout[l] =
  W_out[l] @ v_{l+1} is wrong — it must pass through the frozen post-norm Jacobian. The
  circuit tracing paper used Claude 3 Sonnet (pre-norm only) and never hit this. Pythia (also
  pre-norm) works fine at 0.91. Fix: port post-norm handling from Gemma Scope 2's attribution
  tooling, or use their pre-trained CLTs which were built knowing the architecture.
