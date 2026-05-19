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
- Compute (minimum path, use existing CLT): ~$6 for graph generation on RunPod H100
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

- **Freeze** every multiplicative nonlinearity (attention pattern, LN/RMSNorm scale,
  GeGLU gate) before computing attributions. This makes the model linear in the
  residual stream and gives a well-defined chain rule.
- Nodes: CLT features, token embeddings, reconstruction errors, attention head
  cross-position contributions, output logits.
- Edges: linear effects between nodes. Feature pre-activation = sum of input edges.
- Pruning: keep top-K nodes/edges by contribution to target token logit. See §Appendix:
  Graph Pruning for the exact algorithm.
- Export format must match `anthropics/attribution-graphs-frontend` JSON schema.
  Check `frontend/README.md` for the schema spec before writing `graphs/export.py`.

### How `graphs/build.py` computes attribution gradients

ONE `torch.autograd.grad` backward pass through a frozen-nonlinearity forward
gives every per-layer gradient the edge formulas need. Implementation lives in
`_compute_attribution_gradients`. The forward installs TransformerLens hooks that:

1. **Replace every `attn.hook_pattern`** with its cached `.detach()` value.
2. **Replace every LN/RMSNorm `hook_scale`** with its cached `.detach()` value —
   `ln1`, `ln2`, `ln_final` on every architecture; `ln1_post`, `ln2_post` on
   Gemma 3. Anything new (e.g. `ln3` on a future arch) flows through automatically:
   we freeze every `*.hook_scale` that exists in the cache.
3. **Detach `hook_mlp_out`** at every layer. This severs autograd's path back
   through the MLP so `∂logit/∂hook_resid_pre[l]` flows along skip + attention
   only — matches the paper's linearised replacement model where MLPs are
   replaced by CLT decoder outputs.

After backward, three gradient signals are available for the edge formulas:

- `grad['resid_pre'][l]` for `l ∈ [0, L]` — direct from autograd. Used for
  embedding → logit edges (`embed · grad_at_resid_pre[0]`).
- `grad['attn_out'][l]` for `l ∈ [0, L-1]` — direct from autograd. Used for
  cross-position attention → logit edges (`cross_attn[h] · grad_at_attn_out[l]`).
  The post-attn-norm Jacobian is baked in for free on Gemma 3 because `ln1_post`
  is in the autograd graph with frozen scale.
- `grad['mlp_post'][l]` for `l ∈ [0, L-1]` — derived after autograd as
  `W_out.T @ ((γ_post/scale_post).detach() * grad_at_resid_pre[l+1])`. The
  manual derivation is needed because the `hook_mlp_out` detach severs
  autograd's path to `mlp_post`. Used for feature → logit and error → logit edges.

Why architecture-agnostic: post-norms, GeGLU, RoPE, GQA all flow through the
chain rule for free. Adding a new model = no new Jacobian code, just
`HookedTransformer.from_pretrained(...)`.

Sanity check: `grad_at_resid_pre[L] · resid_post[L-1] == logit - b_U` exactly
under the frozen `ln_final` scale. The build prints both numbers as
`v·r_L=... decomposable=...` — they should match to several digits.

Verified completeness on Pythia-70m at rewrite time: 0.8996 (recorded in
`docs/phase1_autograd.json`). Regression-tested in
`tests/test_attribution_completeness.py`. See `docs/autograd_plan.md` and
`docs/diagnostics_log.md` for the rewrite history and the diagnostic chain
that led to it.

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
git clone https://YOUR_TOKEN@github.com/Jules-Canada/interpretability-clin-trials.git
cd interpretability-clin-trials

# 2. Fix torchvision conflict (breaks transformer_lens import)
pip uninstall torchvision torchaudio -y

# 3. Set HF token then run setup script — it handles venv, torch, deps,
#    HF login, AND symlinks data/ + checkpoints/ to /workspace so the
#    20GB root disk doesn't fill up. Both setup_pod.sh and
#    setup_pod_medgemma.sh do this.
export HF_TOKEN=YOUR_HF_TOKEN
bash scripts/setup_pod_medgemma.sh   # or setup_pod.sh for non-MedGemma
source .venv/bin/activate            # also auto-added to .bashrc for new shells

# 4. SCP corpus + checkpoint from Mac (paths land on /workspace via symlinks
#    set up in step 3). Run from Mac:
#    scp -P <PORT> -i ~/.ssh/id_ed25519 \
#        ~/Desktop/protocol_corpus/ct_corpus/protocols.jsonl \
#        root@<IP>:interpretability-clin-trials/data/protocols.jsonl
#    scp -P <PORT> -i ~/.ssh/id_ed25519 \
#        ~/Desktop/ignis/checkpoints/medgemma-4b-1024/clt_inference.pt \
#        root@<IP>:interpretability-clin-trials/checkpoints/medgemma-4b-1024/
```

**Known gotchas:**
- `huggingface-cli login` and `hf login` are both broken on some images — `setup_pod*.sh` falls back to `hf auth login` automatically; if both fail, drop in: `python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"`
- `torchvision` must be uninstalled before importing `transformer_lens` or you get a segfault
- **Root disk is 20GB on RunPod, /workspace is huge** — never put corpus, HDF5, model checkpoints, or HF cache on root. The setup script handles this via symlinks; if you ever set up a pod manually, replicate those symlinks first or scp will fail at ~50% with "write remote: Failure"
- For graph generation only: A10 (24GB) is sufficient, no need for H100
- Repo dir is `interpretability-clin-trials` (the GitHub repo name), not `ignis` (the Mac local dir)

---

## Compute Notes

- CLT training on Pythia-410m: expect ~4–8 GPU-hours on a single A100 for a reasonable
  run. Use gradient checkpointing if VRAM is tight.
- Activation extraction: extract and cache residual streams + MLP outputs to disk first
  (`scripts/extract_activations.py`), then train CLT offline. Don't re-forward Pythia
  on every CLT training step.
- For local dev/testing, use `pythia-70m` (6 layers) — fast enough to iterate on CPU.

### Pre-termination checklist (run before killing any pod instance)

**If `run_pipeline.sh` completed fully, Step 4 already ran — skip to scp.**
**If the pipeline was interrupted, run `bash scripts/pre_terminate.sh` first.** It handles
checkpoint stripping, `collect_graph_features`, and `find_top_activations` in one shot,
then prints the exact scp commands to run. See `docs/pipeline_lessons.md` for why.

scp commands (run from your Mac):
```
INSTANCE=ubuntu@<ip>
scp "$INSTANCE:interpretability-clin-trials/frontend/graph_data/*.json" frontend/graph_data/
scp "$INSTANCE:interpretability-clin-trials/checkpoints/pythia-410m-2048/clt_inference.pt" checkpoints/pythia-410m-2048/
scp "$INSTANCE:interpretability-clin-trials/data/feature_activations.jsonl" data/
scp "$INSTANCE:interpretability-clin-trials/data/graph_features.json" data/
```

**Rule: run all 4 scp commands before terminating. The HDF5 stays on the instance and is
re-extracted next time. Use `--resid_only` for find_top_activations runs (~491GB on disk);
use full extraction (resid + mlp_post, ~2.5TB) only for CLT training — needs a larger disk.
`feature_labels.jsonl` is generated locally after scp (run `python scripts/label_features.py --resume`).**

---

## Current Status (as of 2026-05-18)

### What works

- **14 PT clinical graphs** in `frontend/graph_data/` — completeness 0.55–0.81 (12/12
  clinical prompts above the 0.5 threshold; france_capital 0.47, water_boil 0.71).
  Generated on MedGemma-4B-pt with the autograd attribution pipeline.
- **148 features extracted**, 358 labels in `data/feature_labels.jsonl`, 2960 top-activation
  examples in `data/feature_activations.jsonl`. All from the 14 PT clinical graphs.
- **Autograd attribution pipeline** (`graphs/build.py`) — architecture-agnostic, handles
  Gemma 3 post-norms. Tested on Pythia-70m (0.90) and MedGemma-4B (0.55–0.81).
  Regression-locked in `tests/test_attribution_completeness.py`.
- **Contrastive readout** implemented in `graphs/build.py` — `contrastive=(pos_ids, neg_ids)`
  parameter for binary-decision targets. Untested at graph level (see "What doesn't work").
- **Contrastive screening** (`scripts/screen_contrastive.py`) — measures `logit(Yes) - logit(No)`
  per prompt and pair separation gap. Results in `data/contrastive_screen_*.json`.
  Weak statistical signal: easy prompts have larger mean gap than categorical (0.68 vs 0.12),
  but ranges overlap and 5/18 categorical pairs don't separate. Not a clean result.
- **PhysioNet credentialing** for MIMIC-IV (granted 2026-05-08).

### What doesn't work yet

- **Contrastive graphs** — CLT L0~91 projects ~0.55-logit constant No-bias onto the Yes-No
  direction. For single-token targets (logit ~20+) this is <3% noise; for contrastive
  targets (logit-diff ~0.2) it's >100% noise. Needs sparser CLT (L0~20-30).
- **IT model CLT** — trained on raw corpus activations, OOD at inference on short prompts.
  Completeness 0.20–0.47 (all below threshold). Artifacts moved to `deferred/phase6_it/`.
  IT path is not ruled out methodologically but needs format-matched training data.
- **Track B PT graphs** (7 medical-factual cloze, completeness 0.76–0.90) — were generated
  on a pod but overwritten locally by the IT experiment. Need regeneration (~$6 pod, ~1hr).
- **Notebook 03** — exists but references `_compute_readout_vector` (deleted in autograd
  cutover). Needs rewrite after Track B graphs are available.

### Next steps (pick one path)

1. **Finish what we have (no new CLT, ~$6 compute):**
   Regenerate Track B PT graphs on pod → run `find_top_activations` for new features →
   `label_features.py --resume` → write notebook 03 → push toward publication.
   Paper scope: "CLT attribution on a medical model, architecture-agnostic pipeline,
   feature labeling on medical-factual cloze prompts."

2. **Retrain sparser CLT first (~$50 compute, ~10hrs):**
   Retrain at L0~20-30 → regenerate all graphs (14 clinical + 7 Track B) with sparser CLT →
   retry contrastive graphs on the 13 separating categorical pairs → then label + notebook.
   Paper scope: adds contrastive eligibility analysis if it works.

### Completed milestones (detail in git history)

- Phase 1: Pythia-410m proof-of-concept (CLT + 14 graphs + feature labels + notebook 02)
- Phase 2-3: Autograd attribution rewrite (post-norm fix, architecture-agnostic)
- Phase 4: MedGemma-4B-pt batch run (14 clinical graphs, completeness validated)
- Phase 5: Contrastive methodology (screening script, Track B validation)
- Phase 6: IT model attempt (failed — OOD mismatch, moved to `deferred/phase6_it/`)

## Findings So Far

- **The circuit tracing paper used an instruction-tuned model (Claude 3 Sonnet), not a base model.**
  IT models are valid targets, but the CLT training data must match the inference format.
  Anthropic trained on the same distribution Claude processes at inference. Our IT attempt
  (Phase 6) failed because the CLT was trained on raw corpus activations, which are OOD for
  short QA prompts. PT models don't have this problem: raw text IS the inference format.
- `sparsity_coeff=2e-4` is too weak — reconstruction dominates and L0 saturates near n_features.
  Updated default to `1e-2`. L0 still high at 500 steps on 50k tokens; expect improvement at scale.
- Activation extraction uses `monology/pile-uncopyrighted` streamed from HuggingFace.
  Requires `zstandard` for decompression. Default slice: 50k tokens for dev.
- Training loop is model-agnostic via `ActivationLoader` protocol — switching models
  only requires a new loader, not changes to `clt/train.py`.
- CLT must always be moved to the same device as the model it's paired with.
  Call `clt.to(next(model.parameters()).device)` at entry points (`build_attribution_graph`,
  test fixtures). Never scatter `.to(device)` calls on individual tensors inside helpers.
- Attribution graph completeness must include attention paths: an MLP-only T matrix
  gives ~0.001 because attention dominates logit prediction in both Pythia and MedGemma.
  The current `graphs/build.py` handles this correctly via the autograd backward pass
  through the linearised model — see "Attribution Graph Facts" above. Phase 1 Pythia
  graphs (pre-autograd) have the MLP-only bug and must NOT be cited as valid; rebuild
  if needed. Pythia-70m baseline under the autograd build is 0.8996.
- H100 training speed: ~1.37 steps/s with batch_size=512, n_features=2048, 24 layers. 50k steps ≈ 10hrs.
  n_features=4096 exceeded H100 VRAM (81GB needed vs 79GB available) — settled on 2048.
- steps/sec timing added to `_log()` in `clt/train.py` (elapsed, eta, steps/s).
- HDF5 random sampling caused 0% GPU utilization (512 random seeks per step). Fixed by sampling
  contiguous blocks instead — critical when chunk size is 1024 tokens.
- HDF5 now stores `token_ids` dataset (int32) for feature labeling context reconstruction.
  Old HDF5 files without this field need to be re-extracted before running label_features.py.
- HDF5 is self-describing: `extract_activations.py` stamps `attrs["model_name"]`.
  `find_top_activations.py` / `fix_feature_activations_tokenizer.py` read it via
  `scripts/_tokenizer_resolve.py` and pick the decode tokenizer from it — **normally
  do NOT pass `--model_name`**. Passing it asserts against the attr (mismatch = hard
  error; this is how a wrong tokenizer is caught now, not just an omitted one).
  `--model_name` is required only for pre-attr legacy HDF5s (e.g. the Phase 4
  MedGemma dump) — those have no recorded truth so a wrong value there is still
  silent; pass the matching model explicitly. Root-caused from the Phase 4 bug
  where a Pythia default tokenizer silently decoded MedGemma ids into garbage.
- flush_every default changed 500→5 to prevent ~200GB RAM accumulation before first disk write.
- torchvision/torchaudio conflict on pods: pins torch==2.5.1, incompatible with torch 2.11.0.
  Removed from setup_pod.sh; uninstall manually on existing instances.
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
  to ~400GB for 2M tokens (resid + mlp_post, 34 layers). Use 1TB pod volume.
- Clinical trial protocol corpus: 49,002 docs from ClinicalTrials.gov, avg ~26k tokens/doc,
  JSONL with `full_text` field. Use --text_field full_text with extract_activations.py.
  2M tokens covers ~77 documents — sufficient for a proof-of-concept CLT run.
  Local copy on Julie's Mac: `/Users/juliecannon/Desktop/protocol_corpus/ct_corpus/protocols.jsonl`
  (6.0 GB). Not in the repo — scp this up to `data/protocols.jsonl` on every fresh pod.
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
- **RMS scale persistence (2026-04-30).** The CLT trains with dataset-level per-layer RMS
  scales computed once at loader init from a 4096-token sample of the HDF5
  (`clt/loader.py:_compute_scales`). Until 2026-04-30 these scales were never persisted —
  `_save_checkpoint` wrote only `model_state_dict` + `optimizer_state_dict`. At inference
  `graphs/build.py` fell back to per-prompt RMS, which drifts wildly across prompts and
  inflates feature contributions by 5-10×, breaking the feat/error cancellation. Symptom:
  on MedGemma, the steroid test prompt happened to land near training-time RMS and gave
  completeness=0.76, but eligibility prompts gave completeness=4.14 / 8.14 (unphysical).
  Recovery path: `scripts/compute_clt_scales.py` reads the HDF5, computes per-layer RMS
  over a 100k-token sample, and writes `resid_scales` + `mlp_scales` into the existing
  checkpoint. `clt/model.py` registers non-persistent buffers populated post-load via
  `clt.load_scales_from_checkpoint(ckpt)`. `graphs/build.py` prefers saved scales and
  warns + falls back to per-prompt only if a checkpoint predates this change. **TODO**:
  update `clt/train.py:_save_checkpoint` to bundle scales automatically so future training
  runs (e.g. the planned L0~20-30 retrain) don't recreate this gap.
- **Contrastive readout for binary decisions (2026-05-16).** `graphs/build.py` now accepts
  a `contrastive=(pos_ids, neg_ids)` parameter. Target becomes
  `mean(logit[pos_ids]) − mean(logit[neg_ids])` — same autograd backward, same completeness
  check, same edge formulas. Features pushing equally toward both answers cancel by
  construction, decontaminating scaffold/format artifacts from the graph. Standard approach
  for binary-decision circuit analysis (IOI, sparse feature circuits).
- **Contrastive graphs require sparser CLTs than single-token graphs.** At L0~91, the CLT
  projects a constant ~0.55-logit No-bias onto the Yes−No direction regardless of prompt
  content. For single-token targets (logit ~20+) this is <3% noise and completeness is
  0.76–0.90. For contrastive targets (logit-diff ~0.2) the same absolute noise is >100% →
  completeness is meaningless. Root cause: thousands of dense feature contributions must
  cancel precisely to hit a near-zero target; at L0~91 they don't. Fix: retrain CLT at
  L0~20–30, reducing active features ~4× and correspondingly reducing cancellation noise.
- **Contrastive screening shows weak signal, not clean discrimination.**
  `medgemma-4b-pt` with the `Eligible?\nAnswer:` scaffold showed no dynamic range on
  absolute p_agg (easy controls ≈ categorical ≈ 0.3). On the contrastive metric
  (`logit(Yes) - logit(No)`), easy prompts have larger mean gap than categorical
  (0.677 vs 0.123), but the ranges overlap substantially — some easy gaps (0.15) are
  smaller than some categorical gaps (0.58), and 5/18 categorical pairs don't separate
  at all. The contrastive metric reveals more structure than absolute-p, but does not
  cleanly discriminate difficulty.
- **Gemma tokenizer splits many medical terms** (imatinib, trastuzumab, protamine, Hodgkin,
  troponin) into multi-token sequences. Single-token medical targets that work: metformin,
  pancreas, heart, embolism, plexus, gallbladder, K. Design prompts around these.
