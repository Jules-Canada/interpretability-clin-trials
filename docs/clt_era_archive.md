# CLT-Era Archive (superseded background)

This file holds reference material from the from-scratch-CLT era that is no longer
load-bearing on the active path (circuit-tracer + Gemma Scope 2 per-layer transcoders —
see `CLAUDE.md` Phase 2 Models and `docs/ignis_approach_handoff.md`).

It is kept for: regression-test history, the diagnostic chain that produced the autograd
attribution build, and any future custom-CLT ("MedScope") work. Nothing here describes the
current active path — treat it as history. Dates are preserved.

---

## How `graphs/build.py` computed attribution gradients (retired pipeline)

> Superseded: the active path uses Anthropic's `circuit-tracer` library, which replaces
> `graphs/build.py`. This documents the autograd build that circuit-tracer retires.

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

## CLT training / extraction / infra findings (retired)

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
  The autograd `graphs/build.py` handled this via the backward pass through the linearised
  model. Phase 1 Pythia graphs (pre-autograd) have the MLP-only bug and must NOT be cited as
  valid; rebuild if needed. Pythia-70m baseline under the autograd build is 0.8996.
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
- HDF5 size for 5M tokens, 24 layers: ~2.5TB (resid + mlp_post, float32) or ~491GB (resid only).
  The "~20GB" estimate was wrong. A10 instances have 1.4TB disk — only fits resid_only. Use
  `--resid_only` flag for find_top_activations runs; full extraction needs H100 or dedicated storage.
- MedGemma-4B-pt CLT config: n_features=1024 chosen to fit H100 VRAM (34 layers, d_mlp=10240
  GeGLU, decoder matrix is O(L*(L+1)/2 × n_features × d_mlp)). float16 storage reduces HDF5
  to ~400GB for 2M tokens (resid + mlp_post, 34 layers). Use 1TB pod volume.
- **Gemma 3 post-norms break attribution completeness.** With our MedGemma CLT, v·r_L = 21.5
  (v is correct) but completeness ≈ −0.002. Root cause: Gemma 3 applies RMSNorm after each
  attention and MLP output before the residual addition. The method's effective_readout[l] =
  W_out[l] @ v_{l+1} is wrong — it must pass through the frozen post-norm Jacobian. The
  circuit tracing paper used Claude 3 Sonnet (pre-norm only) and never hit this. Pythia (also
  pre-norm) works fine at 0.91. This is the diagnostic that produced the autograd rewrite.
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
  warns + falls back to per-prompt only if a checkpoint predates this change.
  **DONE (2026-06-07):** `_save_checkpoint` now bundles `resid_scales` and `mlp_scales`
  from the loader into the checkpoint dict. No more post-hoc `compute_clt_scales.py` step.
- **`find_top_activations.py` missing RMS normalization (fixed 2026-06-01).** The script
  loaded the CLT state dict but never called `clt.load_scales_from_checkpoint(ckpt)`, and
  fed raw (unnormalized) HDF5 residuals into `clt.encode()`. The CLT was trained on
  RMS-normalized inputs, so every pre-activation was off-scale → JumpReLU gated all
  features to zero → 0 activations for all 110 graph features. Symptom: `label_features.py`
  produced plausible-sounding labels ("table of contents dots", "section numbers") from
  empty activation lists — the LLM hallucinated labels with no grounding data. Fix: (1)
  call `load_scales_from_checkpoint` in `load_clt()`, (2) divide each residual batch by
  `clt.resid_scales[l]` before encoding. Any script that calls `clt.encode()` on raw HDF5
  data must normalize first.
- **Contrastive readout for binary decisions (2026-05-16).** `graphs/build.py` accepted
  a `contrastive=(pos_ids, neg_ids)` parameter. Target became
  `mean(logit[pos_ids]) − mean(logit[neg_ids])` — same autograd backward, same completeness
  check, same edge formulas. Features pushing equally toward both answers cancel by
  construction, decontaminating scaffold/format artifacts from the graph. Standard approach
  for binary-decision circuit analysis (IOI, sparse feature circuits). The *finding* that
  contrastive targets need sparser dictionaries carries over to the active path — kept in
  CLAUDE.md.
