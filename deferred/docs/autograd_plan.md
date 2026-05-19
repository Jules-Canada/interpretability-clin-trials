# Autograd Attribution Rewrite — Plan

Replace the manual chain-rule computation in `graphs/build.py` with a torch
autograd backward pass against a frozen forward pass. Same math the paper
specifies, far less bespoke code, architecture-agnostic.

This document is the implementation plan. Background, justification, and the
debugging arc that led here are in `docs/diagnostics_log.md`.

---

## Goals

**Primary:** completeness ≥ 0.5 on MedGemma 4B for all 14 clinical prompts,
without per-prompt manual tweaking.

**Secondary:** completeness ≈ 0.91 on Pythia-70m preserved (regression check).

**Tertiary:** the attribution code is small enough that adding the *next*
model (Llama, Qwen, etc.) requires zero new Jacobian code.

**Out of scope:** changing the graph schema, frontend, CLT training, or any
of the per-feature breakdown logic except where it consumes attribution
gradients.

---

## What gets replaced vs kept

### Replaced

`graphs/build.py` functions to delete (or stop calling):

- `_compute_attention_propagated_v` — manual back-propagation of v through
  attention Jacobians and LN/RMSNorm Jacobians, layer by layer
- `_compute_corrected_logit_transfer` — manual computation of effective
  readout vectors per layer in mlp_post space, including post-MLP norm
- The post-attn-norm Jacobian block at line ~885 in the cross-position
  attention section

### Kept

- `_compute_readout_vector` — the initial v at `r_L`. We still need this as
  the gradient seed for autograd. Same formula.
- `_compute_transfer_matrices` — feature-to-feature transfer matrices for
  feat→feat edges. Still uses W_dec @ W_out structure, but plugs in
  autograd-derived gradient at `hook_mlp_post` instead of
  manually-computed effective_readout.
- All node/edge construction, pruning, export — completely unchanged.
- The CLT (encoder/decoder/scales) — completely unchanged.
- The saved-scale path in build.py — already correct, stays.

### New code

One new function: `_compute_attribution_gradients(model, cache, target_position, target_token_idx, L)`. Returns a dict:

```
{
    'resid_pre':  list[L]    of (d_model,) — ∂target_logit/∂hook_resid_pre[l]
    'mlp_post':   list[L]    of (d_mlp,)   — ∂target_logit/∂hook_mlp_post[l]
    'attn_out':   list[L]    of (d_model,) — ∂target_logit/∂hook_attn_out[l]
}
```

All gradients computed via one autograd backward pass against a frozen forward.

---

## Architecture

The frozen forward pass replaces dynamic nonlinearities with cached values
via TransformerLens hooks. This makes the resulting computation linear
(under the frozen approximation) so PyTorch autograd computes the exact
chain rule the paper specifies.

### What gets frozen (replaced with cached `.detach()` value)

1. **Every LN/RMSNorm `hook_scale`** — `ln1`, `ln2`, `ln1_post`, `ln2_post` per
   layer, plus `ln_final.hook_scale`. ~4 × 34 + 1 = 137 hooks for MedGemma.
2. **Every `attn.hook_pattern`** — frozen attention weights per layer (34 hooks).
3. **GeGLU gate values** — Gemma 3 uses GeGLU: `mlp_post = gelu(W_gate @ x) * (W_up @ x)`. We freeze the `gelu(W_gate @ x)` factor at its cached value so the MLP becomes linear in the residual. The hook to use is `mlp.hook_pre_linear` (the pre-activation gate value) — confirm via probe; may need to swap the multiplication via a custom hook on `mlp.hook_pre`.
4. **JumpReLU thresholds in the CLT** — already not in the autograd path (CLT is queried as a non-differentiable function, returning feature activations as constants). Nothing to do here.

### What stays differentiable (autograd computes Jacobian through it)

- All matmul weights: `W_Q, W_K, W_V, W_O, W_in, W_out`, MLP gate weights, embed/unembed.
- All adds: skip connections, attention output adds to residual, MLP output adds to residual.
- The post-norm scaling: `output = x * w / scale` — `w/scale` becomes a
  constant per-element (because scale is detached), so this is a linear
  diagonal map that autograd handles trivially.

### Why this is exactly the paper's spec

The paper's "interpretable replacement model" linearizes the model by
freezing all multiplicative nonlinearities (attention pattern, LN denominator,
MLP gate). Once frozen, every remaining op is linear, and the chain rule
gives unique attribution. Autograd against this frozen forward = computing
that chain rule exactly. No approximation beyond the freezing the paper
already requires.

### Per-feature accounting (unchanged in form)

Today:
```
weight(feat_l_s_f) = a_value[f] * Σ_{l_t ≥ l_s} mlp_rms[l_t] * decoder[l_s→l_t][:, f] · effective_readout[l_t]
```

After:
```
weight(feat_l_s_f) = a_value[f] * Σ_{l_t ≥ l_s} mlp_rms[l_t] * decoder[l_s→l_t][:, f] · grad_at_mlp_post[l_t]
```

Where `grad_at_mlp_post[l_t]` is the autograd-derived gradient at
`hook_mlp_post[l_t]`, in d_mlp. Same formula, autograd-derived input.

Error contribution: `error[l] · grad_at_mlp_post[l]` — same.
Attn contribution: `cross_attn[h] · grad_at_attn_out[l]` — same.
Embed contribution: `embed_vec · grad_at_resid_pre[0]` — same.

---

## Implementation phases

### Phase 0 — Re-baseline Pythia (~1 hour)

**Goal:** confirm current `build.py` still gives Pythia-70m completeness ≈ 0.91 before we change anything.

**Steps:**
1. On laptop or pod: `pytest tests/test_clt_toy.py -v` — confirms basic plumbing.
2. Run a single pythia-70m attribution graph (need a saved 70m CLT checkpoint or train a fresh one — the toy model fixture should work).
3. Verify `completeness ≥ 0.85`.

**If completeness < 0.85:** the post-norm work broke Pythia. Revert or fix before proceeding. **Do not** start the autograd rewrite without a working baseline.

**Deliverable:** a recorded completeness number for Pythia-70m on a fixed prompt. This becomes the regression check for Phase 2 tests.

### Phase 1 — Build `graphs/build_autograd.py` (~1.5–2 days)

**Goal:** new attribution implementation that uses autograd, alongside the existing `build.py` for A/B comparison.

**Steps:**

1. **Copy** `graphs/build.py` to `graphs/build_autograd.py`. Same exports (`build_attribution_graph` with same signature). Lets us swap one import line in `scripts/run_graph.py` to flip between them.

2. **Write `_compute_attribution_gradients`** in the new file:
   - Cache the original forward pass (already done by existing code via `model.run_with_cache`).
   - Build a fwd-hook list that replaces `hook_pattern`, every `*norm.hook_scale`, and the GeGLU gate factor with the cached detached value.
   - Set `model.eval()` and `requires_grad_(False)` on all parameters.
   - Need `requires_grad_(True)` on the input embedding tensor or `hook_resid_pre[0]` so gradient can be retained at every intermediate residual.
   - Use `value.retain_grad()` on every hook_resid_pre, hook_mlp_post, hook_attn_out value during the second forward.
   - Run forward with `m.hooks(fwd_hooks=...)` context manager.
   - Compute `target_logit = logits[0, target_position, target_token_idx]`.
   - Call `target_logit.backward()`.
   - Read `.grad` off each retained tensor; package into the return dict.
   - All on CUDA float64 (we have an A100, so float64 doesn't crater speed).

3. **Replace usage sites in `build_autograd.py`:**
   - Delete the `_compute_attention_propagated_v` and `_compute_corrected_logit_transfer` calls.
   - Replace `effective_readouts[l]` everywhere it's used with `grads['mlp_post'][l]`.
   - Replace `v_at_layer[l]` everywhere with `grads['resid_pre'][l]`.
   - Replace the cross-attn `v_for_cross` block with `grads['attn_out'][l]`.
   - Embed contribution becomes `embed_vec · grads['resid_pre'][0]`.
   - Error contribution: `error[l] · grads['mlp_post'][l]`.

4. **Validate on Pythia-70m:**
   - Same prompt as Phase 0.
   - Expected: completeness within ±0.02 of the Phase 0 baseline.
   - If wildly different: bug in autograd setup. Debug before moving on. Most likely culprits: missing hook (an LN scale or GeGLU gate not frozen), wrong hook order, gradient not retained on the right tensor.

5. **Validate on MedGemma:**
   - Eligibility prompt that gave completeness = 4–8 with manual code.
   - Expected: completeness in [0.5, 1.2].
   - If still inflated: the frozen-rms approximation is genuinely breaking down for Gemma 3. Different problem; falls back to "Phase 5 contingency" below.

**Risks in this phase:**

- **GeGLU gate freezing.** The MLP nonlinearity in Gemma 3 needs the gate frozen for the forward to be linear in r_l. TransformerLens may not expose the right hook directly — we may need to install a hook that intercepts `hook_pre_linear` and freezes it. Probe ahead with a small script.
- **`hook_resid_pre` may not be a leaf** in the second forward pass (gradient won't flow into it). May need to clone+detach and replace the value via a hook, then track that as the leaf for backward.
- **Hook ordering.** TransformerLens runs hooks in registration order. If we register both a "freeze" hook and a "retain" hook on the same point, order matters. Test carefully on the toy model first.
- **Memory.** Autograd retaining all intermediate activations doubles memory. MedGemma 4B at fp64 may be tight on a 24GB card; use A100 80GB or lower precision (fp32 + Kahan summation if needed).

**Deliverable:** `graphs/build_autograd.py` that produces the same Pythia-70m result and reaches MedGemma completeness ≥ 0.5.

### Phase 2 — Regression tests (~half day)

**Goal:** lock in the new behavior so it doesn't silently break later.

**Steps:**

1. New file: `tests/test_attribution_completeness.py`. Three test cases:
   - **Toy 2-layer model** (existing fixture in `test_clt_toy.py`): completeness within ±0.02 of a recorded value.
   - **Pythia-70m + clinical prompt**: completeness ≥ 0.85.
   - **MedGemma-4b + eligibility prompt**: completeness ≥ 0.5 (or whatever Phase 1 actually achieved minus 0.05 buffer).
2. Tests use `build_autograd.build_attribution_graph` (the new function).
3. Add to CI if/when CI is set up; for now, runnable locally.
4. **Importantly**: the MedGemma test requires a checkpoint and significant compute. Mark it `@pytest.mark.slow` so it's opt-in — but it's the test that actually matters most, so document running it before any future changes to attribution code.

**Deliverable:** `tests/test_attribution_completeness.py` with three passing tests.

### Phase 3 — Cutover (~1–2 hours)

**Goal:** make the autograd path the default; retire the manual code.

**Steps:**

1. Confirm Phase 1 + 2 are green for at least 24 hours (no immediate regressions found).
2. Either:
   - **(a)** Move `build_autograd.py` → `build.py` (overwriting the old file), keeping git history via single commit.
   - **(b)** Update all callers (`scripts/run_graph.py`, `scripts/run_graphs_batch.py`, tests) to import from `build_autograd` and delete `build.py`.
3. Update `CLAUDE.md`'s "Attribution Graph Facts" section to reflect the new approach: remove the long "Attention Jacobians — REQUIRED" block (it's still true mathematically but the implementation is now autograd-based, not manual).
4. Update Phase 2 status checklist in `CLAUDE.md`.

**Deliverable:** single attribution implementation, manual chain-rule code deleted from main.

### Phase 4 — Re-run 14 clinical graphs on MedGemma (~1 day)

**Goal:** the actual Phase 2 milestone — clinical attribution graphs that we trust.

**Steps:**

1. Pod: `git pull && python scripts/compute_clt_scales.py ...` (already done; checkpoint has saved scales)
2. Pod: `python scripts/run_graphs_batch.py --checkpoint_dir ... --prompts_file prompts/trial_prompts.json` — but using the new build_autograd path.
3. Verify all 14 prompts give completeness ≥ 0.5.
4. scp graph JSONs back to laptop.
5. Run `scripts/find_top_activations.py` to label features (re-uses HDF5 if still on pod).
6. Run `scripts/apply_labels.py` to bake labels into graph JSONs.
7. Spot-check 2–3 graphs in the frontend for sensible structure (medical features at relevant layers, sane edge weights, etc.).

**Deliverable:** 14 attribution graphs with completeness ≥ 0.5 and feature labels, ready for Notebook 03.

### Phase 5 — CLT retrain (CONTINGENT, ~$50 + 10–12 hr H100)

**Trigger:** if Phase 4 graphs are too noisy for meaningful feature labels even with completeness in target range. Likely cause: trained CLT has L0 ≈ 91 (paper's interpretable graphs use L0 ~ 20–30).

**Steps:**

1. Spin up H100 (RunPod).
2. Re-extract MedGemma activations if HDF5 isn't preserved.
3. Train new CLT with `sparsity_coeff` increased to target L0 ≈ 25.
   - Estimate based on current `sparsity_coeff=1e-2` giving L0~91: try `sparsity_coeff=4e-2` first, iterate.
4. Run `compute_clt_scales.py` against new checkpoint as part of training (or as separate post-step, but ideally in `clt/train.py:_save_checkpoint` per the existing TODO in CLAUDE.md).
5. Re-run Phase 4.

**Deliverable:** new CLT checkpoint at `checkpoints/medgemma-4b-1024-sparse/clt_inference.pt`, possibly higher reconstruction MSE but interpretable.

### Phase 6 — Notebook 03 (~half day)

**Goal:** medical-features readout for the project's stated objective.

Standard notebook deliverable per CLAUDE.md "Every milestone gets a notebook" rule.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Autograd setup has subtle bug (wrong hook, wrong tensor leaf) | High at start | Medium | Validate on toy model first; require Pythia-70m to match baseline before MedGemma run |
| Frozen-rms approximation genuinely breaks for Gemma 3 with γ in hundreds | Medium | High | Falls back to: (a) accept lower completeness as ceiling, (b) try a different freeze regime, (c) try Gemma Scope 2's pre-trained 270M/1B CLTs as an alternative model |
| Memory on autograd backward exceeds GPU | Medium | Low | Use A100 80GB; if needed, gradient checkpointing or compute backward in fp32 not fp64 |
| GeGLU gate hook isn't where I think it is in TL | Medium | Low | Probe with diagnostic script before relying on it |
| Pythia regression — autograd version doesn't match 0.91 | Medium | High | Stop and debug before MedGemma; do not push forward with broken baseline |
| Phase 4 graphs are too noisy → Phase 5 needed | Medium | Time/budget | $50 budget exists per memory; ~12hr H100; one-time cost |

---

## Time estimate (generous; assumes "no shortcuts")

| Phase | Work | Estimate |
|---|---|---|
| 0 | Pythia re-baseline | 1 hour |
| 1 | Autograd implementation + Pythia/MedGemma validation | 1.5–2 days |
| 2 | Regression tests | half day |
| 3 | Cutover + CLAUDE.md updates | 1–2 hours |
| 4 | Re-run 14 clinical graphs + label features | ~1 day |
| 5 | (contingent) CLT retrain | ~1 day work + 12hr compute + $50 |
| 6 | Notebook 03 | half day |

**Total without Phase 5:** ~3–4 days of focused work.
**Total with Phase 5:** ~5 days + $50.

---

## Open questions to revisit during implementation

1. **Does the CLT need to be re-evaluated against ln_pre vs ln_post inputs?** The CLT was trained reading `hook_resid_pre`, but for Gemma 3 the MLP at layer l actually reads `ln2(r_l_mid)`, not `ln1(r_l)`. Possible the CLT input doesn't match what the model's MLP sees. Worth confirming after Phase 1 lands.
2. **Should `_compute_readout_vector` itself become autograd-based?** Currently it's manual frozen-LN. For consistency we could have autograd handle it too — backward seed becomes `1.0` at the target logit, autograd computes everything down to `hook_resid_pre[0]`. Cleaner, possibly more correct.
3. **Cross-position attention as a separate node type or unified?** The current code creates explicit `attention` nodes for cross-position contributions. Autograd would naturally include cross-position contributions in the gradient at hook_attn_out. We need to make sure we don't double-count: subtract the same-position self-loop as today.
4. **Per-feature feat→feat edges:** still computed via T-matrices. Could be migrated to autograd later but isn't in scope for completeness.
