# MedGemma Attribution Completeness — Diagnostics Log

Notes from the debugging arc that started 2026-04-30 when the MedGemma post-norm fix
landed but completeness blew up to 4–8 (unphysical, expected ≤ ~1.0). This log
records the diagnostic scripts we wrote, what each ruled in or out, and the
state of our understanding when we paused to plan the autograd rewrite.

The bug was **not** what the first hypothesis claimed. We found three real but
distinct problems and one fundamental limitation. None of the manual fixes brought
completeness into the target range.

---

## Timeline

- **2026-04-27** — Manual attention Jacobians + frozen-LN v implemented in `graphs/build.py`. Pythia-70m completeness = 0.91. Shipped.
- **2026-04-28** — Tested on MedGemma. Completeness ≈ −0.002. Diagnosed Gemma 3 post-norms as missing from manual Jacobian. Plan: add post-norm γ/scale corrections.
- **2026-04-29** — Post-attn norm Jacobian for cross-position attention edges added (commit 87a6b9b). Steroid test prompt: completeness 0.39 → 0.76. Encouraged.
- **2026-04-30 (batch)** — Ran 14 prompts. Eligibility / dose-reduction prompts: completeness 4.14 / 8.14. Wrong direction. Hypothesis: per-prompt RMS drift.
- **2026-04-30 (fix)** — `compute_clt_scales.py` + saved-scale buffer in CLT model + build.py path to use them (commit 3b5125b). Theory: drifting per-prompt RMS inflates contributions; fix by pinning to dataset-level scales recovered from HDF5.
- **2026-05-02** — Re-ran with saved scales. Completeness still 6–8. The "fix" addressed nothing. Started the diagnostic chain below.
- **2026-05-03** — Autograd Jacobian comparison confirmed manual `_compute_attention_propagated_v` produces something different from the true chain-rule gradient in non-trivial ways. Paused to plan.

---

## Diagnostic 1 — `scripts/compute_clt_scales.py`

**Not strictly a diagnostic; the first attempted fix.** Reads the training HDF5,
computes per-layer RMS scales over a sample, writes them into the existing CLT
checkpoint under `resid_scales` / `mlp_scales`. Companion buffer wiring in
`clt/model.py` and a saved-scale code path in `graphs/build.py`.

**Theory tested:** completeness inflation comes from per-prompt RMS drift —
each prompt has different activation magnitudes, so the encoder sees different
input distributions across prompts.

**Result on MedGemma after applying:** completeness still 6.0 / 8.0 across
prompts. Saved scales did not help.

**What it ruled out:** per-prompt RMS drift is **not** the dominant cause of
completeness inflation on MedGemma. Saved-scale path is now the default
because it's still architecturally correct (matches training-time
normalization), but it doesn't fix the symptom.

---

## Diagnostic 2 — `scripts/diag_scales.py`

Compares per-prompt resid/mlp RMS measured at runtime against the saved
dataset-level scales for the same layers.

**Findings on MedGemma 4B (eligibility prompt):**
- L0: per-prompt and saved match (ratio 1.00)
- L5–L33: saved scales are **2–5× smaller** than per-prompt RMS at middle/late layers
- L11 mlp_post: saved scale 0.13, per-prompt 21.3 (ratio 0.01)

**Interpretation:** the protocol corpus we sampled has different per-element
activation magnitudes at middle layers than the short eligibility-style
prompts we test on. The CLT was trained on full clinical protocols (long,
structured). Test prompts are 12–15-token clinical sentences. Activation
distributions diverge.

**What it ruled in:** the test-time activation distribution is genuinely
different from training. The CLT may operate slightly out-of-distribution at
inference. But this alone does not produce 8× completeness inflation —
input-distribution shift would change feature firing patterns, not break the
arithmetic identity that contributions should sum to the logit.

---

## Diagnostic 3 — `scripts/diag_rmsnorm.py`

Tests whether Gemma 3's RMSNorm uses `output = w * normalize(x)` or the
HuggingFace convention `output = (1 + w) * normalize(x)`. Compares both
candidate formulas to the actual `r_post - r_mid` delta at L17.

**Findings:**
- ln2_post.w values are **wildly larger than expected** for a normalization weight: L17 mean=34, max=367; L33 mean=171, max=1369
- Both candidate formulas miss the actual delta by ~19M — meaning neither matches
- Required follow-up to figure out the actual structure

**What it ruled out:** the `(1+w)` vs `w` convention is not the bug, because
neither matches. TransformerLens stores something we don't yet understand.

---

## Diagnostic 4 — `scripts/diag_rmsnorm2.py`

Lists every hook at one layer, dumps the actual `RMSNorm.forward` source, and
tries five candidate reconstructions of `r_post - r_mid`.

**Findings:**
- `RMSNorm.forward` source: `(x / scale) * w` — the simple, naive form. No `(1+w)`.
- `delta == hook_mlp_out` matches with max error 0.0005 — i.e., `hook_mlp_out`
  in TransformerLens for Gemma 3 **already contains the post-normed value**.
  The post-norm has been applied before the value is exposed at the hook.
- `hook_scale` at L17 target = 0.0649; rms(mlp_out) = 74.03 — confirms hook_scale
  is the rms of the *input* to post-norm (= `W_out @ mlp_post`), not of the
  output.

**What it ruled in:** the chain rule formula in `_compute_corrected_logit_transfer`
(eff = W_out @ (γ_post/scale_post * v)) is **mathematically correct in form**.
Code uses the right factor. The numbers are extreme but legitimate: Jacobian
factor γ/scale ≈ 524 per layer with these weights.

**What it ruled out:** there is no off-by-one or wrong-formula bug in the
post-MLP norm Jacobian itself. The remaining suspect is order of operations
in `_compute_attention_propagated_v` and/or how the manual chain rule
accumulates across layers.

---

## Diagnostic 5 — `scripts/diag_jacobian.py`

The definitive test. Replays the model with frozen attention patterns and
frozen LN/RMSNorm scales using TransformerLens hooks, then runs torch
autograd on `(v · r_L)` to get the true chain-rule gradient at every
`hook_resid_pre`. Compares to the manual `_compute_attention_propagated_v`
output layer by layer (cosine similarity + relative magnitude diff).

**Findings on MedGemma (eligibility prompt):**

| Layer | Manual norm | Autograd norm | Cosine | Rel diff |
|------:|------------:|--------------:|-------:|---------:|
|     0 |     9.93e-3 |       1.04e+0 | -0.02  |    1.000 |
|     5 |     5.96e-3 |       2.10e-1 | +0.02  |    1.000 |
|    11 |     6.01e-3 |       6.51e-2 | +0.04  |    1.001 |
|    17 |     6.23e-3 |       2.77e-2 | +0.18  |    0.984 |
|    23 |     5.99e-3 |       9.41e-3 | +0.41  |    0.940 |
|    29 |     5.75e-3 |       6.46e-3 | +0.52  |    0.932 |
|    33 |     6.26e-3 |       6.23e-3 | +0.88  |    0.489 |

**Interpretation:** at the top of the network the manual code is mostly right
(cos 0.88 at L33). Going deeper, manual and autograd progressively *diverge in
direction*, and at the very bottom they're orthogonal (cos −0.02).

The manual function is documented as computing only the **skip + same-position
attention** path, by design — it deliberately excludes MLP paths because MLP
contributions are accounted for separately via per-feature edges. So a
difference from full autograd is expected. The question is whether that
particular skip-only quantity is what build.py's contribution accounting
actually needs.

**What it ruled in:** the manual computation is producing a quantity that
diverges meaningfully from the full chain rule. We cannot easily verify it's
the *right* quantity by inspection, and the documentation (comments in the
code) doesn't unambiguously prove it either. Every Gemma-3-specific patch
we add (post-attn-norm, post-mlp-norm, GQA handling) compounds the surface
area we'd need to formally re-derive.

**What it ruled out (or strongly suggests):** there is no single-line patch
that closes the gap. The class of bugs in this code path is "subtle
chain-rule errors that are easy to make and hard to verify."

---

## Synthesis

Three real findings, one structural takeaway:

1. **Saved scales are now correct but didn't fix completeness** (Diag 1). Per-prompt RMS drift was not the bug. Saved-scale path stays as the architecturally-correct default but isn't the lever.
2. **Out-of-distribution test prompts** (Diag 2). The CLT was trained on long protocol documents; we test on short clinical sentences. This shifts the encoder's input distribution. Real but secondary — wouldn't cause arithmetic non-closure.
3. **Post-norm Jacobian formula is correct in form but extreme in magnitude** (Diag 3, 4). γ/scale ≈ 524 per layer for Gemma 3. The frozen-rms approximation drops a correction term that's negligible for normal RMSNorm but plausibly significant when γ is in the hundreds.
4. **Manual v_at_layer diverges from any natural chain-rule reference** (Diag 5). Whether the divergence is "by design" (skip-only) or "by bug" is no longer cheap to determine by inspection.

**Structural conclusion:** continuing to patch the manual chain rule is a
losing trade. Each Gemma-3 quirk (post-attn norm, post-mlp norm, GQA, GeGLU)
adds bespoke code that's individually plausible and collectively unverifiable.
The right move is to compute Jacobians via torch autograd against a frozen
forward pass — same math the paper specifies, far less code, architecture-agnostic.

See `docs/autograd_plan.md` for the rewrite plan.

---

## Open questions left to the rewrite

- Does the frozen-rms approximation itself break down on Gemma 3 with γ in the hundreds? Autograd doesn't fix this if we still freeze the rms denominator. We'll find out empirically once autograd lands.
- Is the CLT's L0 ≈ 91 too high for clean per-feature attribution even with correct math? Possible we'll need to retrain at L0 ~ 20–30. Budget for that contingency in the plan.
- The CLT was trained on full-protocol activations but tested on short-sentence activations. We should not assume good cross-distribution generalization. May need to re-evaluate which prompts are appropriate.

---

## Phase 0 — Pythia-70m baseline (2026-05-03)

Re-ran the existing manual-Jacobian `graphs/build.py` against Pythia-70m on a
fixed prompt before starting the autograd rewrite, to establish the regression
target Phase 1 must match within ±0.02.

| field | value |
|---|---|
| script | `scripts/phase0_baseline.py` |
| record | `docs/phase0_baseline.json` |
| model | EleutherAI/pythia-70m (6 layers, d_model=512, d_mlp=2048) |
| CLT | random init, n_features=128, seed=0 (decoders ~ N(0, 0.01)) |
| prompt | `"The capital of France is"` |
| target | `" Paris"` (vocab idx 7785) |
| device | mps |
| logit_value | 15.2718 |
| **completeness** | **0.9133** |
| nodes | 415 |
| edges | 49,791 |

Confirms CLAUDE.md's "0.91 with random CLT" is reproducible. Phase 1's
`graphs/build_autograd.py` must produce completeness in `[0.8933, 0.9333]`
on this exact configuration to count as a passing baseline.

---

## Phase 1 — Autograd build, Pythia-70m validation (2026-05-03)

`graphs/build_autograd.py` lands. ONE autograd backward pass against a
frozen-nonlinearity forward replaces the manual `_compute_readout_vector` +
`_compute_attention_propagated_v` + `_compute_corrected_logit_transfer` chain.
Hooks freeze attention pattern + every LN/RMSNorm scale; `hook_mlp_out` is
detached so the backward at `hook_resid_pre` only includes skip + attention
(matches the paper's linearised-MLP-via-CLT model). `grad_at_mlp_post` is
derived manually after the autograd pass since the mlp_out detach severs the
autograd path to mlp_post.

**Regression result** (`scripts/phase1_validate_autograd.py`,
`docs/phase1_autograd.json`):

| | manual baseline | autograd | Δ |
|---|---|---|---|
| completeness | 0.9133 | 0.8996 | −0.0137 |

Inside the ±0.02 acceptance window. Phase 1 PASS.

### Substantive finding — autograd reveals cross-position indirect paths

`scripts/diag_autograd_vs_manual.py` compares `v_at_layer[l]` (manual) to
`grads['resid_pre'][l]` (autograd) layer by layer:

| layer | ‖manual‖ | ‖autograd‖ | cos | rel L2 |
|---|---|---|---|---|
| 6 | 2.1917 | 2.1917 | **1.000000** | 0.000 |
| 5 | 2.2459 | 2.2464 | 0.995649 | 0.093 |
| 4 | 2.2675 | 2.2687 | 0.992319 | 0.124 |
| 3 | 2.3362 | 2.3651 | 0.950772 | 0.312 |
| 2 | 2.3943 | 2.4292 | 0.911903 | 0.417 |
| 1 | 2.4452 | 2.4751 | 0.871103 | 0.505 |
| 0 | 4.9215 | 5.3372 | **0.133158** | 1.267 |

The two agree exactly at the readout (layer L), then drift by exactly one
layer's worth per step — the fingerprint of a per-layer term that manual
omits and autograd captures.

**What manual misses.** `_compute_attention_propagated_v` propagates v as
`v_at_layer[l] = v_at_layer[l+1] + (J_attn[l] = a_tt[h] · W_V@W_O)^T @ v_at_layer[l+1]`.
The `a_tt[h] = pattern[target, target]` term only models the **self-attention**
path (target query reading from target key). But in the linearised replacement
model, perturbing `resid_pre[l, target]` also perturbs `v[l, target]`, which
perturbs `attn_out[l, q]` for every query q via `pattern[q, target]`. Those
perturbations at OTHER positions then feed back to `attn_out[l+k, target]`
through cross-position attention reads at later layers. This **cross-position
indirect** path is real in the linearised model and autograd traces it; manual
silently drops it.

**Why completeness still lands close (−0.014 vs the manual baseline):** the
total decomposition `feat + error + embed + attn = decomposable_logit` still
holds for both linearisations (it's just `v · r_L` at the readout). What
shifts is the per-component breakdown. With autograd:

| component | manual | autograd | Δ |
|---|---|---|---|
| feat_sum | −0.5498 | −0.4081 | +0.142 |
| error_sum | 6.2618 | 5.5790 | −0.683 |
| attn_sum | 5.3457 | 5.3867 | +0.041 |
| embed_sum | 0.0692 | 0.4030 | +0.334 |
| total | 11.1270 | 10.9606 | −0.166 |

The cross-position indirect contributions get redistributed across
embed/attn/feat. The 0.166 gap to `decomposable_logit = 12.18` widens slightly
because the redistribution pulls some weight onto features below the
`min_activation` threshold (currently 0).

**Implication.** The autograd version is the paper-faithful one (the paper
specifies "Jacobian with stop-gradients on all nonlinearities" = full chain
rule of the linearised model = what autograd computes). The manual code was a
simplification that happened to land at 0.91 because cross-position
contributions were absorbed into the error and attn nodes. We're not
regressing — we're correcting.

Open question for Phase 4: on MedGemma, where the manual code reached
completeness 4–8 (unphysical), the cross-position indirect path may be the
dominant inflation source. If so, autograd should land closer to ~1.0
naturally, without per-prompt tuning.
