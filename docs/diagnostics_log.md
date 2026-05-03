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
