# Contrastive + Classic CLT Tracing on MedGemma — Workplan

Status: active (created 2026-05-16)

## Motivation

The `Eligible?\nAnswer:` scaffold produces a flat ~0.3 p_agg on `medgemma-4b-pt`
with no dynamic range between trivial and hard prompts (confirmed float32, H100,
quant_tie_count: 0). Scaffold engineering cannot fix a pretrained base model's
output distribution. The Anthropic Circuit Tracing paper never forces a target
the model doesn't commit to — it decomposes the model's *own* behavior.

Two complementary approaches recover traceable signal without scaffold hacks:

1. **Contrastive readout** (Track A): trace `logit(Yes) − logit(No)` instead of
   `logit(Yes)` alone. Features that push equally toward both answers cancel;
   only features that differentially drive the eligibility *decision* appear.
   Decontaminates scaffold/format artifacts by construction. Requires only that
   matched pos/neg pairs *separate* on the logit difference — does not require
   a peaked softmax.

2. **Classic medical-factual** (Track B): find prompts where the base model
   *does* commit confidently to a single medical token (the medical equivalent
   of "The capital of France is" → " Paris"). These work with the standard
   single-token readout because the model genuinely predicts them.

## Connection to the CLT methodology

The attribution graph readout vector `v` is already arbitrary in `graphs/build.py`.
The frozen-autograd backward pass, completeness check, and feature-contribution
decomposition are invariant to the choice of `v`. Three valid choices:

| v | Traces |
|---|---|
| `W_U[:,tok] − W_U.mean(dim=1)` | Paper standard: why tok vs vocabulary mean |
| `mean(W_U[:,Yes_ids]) − mean(W_U[:,No_ids])` | Contrastive: why Yes over No |
| `W_U[:,tok]` (no centering) | Raw logit (not used, included for completeness) |

Contrastive is the standard in binary-decision circuit analysis (IOI, sparse
feature circuits). For the CLT specifically: cross-layer decoder paths
(`W_dec[l'→l]`) that carry category-recognition features to the decision
direction are exactly the "reasoning circuits" the project aims to find.

## Tracks

### Track B — Classic medical-factual (do first)

**Purpose:** validate the MedGemma CLT pipeline end-to-end; prove interpretable
medical features exist. Zero methodology risk — this is what the paper was
designed for.

**Prompts:** 10–15 factual cloze completions where `medgemma-4b-pt` predicts a
single medical token confidently. Examples:

- "The standard first-line treatment for CML is" → " imatinib"
- "HER2 amplification is targeted by the monoclonal antibody" → " trastuzumab"
- "The antidote for heparin overdose is" → " protamine"
- "Warfarin acts by inhibiting vitamin" → " K"

**Screen:** absolute p(target) via existing `screen_prompts.py`. Expect p > 0.3
easily. Drop/rephrase any < 0.2.

**Graphs:** standard single-token readout, completeness ≥ 0.5 gating.

**Output:** labeled medical-knowledge features, layer distribution, activation
examples. Table for the paper: "MedGemma CLT features."

### Track A — Contrastive eligibility (the novel contribution)

**Purpose:** trace the categorical-reasoning circuits that distinguish eligible
from ineligible. The publishable finding.

**Screening metric:** for each pos/neg pair, compute:
```
logit_diff(prompt) = mean_logit(Yes_variants) − mean_logit(No_variants)
gap = logit_diff(pos_prompt) − logit_diff(neg_prompt)
```

Success: `gap > 0` with margin. Easy controls should show larger gaps than hard
categorical (dynamic range on the difference).

**Readout vector:**
```python
v = mean(W_U[:, Yes_variant_ids]) − mean(W_U[:, No_variant_ids])
```
scaled by `ln_final_w / hook_scale` as usual.

**Graphs:** for each separating pair, two graphs (one per prompt) with shared
contrastive `v`. Completeness on the logit *difference*.

**Analysis:**
- Features in pos-graph but not neg-graph (or sign-flipped) = the categorical
  reasoning features.
- Within each of the 6 axes (drug-class, histology, site, surgical, allergy,
  receptor): do the same features fire across the 3 pairs?
- Across axes: are there generic "eligibility-reasoning" features vs
  category-specific ones?

## Deliverables (code)

| File | Purpose | Track |
|---|---|---|
| `prompts/medical_knowledge.py` | Factual cloze prompts | B |
| `scripts/screen_contrastive.py` | Logit-diff pair separation screen | A |
| `graphs/build.py` (small edit) | `contrastive_readout()` helper | A |
| `scripts/run_graphs_batch.py` (flag) | `--contrastive` mode | A |

## Sequencing

All local code/prompts built first (this session). One pod trip screens + graphs
everything.

```
Local (this session, ~3 hr):
  1. prompts/medical_knowledge.py          — design + write
  2. scripts/screen_contrastive.py         — contrastive pair-separation screen
  3. graphs/build.py contrastive_readout() — helper function
  4. scripts/run_graphs_batch.py --contrastive — wire up

Pod (one session, A100 24GB sufficient, float32 screen / bf16 graphs):
  5. Screen Track B (absolute p, ~2 min)
  6. Screen Track A (contrastive gap, ~2 min)
  7. Graphs Track B (standard readout, ~20 min)
  8. Graphs Track A (contrastive readout, ~30 min)
  9. find_top_activations on new graph features
  10. scp artifacts back

Local (next session):
  11. label_features.py on Track B graphs
  12. Differential feature analysis on Track A pairs
  13. Cross-pair generalization (6 axes × 3 pairs)
```

## Decision points (data-driven, not pre-committed)

After step 6:
- Track A pairs separate → proceed to contrastive graphs
- Track A pairs don't separate → report as finding; focus paper on Track B

After step 7:
- Track B completeness ≥ 0.5 → pipeline validated on MedGemma
- Track B completeness < 0.5 → CLT quality issue; consider retrain at L0~20–30

After step 12:
- Labeled features are interpretable → proceed to cross-pair analysis
- Features are noise → retrain CLT (L0~91 is too dense)

## Compute estimate

- Screening (steps 5–6): model forward only, no CLT. ~5 min on any 16GB+ GPU.
- Graphs (steps 7–8): model + CLT in memory. A100 24GB sufficient for 4B float16
  + CLT. ~50 min total for ~25 graphs.
- find_top_activations (step 9): HDF5 scan, ~1 hr if HDF5 on pod.
- Total pod time: ~2 hr. Cost: ~$3–5 on A100.

## What this produces for the paper

1. **Track B result:** "MedGemma's CLT contains interpretable medical-knowledge
   features" — table of features with activation examples, layer distribution.
   Validates that the CLT methodology transfers to a medical domain model.

2. **Track A result:** "Contrastive attribution reveals category-specific
   eligibility features" — which features fire when the model distinguishes a
   corticosteroid from an NSAID, HCC from cholangiocarcinoma, etc. Cross-layer
   paths from early recognition to late decision.

3. **Dynamic-range reframe:** easy-control contrastive gaps > hard-categorical
   gaps confirms the model *does* represent difficulty internally — it just
   didn't show in absolute p. Turns the "diffuseness finding" into a positive
   methodological result.
