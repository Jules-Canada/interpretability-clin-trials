# Round 1 Summary — Eligibility Circuits in Gemma-3-4B-IT & MedGemma-4B-IT

*Two-day arc, 2026-06-17 → 2026-06-19. Active path: circuit-tracer (nnsight) +
Gemma Scope 2 per-layer transcoders. See CLAUDE.md Phase 2 for context.*

## TL;DR

- Built and validated a new attribution-graph pipeline (circuit-tracer + Gemma Scope 2
  transcoders) — the first completeness-passing path after the from-scratch-CLT era.
- Found a clinically-consequential failure: both a **general** model (Gemma-3-4B-IT) and a
  **medical** model (MedGemma-4B-IT) **over-exclude eligible patients** on knowledge-dependent
  trial-eligibility criteria.
- The failure is **robust**: same behavioral errors, same within-model circuit signature, and it
  **survives an ~85% reorganization** of the circuit between the two models. Medical fine-tuning
  rebuilt the machinery but did not fix the bug.

## Method / setup

- **Models:** `google/gemma-3-4b-it` (general control) and `google/medgemma-4b-it` (medical),
  both Gemma-3-4B architecture.
- **Transcoders:** `mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine`
  (16k width, low-L0, affine skip). Same set used on both models → features directly comparable.
- **Prompts:** 5 matched contrastive eligibility pairs (`prompts/eligibility.py`), a
  "decision-primitive ladder": numeric (age) → ordinal (ECOG) → lexical (histology) →
  world-knowledge (drug class: pemetrexed/pembrolizumab; staging: III/IV). Each pair flips one
  deciding token across the eligibility threshold.
- **Target:** single-target attribution on the predicted Yes/No token (both captured via top-K
  logits). True contrastive `logit(Yes)−logit(No)` deferred to a later round.
- **Tooling (all committed):** `setup_pod_circuit_tracer.sh`, `sweep_eligibility.py` (behavioral
  gate), `run_graphs_ct.py` (graph batch), `compare_graphs.py` (pairwise feature overlap),
  `analyze_graphs.py` (per-graph structural summary), `stage0_tokenizer_check.py`.

## Stage 0 (hosted Neuronpedia) — a discarded lead

- Confirmed Neuronpedia *does* run on-demand Gemma-3-4B-IT graphs (public docs say otherwise).
- Hosted UI limits forced raw-cloze prompts (no chat template, 64-token cap, mangled special
  tokens).
- Raw-cloze sweep suggested the model **ignores stated age bounds** (effective band ~[16,97]).
- **This was a format artifact** — overturned in Stage 1 under proper chat formatting. Caught
  before it became a claim. Lesson: always reproduce hosted/raw-cloze leads under the real IT
  chat template.

## Behavioral findings (forward-pass gate, both models)

- Under the proper chat template the model reads eligibility well: age sweep 17/18, bound-variation
  4/4 — it *does* use the stated criteria.
- **Over-exclusion on knowledge-dependent EXCLUSION criteria** (the real finding): the *eligible*
  arms wrongly answer No —
  - `pemetrexed` → No (chemotherapy, not a checkpoint inhibitor → should be eligible)
  - `stage III` → No (not metastatic → should be eligible)
  while the true-exclusion arms (`pembrolizumab`, `stage IV`) correctly → No.
- The model says "ineligible" whenever a prior drug / cancer stage is mentioned in an exclusion
  context, without applying drug-class / staging knowledge to *clear* the patient.
- **Identical on MedGemma** — same two errors. Medical fine-tuning did not fix it.
- Minor: a single off-by-one age glitch in each model (Gemma misses 75; MedGemma fixes 75 but
  misses 20). Both 17/18. A curiosity, not the headline.

## Circuit findings (attribution graphs)

- **Pipeline validated:** completeness 0.80–0.82 (Gemma), 0.84–0.85 (MedGemma) — all above the
  0.5 gate. The off-distribution gap from using generic transcoders on the medical model is
  **negligible** (MedGemma completeness ≥ Gemma) — the transcoders transfer cleanly.
- **Over-exclusion is visible in the circuit.** The deciding token moves the circuit far less for
  knowledge pairs than for controls:

  | pair | type | Gemma jaccard | MedGemma jaccard |
  |---|---|---|---|
  | age | control | 0.71 | 0.73 |
  | ecog | control | 0.67 | 0.64 |
  | histology | control | 0.67 | 0.64 |
  | priortx | knowledge | **0.87** | **0.78** |
  | stage | knowledge | **0.91** | **0.81** |

  Controls (model discriminates, answer flips) → low overlap. Knowledge pairs (model fails to
  discriminate, answer stays No) → high overlap. Same fingerprint in both models.
- **"Right answer, wrong reason":** even the *correct* knowledge No's (pembrolizumab, stage IV)
  come through the same generic "exclude scaffold" that mis-fires on the eligible arms — so the
  correct answers aren't evidence of real reasoning either.
- **Shared shallow scaffold dominates** every graph: a few recurring early/mid features carry
  most influence; prompt-specific clinical content contributes little (decision is ~48–70%
  early-layer). Consistent with a pattern-driven, not concept-driven, decision.
- **Cross-model: same failure, different circuit.** Gemma↔MedGemma overlap on the same prompt is
  only ~0.13–0.17 jaccard — medical fine-tuning **reorganized ~85% of the circuit** (MedGemma
  leans on more late-layer features, L27/L25/L29/L30, vs Gemma's L16-dominated scaffold) yet
  preserved the failure.
- **Nuance:** MedGemma's knowledge-pair overlap (0.78–0.81) is lower than Gemma's (0.87–0.91) →
  the drug/stage token perturbs MedGemma's circuit slightly *more*. The medical model "notices"
  the distinction a bit more — just not enough to flip the answer.

## Caveats (do not overclaim)

- **Single-target, not contrastive yet.** True `logit(Yes)−logit(No)` deferred.
- **Same-answer confound.** Controls flip the answer (which itself forces circuit divergence), so
  part of their lower overlap is the answer change, not pure discrimination. Needs a same-answer
  minimal-pair baseline.
- **"Knowledge gap" vs "exclusion-phrasing → No bias" not yet separated** — the 3 clean pairs are
  inclusion criteria, the 2 failing ones exclusion. One cheap behavioral probe resolves this.
- **No feature labels yet** (clerp empty). Circuit claims are structural (overlap, layer, influence),
  not yet semantic. Feature labeling is the next mechanistic step.
- **Local `elig_age_pos.json` (Gemma) truncated to 0B** by an accidental shell paste; regenerable,
  derived results retained.

## Artifacts

- Gemma graphs: `frontend/graph_data/elig_*.json` (9 intact + 1 truncated)
- MedGemma graphs: `frontend/graph_data/medgemma/elig_*.json` (10)
- Sweeps: `data/eligibility_sweep_gemma-3-4b-it.json`, `data/eligibility_sweep_medgemma-4b-it.json`
- All scripts in `scripts/`; findings in CLAUDE.md (Current Status / Findings) + memory.

## Next steps (analysis is GPU-free; new graphs need a pod)

1. **Kill the confound** — same-answer minimal-pair baseline; separate knowledge-gap from
   exclusion-phrasing bias (probe e.g. `Exclusion: prior chemotherapy. Patient: treatment-naive.`
   → should be Yes).
2. **Feature labeling** — name the shared exclude-scaffold, the candidate Yes-direction feature
   (Gemma L14 f7595238), and MedGemma's late-layer features. Turns structural claims into
   mechanistic ones.
3. **Contrastive readout** — re-run the knowledge pairs with `logit(Yes)−logit(No)` now that the
   transcoders are sparse enough.
4. **Write up** — the cross-model-robust over-exclusion is the headline result.
