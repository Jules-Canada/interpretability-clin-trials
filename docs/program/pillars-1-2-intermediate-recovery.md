# Pillars 1 & 2 — Intermediate Recovery and Bypass

**Status:** active, run jointly.
**Supersedes** the attribute-use framing. See ADR-0004.

---

## The question

Protocols are not written in raw patient facts. They are written in clinical abstractions:
ECOG ≤1, creatinine clearance ≥60, NYHA ≤II, measurable disease by RECIST. Each names an
intermediate the model must compute before it can compare anything to a threshold.

So the pillar-1 question is not "did the circuit check criterion 3." It is:

> Does the model represent the clinical intermediates its specifications are written in, or
> only the vocabulary those intermediates are usually described in?

Worked case. Protocol requires ECOG ≤1. Patient walks unassisted but cannot work. That is
ECOG 2 — the 1/2 boundary is exactly *able to do light or sedentary work* versus *unable to
carry out any work activities* — so the answer is ineligible. Two steps: map narrative to
grade, compare grade to threshold.

Pillar 2 rides on the same object. If a demographic attribute moves the eligibility answer
while leaving the intermediate representation unchanged, it is bypassing the clinical
computation. That is a sharper illegitimacy claim than "direct path to the answer logit" and
it needs no new instrument.

---

## Why paraphrase generalisation is the load-bearing test

Trial screening runs on clinical notes. Notes do not use protocol phrasing. A model that
holds the ECOG table but not the concept fails silently on exactly the population it would
be deployed against, and passes any evaluation written in protocol language.

Every other step here is setup for this one.

---

## Starting intermediates

Three for the first result. More is not better; the cost is in vignette writing, not compute.

| Intermediate | Type | Why |
|---|---|---|
| ECOG | Judged | Ordinal, narrative-describable, near-universal, decisive 1/2 boundary |
| NYHA class | Judged | Second judged case, different disease area, tests generality |
| Creatinine clearance | Computed | Tests arithmetic vs recall, and drags eGFR/Cockcroft-Gault and the race coefficient into scope for pillar 2 |

Judged intermediates test whether the model has the abstraction. Computed ones test whether
it calculates or pattern-matches to a remembered result. Both are needed.

Later candidates: RECIST measurability, Child-Pugh, CTCAE residual-toxicity grade, EDSS,
GOLD stage.

---

## Protocol

**Step 0 — Stimulus set.** Clinician time, no GPU. The bottleneck.

Per intermediate: ~25 vignettes per grade or band, describing the clinical state in narrative
that never uses the defining vocabulary. Then a **held-out set written to be lexically
distant** — different register, patient voice, note fragments. That held-out set is the
experiment.

Each vignette pairs with an eligibility prompt against a protocol stating the threshold.
Hold everything else fixed; vary only the clinical description. Otherwise the probe learns
sick-versus-well, which the model certainly has and which is not ECOG.

**Step 1 — Behaviour.** Eligibility accuracy, broken out by grade. Expect ceiling at the
extremes and failure at the decisive boundary. If accuracy is at ceiling everywhere, the
vignettes are too easy and nothing downstream will be informative.

**Step 2 — Probe and feature search, in parallel.**

*Probe:* linear, ordinal, residual stream, every layer, fixed token position at the end of
the narrative. Read where the intermediate becomes decodable and how sharply. Train on
non-adjacent grades and test on the held-out grade as one condition — interpolating an
unseen grade is a stronger claim than in-distribution accuracy.

*Feature search:* same stimuli through MedGemma with Gemma Scope 2 transcoders attached.
Does any existing feature track the intermediate? Forward pass and correlation only — no
graph construction, no pruning, no export.

These cross-validate. Probe succeeds and feature search fails: the variable exists but is not
in the dictionary at this width. Both succeed and agree: strong. Feature search succeeds:
you have a named, pre-existing, ablatable object rather than a direction fitted to your own
labels — which is the difference between this and a probing paper.

**Step 3 — Paraphrase generalisation.** Probe and feature tested on the lexically distant
held-out set. Collapse means vocabulary, not concept. This is the headline either way.

**Step 4 — Causal.** Patch activations from a below-threshold run into an above-threshold run
at the peak layer, narrative token positions. Does eligibility flip? Then the sharper
version: rank-1 edit along the probe direction, and ablation of the transcoder feature. If
either flips the answer, the variable the probe found is the one the answer computation
reads.

**Step 5 — Bypass (pillar 2).** Vary a demographic attribute, hold the clinical narrative
fixed. Measure the shift in the answer and the shift in the intermediate representation.
Answer moves, intermediate does not → bypass. This is where the legitimacy taxonomy is
needed, and only here.

**Step 6 — Graphs, contingent.** On cases where probe and patching disagree, or where the
model answers correctly with no decodable intermediate. Genuine puzzles, worth the export
size and the reading time. Ten prompts, not seventy.

---

## Early graph batch

Four to six graphs on boundary pairs, run immediately after Step 0. Not for exhaustive
reading. Two purposes: confirm the pipeline runs on the new prompt format while operational
knowledge is current, and check whether narrative tokens reach the answer through an
intermediate node or bypass it entirely. A bypass finding here reframes Steps 2–4 before
they are run.

---

## Baselines — same week, not after

| Claim | Comparator that must lose |
|---|---|
| Model represents the intermediate | Ask it directly for the grade. If prompting for the intermediate first fixes eligibility errors, the representation was present and the failure is downstream — different story, worth knowing |
| Probe finds a used variable | Probe on the answer logit directly. If equal, the intermediate layer adds nothing |
| Feature is the mechanism | Random feature ablation, matched activation magnitude |
| Bypass is real | Behavioural counterfactual alone |

---

## Kill criteria

- Intermediate not linearly decodable above chance at any layer → pillar 1 dies for that
  intermediate; report and move to the next.
- Decodable but paraphrase accuracy collapses → that *is* the result. Do not soften it into
  "partially generalises."
- Patching and feature ablation both fail to flip the answer → the representation is present
  but unused; the eligibility computation runs on something else, and that becomes the
  question.
- Probe matches transcoder feature everywhere → the dictionary adds nothing here; say so.

---

## Dependencies

- Stimulus sets per intermediate. **Critical path.**
- Legitimacy taxonomy, scoped to Step 5 only. Start with the four formula-embedded cases:
  eGFR race coefficient, Cockcroft-Gault sex term, FEV1 reference equations, MELD 3.0.
- `specs/schema/spec_item.json` extended with an `intermediate` block.
- Locked PREREG before Step 1. Steps 0 and the early graph batch are pilot work and do not
  need one, but their numbers feed the sample-size field.
