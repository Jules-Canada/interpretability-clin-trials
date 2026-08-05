# Strata — Program Thesis

**Specification-grounded interpretability: measuring whether a clinical model is right
for the right reasons.**

---

## The claim

Mechanistic interpretability lacks ground truth about what the *correct* computation is.
We can check whether a model got Paris right; we cannot check whether it got there the
right way, because no one wrote down the right way.

Clinical medicine is the exception. The correct computation is externally specified, in
writing, by humans, before the model exists:

- Trial eligibility is an explicit conjunction of criteria in a protocol document.
- Guideline-directed therapy is a published decision rule.
- Which patient attributes may legitimately enter a decision is adjudicated in the
  clinical literature — including the cases where demographics *are* legitimately
  diagnostic.

This makes it possible to measure **mechanistic correctness**, not merely behavioural
correctness. That is a new evaluation axis for interpretability, and medicine is the
domain that supplies it.

The contribution is therefore *not* "interpretability applied to medicine." It is
"medicine as the testbed that lets interpretability be evaluated at all."

---

## Falsification

The thesis fails if any of the following hold:

1. Clinical intermediates are not decodable from the model at rates above chance — the
   specification does not bind to internals at all.
2. Intermediates are decodable but no intervention on them changes the answer, meaning the
   representation is present and unused.
3. Nothing in the mechanistic layer beats behavioural baselines — asking the model for the
   intermediate directly fixes the errors, and the honest headline is a null.

Each is a prespecified kill criterion, not a limitation to be buried in discussion.

---

## The three pillars

| Pillar | Question | Spec supplies |
|---|---|---|
| **1. Completeness** | Does the model compute the clinical intermediates the spec is written in? | The named intermediate and its threshold |
| **2. Legitimacy** | Does a demographic attribute move the answer without moving the intermediate? | The permitted/forbidden partition |
| **3. Calibration** | Does the circuit know when the spec does not determine the answer? | The evidence-strength boundary |

**Pillars 1 and 2 share one object: the clinical intermediate.** Specifications are not
written in raw patient facts but in abstractions — ECOG grade, creatinine clearance, NYHA
class, RECIST measurability. Each names something the model must compute before it can
compare anything to a threshold.

Pillar 1 asks whether the model represents those intermediates or only the vocabulary they
are usually described in. The decisive test is paraphrase generalisation: clinical notes
never use protocol phrasing, so a model holding the ECOG table but not the concept fails
silently on the population it would be deployed against.

Pillar 2 asks whether a demographic attribute moves the answer while leaving the
intermediate unchanged. That is bypass — the clinical computation being routed around
rather than informed. Same instruments, same stimuli, run together.

Method is probing and activation patching first, with transcoders used as a dictionary
(does a pre-existing feature track the intermediate?) before they are used to build graphs.
See ADR-0004.

Pillar 3 is sequenced after, and reuses the same spec items with the answer withheld.

---

## Second thread — trial-grade epistemics

Interpretability has an acknowledged over-claiming problem. Attribution graphs are
hypotheses over an imperfect replacement model; feature labels are LLM-generated and
largely unvalidated. This project imports the machinery clinical research already has:

- Pre-registration of hypotheses and primary endpoints before execution
- Blinded adjudication of feature labels with inter-rater reliability reported
- Sample-size justification for circuit-level claims
- Prospective validation on a locked held-out set
- A STARD-style reporting checklist for mechanistic claims

This thread costs almost no compute and generalises far beyond medicine. It is enforced
structurally: see `experiments/TEMPLATE-PREREG.md`. An experiment without a committed
pre-registration is not part of the program.

---

## What ships

1. **The benchmark** — spec items runnable against any model, not just ours.
2. **The intermediate-recovery harness** — stimulus sets, probes, feature search — plus
   hosted attribution graphs for the cases that warrant them. (Domain-adapted transcoders
   were struck by ADR-0004: Stage 2 is killed.)
3. **The clinical trial protocol corpus.**
4. **A regulatory annex draft** — what mechanistic evidence belongs in a PCCP or 510(k),
   given the January 2026 FDA CDS guidance's emphasis on transparency about inputs and
   logic.

---

## Non-goals

- Training a better transcoder. Method is instrument, not contribution.
- Demonstrating that a medical model has medical features. Expected, and already crowded.
- Generic model diffing of MedGemma vs Gemma 3. Partly claimed elsewhere; only pursued
  where scoped to a pillar.
