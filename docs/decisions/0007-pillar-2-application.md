# ADR-0007 — Pillar 2 becomes Application; Legitimacy is withdrawn

**Date:** 2026-08-07
**Status:** Proposed
**Supersedes if accepted:** ADR-0003's pillar 2 (Legitimacy). Pillars 1 and 3 unchanged.

## Context

ADR-0003 named three pillars: completeness, legitimacy, calibration. Pillar 2 asked whether
a demographic attribute moves the answer without moving the intermediate — whether it
*routes around* the clinical reasoning rather than passing through it.

That formulation was loose in four ways, and they compound.

**It lacked the ground truth the programme runs on.** Pillar 1 scores against the ECOG
table: published, unambiguous, older than the model. Pillar 2 scores against an "attribute
policy" that mostly does not exist and is contested where it does — whether age may inform
oncology trial eligibility is argued, not settled. The thesis is that we can score
computation against a spec that existed before the model. Pillar 2 was where that premise
was weakest. The one canonically settled case, the race coefficient formerly in eGFR, is
not trial eligibility at all.

**Its signal was a discordance with no measured null.** The pillar fires when the
eligibility answer moves while the grade does not. Those are two independent prompts, and
some fraction disagree with no manipulation whatever. That baseline had never been measured,
so any result would have been a numerator without a denominator.

**It was downstream of pillar 1, not parallel to it.** ADR-0003 has pillars 1 and 2 sharing
stimuli and instruments, "run together". But if the model does not compute the intermediate,
then "moves the answer without moving the intermediate" is trivially true of everything.
Pillar 2 is interpretable only conditional on pillar 1 succeeding.

**It had no stimuli, and the design rule blocked them.** Zero age or gender markers across
all 72 vignettes in `specs/stimuli/`. Vignettes vary only the clinical description, so
demographic variants are a second axis that was never built.

A positive result was also multiply interpretable: bias, a legitimate clinical proxy,
readout contamination from register shift, or turn-to-turn noise.

## Decision

Pillar 2 is **Application**: *is the inclusion decision guided by the guideline score?*

The measured quantity is the **coupling between the two turns**, not the accuracy of either.
The eligibility turn and the grading turn are already independent prompts; the pillar asks
whether the verdict follows from the intermediate the protocol is written in. The
informative cell is `right_answer_wrong_grade` — the model reaching the correct verdict
while holding the wrong grade, which is "right answer, wrong reasons" measured directly.

This fixes the ground-truth problem. The threshold is in the protocol, unambiguous, and
predates the model — the same quality of spec pillar 1 enjoys. No attribute policy is
needed.

It is already instrumented. `score_rows` in `scripts/run_ecog_stimuli.py` computes
`self_consistent` and buckets every row into `both_right`, `right_answer_wrong_grade`,
`wrong_answer_right_grade`, `both_wrong`. No new stimuli are required: the existing ECOG
set exercises it, and E026 (`reversed_threshold`) is a pillar-2 item under this definition —
the criterion direction is flipped, so a model anchoring on a memorised "ECOG ≤ 1 is
standard" prior fails it while grading correctly. `mrs_v0.csv` carries an equivalent row.

**Legitimacy leaves the programme.** This is a deliberate narrowing, not an oversight.
Behavioural bias testing is the crowded lane ADR-0003 warns against, and the version worth
returning to is mechanistic rather than behavioural: *is there a causal path from the
demographic token to the eligibility logit that does not pass through the intermediate's
representation?* That asks only whether an attribute bypasses, never whether it is
permitted, so it needs no contested policy. It is also downstream of pillar 1 locating the
intermediate as a representation. Not scheduled.

**Pillar 3 (Calibration) is retained as Later, not dropped.** It is currently absent from
the `CLAUDE.md` table while live in ADR-0003 and `docs/program/thesis.md` (:47, :53, :73),
and `recist_v0.csv` has an `indeterminate` row that serves no other pillar. Retaining it
resolves that inconsistency; dropping it would need its own argument, which nobody has made.

## Consequences

- `CLAUDE.md` already reflects the Application framing, so it has been running ahead of the
  ADR trail. Accepting this closes that gap. The pillar-3 row needs restoring to the table.
- **The first measurement of the new pillar 2 says its floor is high.** On the 2026-08-06
  ECOG pilot, self-consistency was 24/39 (gemma-3-4b-it) and 29/39 (medgemma-4b-it) — the
  two turns disagree between a quarter and two fifths of the time with no manipulation at
  all. `right_answer_wrong_grade` was 5 and 10. The interesting cell is not yet
  distinguishable from that noise. Pillar 2 is now measurable; it is not yet measured.
- **The designed item, however, fired cleanly.** E026 flips the criterion direction while
  describing a grade-1 patient. Both models graded it correctly — `pred_grade` 1, matching
  the annotation — and both answered eligibility wrongly, giving
  `wrong_answer_right_grade` on both. That is the intermediate computed correctly and then
  not applied: a pillar-2 failure in exactly the form the row was written to elicit, and
  identical across two models, which is not what turn noise looks like. The aggregate cell
  is noise-limited; a purpose-built item is not.
- Reducing the floor is the pillar's first task. Whether the residual disagreement is real
  dissociation or prompt-format instability is unknown and is answerable offline from the
  existing results. The immediate implication for stimulus writing is that
  `reversed_threshold` rows are worth more per row than general vignettes for this pillar,
  and ECOG has one of 39.
- The name "Legitimacy" is retired. Nothing in `specs/schema/spec_item.json` depends on it;
  the `attribute policy` field remains for spec items but no pillar currently consumes it.
- ADR-0003's pillars 1 and 3, its method, and the PREREG discipline are unaffected.
