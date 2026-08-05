# ADR-0003 — Specification-grounded interpretability as the program thesis

**Date:** 2026-08-03
**Status:** Accepted
**Supersedes:** the framing in ADR-0002 (not its technical staging)

## Context

ADR-0002 pivoted from from-scratch CLT training to circuit-tracer + Gemma Scope 2 PLTs,
and correctly identified that the finding, not the method, is the contribution. It did
not say what *kind* of finding, leaving three candidate behaviours ranked by
"noticeability x clinician edge."

That framing is interpretability-in-service-of-medicine. Assessed against the current
literature it is a crowded lane: multiple 2026 papers apply SAEs to medical models and
report medical features, and generic domain-finetuning model diffing using medical LLMs
as the case study is already published. A frontier interpretability audience treats
"medical model has medical features" as expected.

## Decision

Invert the framing. Medicine is adopted as the **testbed for interpretability's ground
truth problem**, not as the application domain.

The load-bearing observation: in almost every interpretability setting we know the
correct *answer* but not the correct *computation*. In clinical medicine the correct
computation is externally specified in advance — trial eligibility is a written
conjunction, guideline therapy is a published decision rule, and the legitimacy of using
a given patient attribute is adjudicated in the clinical literature. This permits
measurement of mechanistic correctness rather than behavioural correctness.

Three pillars follow: completeness, legitimacy, calibration. Pillars 1 and 2 are
recognised as one measurement in opposite directions (causal attribute-use, auditing
omission and commission respectively) and are executed jointly.

A second thread imports clinical-research epistemics — pre-registration, blinded
adjudication, sample-size justification, prospective validation, reporting checklist — and
is enforced structurally via a required PREREG template rather than left to intent.

## Consequences

- The scarce asset shifts from "71 contrastive prompts" to "adjudicated spec items with
  required-condition and attribute-policy annotations." The prompts are the seed; the
  adjudication is the moat.
- Clinical adjudication time, not GPU time, becomes the critical path.
- Baselines (probes, logprob, prompt-debiasing) become first-class and must run in
  parallel with mechanistic work, not after it.
- A well-powered pre-registered null becomes a publishable outcome rather than a failure.
- Generic MedGemma-vs-Gemma diffing is demoted to non-goal except where scoped to a pillar.
- Repo reorganises around claims rather than method: doctrine / spec asset / execution.

## Rejected alternatives

- **Lead with a single clinical failure-mode case study.** Higher noticeability per unit
  effort, but produces an anecdote rather than an instrument, and does not transfer to
  other models or other labs.
- **Ship the transcoder as the contribution (MedScope play).** Real artefact value, but
  competes directly with labs on their strength and does not use clinical judgement.
