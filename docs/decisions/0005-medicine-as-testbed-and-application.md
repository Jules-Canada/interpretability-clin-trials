# ADR-0005 — Medicine as both testbed and application domain

**Date:** 2026-08-05
**Status:** Proposed
**Supersedes if accepted:** the framing in ADR-0003 (not its method, pillars, or PREREG discipline)

## Context

> **DRAFT — this section is the one that matters and is not yet written.**
>
> ADR-0003 chose testbed-only framing for a stated empirical reason (0003, lines 14–18):
> interpretability-in-service-of-medicine was assessed as a crowded lane — multiple 2026
> papers apply SAEs to medical models and report medical features, generic
> domain-finetuning model diffing using medical LLMs as the case study is already
> published, and a frontier interpretability audience treats "medical model has medical
> features" as expected.
>
> That assessment does not dissolve on its own. For this ADR to supersede 0003, this
> section has to say what changed. Candidate answers, none yet chosen:
>
> - **Audience/venue.** "Crowded lane" was scoped to a frontier interpretability
>   audience. Against a clinical or ML-for-health venue the objection largely does not
>   apply, and the application side is the stronger position.
> - **Cost of the constraint.** Testbed-only ruled out work worth doing (clinical failure
>   cases, MedGemma diffing, deployment-relevant claims) and was costing more than it
>   bought.
> - **A differentiator.** Something specific about this programme answers the crowding
>   directly — most plausibly that spec-grounded mechanistic correctness is not what the
>   crowded work measures.
>
> Until this is filled in, the ADR stays Proposed and ADR-0003 governs.

## Decision

*(Proposed, not in force.)*

Medicine is adopted as **both** the testbed for interpretability's ground-truth problem
and an application domain in its own right, superseding ADR-0003's "not as the application
domain."

Explicitly unchanged by this ADR:

- The load-bearing observation that clinical computation is externally specified in
  advance, which is what permits measuring mechanistic rather than behavioural
  correctness. This holds on either side of the divide and is the reason spec items work
  as ground truth at all.
- The three pillars — completeness, legitimacy, calibration — and the joint execution of
  1 and 2.
- The clinical-research epistemics thread: pre-registration, blinded adjudication,
  sample-size justification, prospective validation, reporting checklist, enforced via the
  PREREG template.
- Everything in ADR-0004.

## Consequences

**Open — these need answers before this ADR can be accepted.** ADR-0003's Consequences
section derives six commitments from testbed-only framing. Co-equal framing does not
obviously preserve them:

- *"Clinical adjudication time, not GPU time, becomes the critical path."* If application
  work is co-equal, what is dropped when application work and adjudication compete for the
  same clinician hours? The current answer in CLAUDE.md is "writing wins." Does it still?
- *"Generic MedGemma-vs-Gemma diffing is demoted to non-goal except where scoped to a
  pillar."* Application framing weakens the grounds for this exclusion. Keep, drop, or
  re-scope?
- *"A well-powered pre-registered null becomes a publishable outcome rather than a
  failure."* This is strongly audience-dependent and should be re-checked against whatever
  venue the Context section names.
- The remaining "Not doing" entries in CLAUDE.md were derived from testbed framing. They
  need re-deriving rather than silent retention.

The general risk to answer somewhere in this section: testbed-only framing was doing
defensive work — it supplied the grounds for declining crowded-lane work. "Both" permits
everything, so it cannot by itself say what to decline. If it is adopted, something else
has to carry that load.

## Downstream if accepted

- `CLAUDE.md` — §What we're doing already reflects the "both" direction; §History still
  describes 0003 as "Holds" and would need updating.
- `docs/program/thesis.md:28` — quotes the testbed framing directly.
- `docs/program/thesis.md:111` — lists "Demonstrating that a medical model has medical
  features. Expected, and already crowded" as out of scope.
- `docs/decisions/0004-intermediate-recovery.md:5` — records "ADR-0003 (thesis holds,
  pillar 1 method changes)"; the amendment chain would need a note.
