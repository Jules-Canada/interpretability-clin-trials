# PREREG — EXP-XXX-<short-name>

**Committed:** <date>  **Commit hash at lock:** <hash>
**Spec snapshot:** `specs/vX.Y/` (immutable)
**Status:** LOCKED / DRAFT

An experiment without a committed PREREG is not part of the program. Do not soften
fields after seeing results — supersede with a new numbered experiment instead.

---

## 1. Hypothesis

State one falsifiable claim. Not "we will investigate X."

> H1:

## 2. Primary endpoint

Exactly one. Define the computation precisely enough that someone else could compute it
from the artefacts without asking you.

> Primary:

Secondary endpoints (exploratory, not for headline claims):

> S1:

## 3. Baseline comparator

**Required field.** Name the cheaper method that would make this result uninteresting if
it matched. State the threshold by which the mechanistic method must beat it.

> Baseline:
> Must exceed baseline by:

## 4. Sample size justification

How many items, and why is that enough to support a circuit-level claim? Reference the
expected effect size and the variance observed in any pilot.

>

## 5. Adjudication and blinding

Who labels features, are they blind to condition, how is inter-rater reliability
computed, what κ is acceptable.

>

## 6. Kill criteria

Prespecified conditions under which the hypothesis is abandoned rather than qualified.
Write these as you would want a reviewer to hold you to them.

> K1:
> K2:

## 7. Analysis plan

Including how confounds are handled and what will *not* be done (no post-hoc subgroup
selection, no threshold tuning after unblinding).

>

## 8. What a null looks like

Describe the negative result concretely. If you cannot describe it, the endpoint is not
well defined.

>

---

## Post-hoc log

Deviations from this pre-registration, dated, with reasons. Additions only — never edit
sections 1–8 after lock.

| Date | Deviation | Reason |
|---|---|---|
