# ADR-0006 — Pre-registration applies to formal runs, not to every run

**Date:** 2026-08-05
**Status:** Accepted
**Amends:** rule 1 in `CLAUDE.md` (not the PREREG template, which is unchanged)

## Context

Rule 1 as written — "no experiment without a locked `PREREG.md`… a directory without one
isn't part of the program" — applies late-phase clinical trial discipline to every run.

That discipline answers conditions this program does not have. Pre-registration, locked
endpoints and prespecified stopping rules exist because a Phase III trial is one expensive
irreversible shot, with patient harm on the line and a sponsor incentivised to spin the
result. Here a run is a few thousand forward passes on a 4B model, repeatable the same
afternoon for negligible cost. The premise pre-registration is built on — *you only get one
look* — is false.

Applied literally the rule has two failure modes, and both are worse than a narrower rule.
Either exploratory work stops, or the rule is ignored wholesale and the discipline is lost
where it matters.

Two pieces of evidence from this repo:

**The failures pre-registration would have caught are not the failures we had.**
`data/feature_labels.jsonl` was populated from an empty-context bug and the labeller
produced confident nonsense (ADR-0001). Two Round-1 claims failed re-verification: the
early/late layer weighting was backwards, and the cross-model jaccard was size-confounded.
All three were code and analysis errors. All three were caught by re-running and checking —
which is why `scripts/reproduce_round1.py` exists and takes three minutes. A locked PREREG
would have caught none of them.

**Locking early can freeze a broken instrument.** On 2026-08-05 a 20-vignette "held-out"
set was written, then measured and found not to be lexically distant: paired rows shared 4x
more vocabulary than unrelated rows, and the discriminating stems survived the rewrite. What
had changed was register, not lexis. That was found by poking at it. Had it been locked as a
primary endpoint, rule 4 would have required a new numbered experiment to undo.

The epistemics thread in ADR-0003 is not withdrawn. Its argument is that clinical-research
rigour is part of the contribution. That is an argument for rigour **at the point a claim is
made**, not at every exploratory run.

## Decision

Rule 1 splits into two tiers.

**Exploratory runs.** No PREREG. Live outside `experiments/`. Free to iterate, re-run,
change the analysis, and abandon. They may not be cited as evidence for a pillar claim, in a
paper or a report, however good they look.

**Formal runs.** A locked `PREREG.md` as before, in `experiments/EXP-NNN-name/`,
with rule 4 (no editing after lock) intact. A run is formal if its result is intended
to support a pillar claim externally.

Two obligations survive into the exploratory tier, because they are what pre-registration
buys and they are cheap:

1. **Log every run.** Append to `docs/run-log.md`: date, what was run, on what stimuli, what
   came out, one line. Without this the number of looks preceding a registered result is
   unrecoverable, and multiplicity is the real threat — not any single unregistered run.
2. **Write the sizing rule before expanding.** When an exploratory run is used to size a
   confirmatory one, the rule mapping the observed effect to the required n is written down
   *before* the stimuli are expanded. A confirmatory run uses expanded or new stimuli; it
   does not re-analyse the rows the exploratory run was read from.

## Consequences

- The ECOG pilot runs as exploratory work. No EXP-001 yet.
- `experiments/` stays empty until there is a claim to register, and an empty
  `experiments/` is no longer evidence that the program is not running.
- The obvious abuse — explore until something works, then register it and re-run — is
  constrained by the log and by the requirement that confirmatory runs use expanded stimuli.
  It is not eliminated. Nothing eliminates it; the log makes it visible.
- Rule 4 is unchanged and applies to locked PREREGs only.
- The PREREG template is unchanged.
