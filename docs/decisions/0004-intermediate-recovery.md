# ADR-0004 — Intermediate recovery as the primary method; transcoders as dictionary before graph

**Date:** 2026-08-04
**Status:** Accepted
**Amends:** ADR-0002 (Stage 2 is killed), ADR-0003 (thesis holds, pillar 1 method changes)

## Context

Three things came together.

**Stage 1 answered its question and the answer was negative.** Graph completeness on
MedGemma-4B with Gemma Scope 2 transcoders is 0.84–0.85, against 0.80–0.82 on
gemma-3-4b-it. The off-distribution penalty is not merely tolerable, it is absent —
MedGemma scores slightly higher. ADR-0002 made Stage 2 conditional on the medical
computation being buried in error nodes. It is not.

**The pillar-1 method as written in ADR-0003 was underspecified in a way that flattered
attribution graphs.** "Does the circuit contain a causally-validated representative for each
spec condition" assumes feature-level claims of a kind the tooling does not reliably
deliver. Feature labels are LLM-generated from top activations; searching a pruned graph for
a node labelled with the criterion you are looking for is confirmation, not measurement.
The repo has already been burned once by trusting labels (`data/feature_labels.jsonl`,
ADR-0001).

**A worked example reframed the question.** A protocol requiring ECOG ≤1, against a patient
described as walking unassisted but unable to work. The correct answer is ineligible,
because the 1/2 boundary is exactly *able to do light work* versus *unable to work*. Getting
there requires computing an intermediate — performance status grade — and comparing it to a
threshold. The protocol names that intermediate.

That is the sharper reading of specification-grounding. The spec does not merely list
conditions; it names the clinical abstractions the model is supposed to compute. Whether the
model has those abstractions, or only the vocabulary they are usually described in, is a
question about a variable, and variables are what probing and activation patching handle
well.

## Decision

**1. Pillar 1 becomes intermediate recovery.**

Primary question: does the model represent the clinical intermediates its specifications are
written in — ECOG, creatinine clearance, NYHA class, RECIST measurability — or does it
lexically match the defining vocabulary?

The load-bearing test is paraphrase generalisation. Trial screening runs on clinical notes
that never use protocol phrasing. A model holding the ECOG table but not the concept fails
silently on exactly the population it would be deployed against.

**2. Pillar 2 becomes bypass.**

Restated: does a demographic attribute move the answer while leaving the intermediate
unchanged? If race shifts eligibility without shifting the performance-status
representation, it is bypassing the clinical computation. This is better defined than
"direct path to the answer logit" and runs on the same instruments as pillar 1.

**3. Transcoders are used as a dictionary before they are used to build graphs.**

Running the stimulus set through MedGemma with Gemma Scope 2 transcoders attached, and
asking whether any existing feature tracks the intermediate, is a forward pass and a
correlation. No graph construction, no pruning, no export. It scales to hundreds of prompts.

It is also better evidence than a probe. A probe is fitted to labels and will find
something; the standard objection is that it reads information the model has available but
does not use. A pre-existing transcoder feature that tracks the intermediate across
paraphrases, and whose ablation flips the answer, is evidence the model already carried the
variable. Probe and feature search run in parallel and cross-validate: probe succeeds where
feature search fails means the variable exists but is not in the dictionary at this width.

**4. Attribution graphs run early but small, then late and contingent.**

A batch of four to six graphs on intermediate-boundary pairs, immediately after the stimulus
set exists. Two purposes: confirm the pipeline runs on the new prompt format while knowledge
of it is current, and check the one thing only a graph shows — whether narrative tokens
reach the answer through an intermediate node or bypass it. If they bypass, that reframes
the probe work before it is done.

Exhaustive graph reading stays late and contingent, reserved for cases where cheap methods
disagree.

**5. Stage 2 is killed.** No domain adaptation of the transcoder. Stage 3 remains deferred
under ADR-0002.

## Consequences

- The first result does not require a custom coder, and may not require attribution graphs
  at all beyond validation. This is an acceptable outcome and should be stated plainly
  rather than hidden.
- Stimulus construction becomes the critical path alongside adjudication. Roughly 25
  vignettes per grade per intermediate, plus a lexically-distant held-out set. Clinician
  time again, not GPU.
- The legitimacy taxonomy shrinks in scope. It is only load-bearing where a demographic
  attribute is a candidate for bypass, or where it sits inside the intermediate's own
  formula — the eGFR race coefficient, Cockcroft-Gault sex term, FEV1 reference equations,
  MELD 3.0. Those four are dated, documented controversies and make a tighter starting set
  than a general attribute pass.
- `specs/schema/spec_item.json` needs an `intermediate` block: variable name, true value,
  defining threshold, and whether the narrative uses the defining vocabulary. Addition, not
  redesign.
- Transfer of Gemma Scope 2 transcoders to MedGemma with no fidelity loss is a minor finding
  in its own right, and constrains what the pillars can claim: if medical fine-tuning moved
  the representations that little, MedGemma-specific circuit claims get harder to make.
  Record it, do not headline it.

## Rejected

**Continue with graph-first pillar 1.** The pipeline works and the sunk cost is real, but
searching pruned graphs for expected labels is the failure mode this program's epistemics
exist to prevent.

**Drop transcoders entirely and run a probing study.** Cheaper and faster, but gives up the
one advantage over a generic probing paper — a named, pre-existing, ablatable feature is a
different claim from a fitted direction. It also lets a working pipeline and current
operational knowledge go cold for no gain.
