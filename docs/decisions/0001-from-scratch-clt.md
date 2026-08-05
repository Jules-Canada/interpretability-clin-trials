# ADR-0001 — Train a cross-layer transcoder from scratch

**Date:** ~2026-04 (backfilled retrospectively 2026-08; first commit 2026-04-09)
**Status:** Superseded by ADR-0002

Written after the fact from `docs/clt_era_archive.md`, `docs/diagnostics_log.md`,
`docs/pipeline_lessons.md` and the session notes. Dates in those files are authoritative
where they conflict with this summary.

## Context

The goal was to replicate the Circuit Tracing methodology (Anthropic, 2025) and apply
attribution graphs to clinical trial prompts. At the time, `circuit-tracer` supported only
Gemma-2-2b and Llama-3.2-1b, and no pretrained transcoders existed for Gemma 3. MedGemma is
a Gemma 3 fine-tune. There was no path to attribution graphs on a medical model that did
not involve training our own coder.

## Decision

Train a cross-layer transcoder from scratch and build the attribution pipeline in-house.

- Pythia-410m first as proof of concept, then MedGemma-4B.
- `n_features=1024` per layer on MedGemma (VRAM-constrained; 4096 exceeded an H100 at
  the Pythia stage).
- Decode target `d_mlp` (pre-`W_out`), not `d_model`.
- Custom `graphs/build.py` with a single frozen-nonlinearity autograd backward pass.
- Corpus: clinical trial protocol PDFs, ~120M tokens.

## Consequences

**Worked.** The autograd attribution build validated on Pythia-70m at completeness 0.8996
against a manual baseline of 0.9133, inside the ±0.02 window. The rewrite also surfaced a
real result: the manual implementation silently dropped cross-position indirect paths, and
autograd recovers them. Eight MedGemma medical-factual graphs cleared the completeness
threshold.

**Did not work.**

- **Width was undercomplete.** 1024 features against d_model=2560 is 0.4x expansion. Below
  the smallest width Google ships for this family. Caps interpretability regardless of how
  much compute goes in.
- **Corpus artifacts crowded the graphs.** 92 of 110 attributed features landed at layer 0.
  Of those 92, roughly 40 were document scaffolding (table-of-contents dot leaders,
  form-field underscores, duplicated newlines) and 2 were extraction corruption; the
  remaining ~40 were shallow medical vocabulary and ~10 generic syntactic features, both of
  which legitimately belong at layer 0. So "92/110 structural" overstates it — the real
  problem was ~40 formatting features eating the pruner budget. Layers 2-25 held healthy
  clinical features (subclavian access, HbA1c, squamous cell carcinoma), so the coder had
  learned real structure that formatting noise was crowding out. Diagnosis: corpus cleaning,
  not architectural collapse. Per `docs/ignis_L0_diagnosis_note.md`, which also notes that
  the cleaning fix was never carried through to a measured before/after.
- **Decode target was non-standard.** `d_mlp` rather than the canonical `d_model` with skip
  connections.
- **Feature labels were unvalidated and briefly wrong.** `find_top_activations.py` was
  never updated when RMS scale persistence landed, so it fed unnormalised residuals to the
  encoder, JumpReLU gated everything to zero, and the labeller produced confident
  descriptions from empty context. Caught by chance.

**Carried forward.** The debugging chain, the JumpReLU and GeGLU reconstruction work, the
L0 diagnosis, and the pipeline hygiene lessons in `docs/pipeline_lessons.md`. Also the
habit of checking completeness numerically rather than trusting that a graph renders.

## Note

This ADR is not an argument that the work was wasted. It was the only available path at the
time, it produced the diagnostic capability the project now relies on, and the failure modes
it exposed are the reason the current program treats unvalidated feature labels as wrong by
default.
