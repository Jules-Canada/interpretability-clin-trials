# ADR-0002 — Use pretrained per-layer transcoders instead of training our own

**Date:** 2026-06-09
**Status:** Accepted. Framing superseded by ADR-0003. Technical staging holds except Stage 2,
which ADR-0004 killed — Stage 1 showed no off-distribution penalty, so its trigger never fired.
**Supersedes:** ADR-0001

## Context

Two things changed after ADR-0001 was taken.

- **Gemma Scope 2** shipped around December 2025, including per-layer transcoders for
  Gemma 3 4B in both PT and IT variants.
- **`circuit-tracer` v0.3.1** (~January 2026) added an nnsight backend, so it runs against
  any Transformers model rather than the two it originally supported.

MedGemma-4B is architecturally identical to Gemma 3 4B. The combination made attribution
graphs on MedGemma reachable in a weekend on one H100, against a from-scratch path that
needed a corpus build, 1-5B tokens, and multi-GPU training before it produced anything
credible.

Separately, ADR-0001's own results argued against continuing: at 0.4x expansion the width
was not credible, and the decode target was non-standard where Gemma Scope 2 had already
made the canonical choice (MLP output at `d_model`, with affine skip connections).

## Decision

Generate attribution graphs on MedGemma using `circuit-tracer` plus Gemma Scope 2
pretrained per-layer transcoders. Stop treating coder engineering as the work.

Staged, cheap to expensive:

- **Stage 0** — validate prompt format and target tokens on hosted Neuronpedia against base
  Gemma 3 4B-IT. No custom code.
- **Stage 1** — self-host circuit-tracer + nnsight on MedGemma-4B with Gemma Scope 2 PLTs.
  Decision gate.
- **Stage 2** — warm-start domain adaptation of the transcoder, only if Stage 1 buries the
  medical computation in error nodes.
- **Stage 3** — from-scratch CLT, deferred rather than abandoned. Revisit only if a finding
  is worth tightening, or if the coder itself becomes the contribution.

Supporting calls:
- PLT not CLT at 4B. No pretrained 4B CLT exists; Gemma Scope 2's CLTs stop at 1B. Expect
  longer attribution paths. Acceptable for a first result.
- 64k width recommended, 25x expansion. 16k is the floor.
- IT model to IT transcoders to chat template, consistently. Mixing reintroduces the
  out-of-distribution failure seen earlier.

## Consequences

- `graphs/build.py` is retired. It has no skip term and the wrong decode target, so it is
  incompatible with skip-transcoders. Regression test kept for history.
- The 12GB `checkpoints/medgemma-4b-1024/clt_inference.pt` becomes a deferred artifact.
- Gemma Scope 2 transcoders were trained on Gemma 3, not MedGemma, so fidelity degrades on
  exactly the medical features of interest. The size of that gap is a measurement worth
  making, not only a nuisance.
- The scarce asset shifts to the clinical prompt corpus and the judgment behind it. Labs win
  on coder quality; this path spends effort where they cannot easily follow.

## What ADR-0003 changed

ADR-0002 correctly said the finding is the contribution, but left the finding
underspecified — three candidate behaviours ranked by intuition. ADR-0003 replaces that with
specification-grounding as the thesis and redefines the Stage 1 endpoint accordingly. The
staging above is unchanged.
