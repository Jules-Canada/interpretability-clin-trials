# ignis — Current Approach (Handoff)

**Status:** This supersedes the from-scratch-CLT plan. The earlier
`clt_feature_count_handoff.md` math is still valid and is referenced below, but the *plan*
has changed.

Base model: MedGemma-4B (Gemma 3 4B fine-tune; L=34, d_model=2560, d_mlp=10240).

---

## TL;DR — the pivot

Stop training a coder from scratch. Generate attribution graphs on MedGemma directly with
existing tooling — Anthropic's `circuit-tracer` (v0.3.1+, nnsight backend) + Gemma Scope 2's
pretrained Gemma 3 4B transcoders. Spend effort on a **causally-validated clinical finding**,
not on coder engineering. Method = instrument; finding = paper.

---

## Why now (timeline, grounded)

- `circuit-tracer` open-sourced **29 May 2025** — supported Gemma-2-2b and Llama-3.2-1b only.
  Could not touch MedGemma (Gemma 3).
- **Gemma Scope 2** (Gemma 3 transcoders, incl. 4B) released ~Dec 2025.
- `circuit-tracer` **v0.3.1** (~Jan 2026) added nnsight backend → works on *any* Transformers
  model, including Gemma 3 PLTs (PT & IT) for 270M–27B.

So the cheap MedGemma path only became viable Dec 2025–Jan 2026. Building from scratch was
reasonable before that. It is not the right starting point now.

**Comparative advantage:** labs win on coder quality. The scarce assets here are clinical
judgment + the prompt corpora — both of which this path uses directly.

---

## Paper framing

NOT "we trained a coder on a medical model" (expected, unexciting; Anthropic already published
a medical-diagnosis circuit). INSTEAD: a mechanistic, causally-validated account of a
clinically consequential behavior in MedGemma, framed as the kind of evidence
regulators/assurance teams need. Candidates, ranked by noticeability × clinician edge:

1. A causally-validated **clinical failure mode** (circuit rides on a spurious feature;
   intervene to confirm; validate against held-out cases).
2. **Faithfulness** of stated clinical reasoning (is the explained differential the real one?).
3. **Hallucination vs. abstention** boundary (confabulated contraindication / drug interaction).

---

## Staged plan (cheap → expensive)

**Stage 0** — hours, ~free. Run `circuit-tracer` on base Gemma 3 4B-IT (hosted Neuronpedia is
fine). Reproduce a known graph, then push 2–3 clinical prompts through. Validates prompt format
+ target-token setup with zero custom code.

**Stage 1** — a weekend, 1× H100, ~$10–30. Self-host `circuit-tracer` + nnsight, load
MedGemma-4B, apply Gemma Scope 2 4B transcoders (16k or 64k width fits alongside the ~8 GB model
on one 80 GB card). Run all 71 prompts. Deliverable: first MedGemma attribution graphs + a
measured off-distribution reconstruction/completeness gap. **Decision gate.**

**Stage 2** — days, 1–2× H100, *only if* Stage 1 buries the medical computation in error nodes.
Warm-start the Gemma Scope 2 transcoder, lightly domain-adapt on existing ~120–500M medical
tokens (enough because it's adaptation, not from-scratch). Rebuild graphs.

---

## Technical facts that justify the choices

- **Decode target resolved.** Gemma Scope 2 transcoders reconstruct the MLP **output**
  (`d_model`) with affine skip connections. That's the canonical choice, so the old
  `d_mlp`-vs-`d_model` debate is moot — we adopt the artifact that already chose correctly.
  Note: our old `graphs/build.py` is *incompatible* with skip-transcoders (no skip term, wrong
  decode target) — the library replaces it.
- **PLT, not CLT, at 4B.** No pretrained 4B CLT exists (Gemma Scope 2 CLTs stop at 1B). Expect
  longer attribution paths / more features than a CLT would give. Fine for a first paper;
  "a CLT tightens this" is the future-work hook.
- **Width / expansion (d_model=2560):** 16k = 6.4×, 64k = 25× (recommended), 256k = 100×.
  Our old 1024 = 0.4× (undercomplete) — not credible. 4096 = 1.6× — still below the smallest
  width Google shipped for this model family.
- **Off-distribution caveat.** Gemma Scope 2 transcoders were trained on base/IT Gemma 3, not
  MedGemma. Running on the medical fine-tune will degrade fidelity on exactly the
  medical-specific features. The *size* of that gap is itself a publishable finding.

---

## On training our own CLT — deferred, not impossible

We *can* train one; "can't" was wrong. It's the expensive corner, for three reasons:

1. **No warm-start at 4B** → necessarily from-scratch.
2. **Credible width from-scratch needs compute + tokens:** ~26B params at ≥16k / `d_model`
   decode (2–4× H100) and ~1–5B training tokens. We're at ~120M — the token gap is the real
   bind, forcing a corpus build before the run is worth doing.
3. **It's a refinement, not a prerequisite** — buys tighter graphs over the PLT, not graphs
   per se.

Revisit it when: (a) Stage 1/2 confirm a finding worth tightening, or (b) the CLT *itself*
becomes the contribution (a "MedScope" open-artifact play). Sequenced to Stage 3, not killed.

---

## Asset inventory

- **Reused intact:** the 71 contrastive clinical prompts, `TrialPrompt` schema, token
  normalization. (The scarce, expensive, clinician-built part — no library provides it.)
- **Deferred:** `clt/model.py`, `train_clt.py`, the `d_mlp` CLT checkpoint, custom
  `graphs/build.py`. Capability/learnings (L0 collapse, JumpReLU, GeGLU reconstruction) carry
  over to any later custom training and to your credibility as an interp hire.

---

## Open decisions (live)

1. **Which clinical behavior to graph first** — differential diagnosis, drug-interaction /
   contraindication recall, or known-vs-unknown-entity (hallucination/abstention). Pick the one
   where you're the authority on whether the circuit's behavior is correct or dangerous. This,
   not compute, is now the bottleneck. Once chosen → spec exact prompts + target tokens for
   Stage 1.
2. **Methods vs. findings paper?** (Findings pushes toward 64k width + domain adaptation;
   methods can stop at the minimum-credible 16k.)
3. **Is the CLT itself part of the contribution**, or just a means to graphs? (Determines
   whether Stage 3 is mandatory or optional.)
4. **Token sourcing** — if Stage 2 needs MIMIC for tokens, training moves to GCP (PhysioNet DUA;
   no third-party APIs), which decides the compute venue independent of cost.

---

*Grounded:* timeline dates; circuit-tracer arbitrary-model support + Gemma 3 PLT packaging;
Gemma Scope 2 decode target (MLP output) + skip connections + widths + CLTs only at 270M/1B;
expansion-factor arithmetic; param formulas.
*Inferred / to verify in practice:* single-H100 fit at 64k width alongside MedGemma; the degree
of off-distribution degradation on MedGemma; whether the IT or PT transcoder variant transfers
better to MedGemma.
