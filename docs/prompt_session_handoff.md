# Prompt session handoff — 2026-05-16

Read this + auto-memory `project_prompt_design_flaw` to resume. This was a
dedicated prompt-quality session. Nothing committed; all changes are in the
working tree (see "Working tree state" below).

## One-line state

The clinical prompts are NOT the problem. The categorical set is sound. The real
blocker is **base-model output diffuseness** on `medgemma-4b-pt` — proven, not
hypothesised. No prompt rewrite fixes it. The next session's job is to choose how
to sharpen the output distribution, not to edit prompts.

## What was done

1. **Diagnosed the screening "failure."** The categorical set
   (`prompts/categorical_prompts.py::CATEGORICAL_PROMPTS`, 36 prompts / 18
   pos-neg pairs) was only passing 4/18 pairs. Root cause: the screen scored a
   single canonical token (`' Yes'`/`' No'`) while a pretrained base model
   spreads probability across `Yes/yes/YES` etc.

2. **Fixed the metric.** `scripts/screen_prompts.py` now aggregates p over
   casing variants (`p_agg`), gates on `p_agg`, and reports `p_raw` vs `p_agg`
   side by side. Also added an MPS+float16→bfloat16 auto-guard (Gemma 3 gives
   all-NaN logits in fp16 on MPS; float16 default is for CUDA pods only).

3. **Added a dynamic-range control set.**
   `prompts/categorical_prompts.py::EASY_INCLUSION_PROMPTS` (10 prompts / 5
   pairs) — trivial lexical/identity matches on the identical scaffold, to bound
   the top of the model's confidence range.

4. **Ran a real screen locally** (Mac, MPS, bfloat16, full softmax — not the
   old top-3 estimate). Outputs: `data/categorical_screen_v2.json`,
   `data/easy_screen.json`. Original `data/categorical_screen.json` kept as the
   before record.

## What was found (all evidence-backed, in the v2 JSONs)

- **Metric artifact fully explains the screening failure.** With aggregation,
  18/18 categorical pairs clear p_agg ≥ 0.2. The 4/18 was a measurement bug.

- **No dynamic range — the decisive result.** Easy controls p_agg 0.17–0.42;
  categorical 0.30–0.47. They overlap almost entirely. A near-tautology
  ("Patient: enrolled in the study. Eligible?" → Yes) scores only 0.254; its
  negative FAILS at 0.173 with `'\n'` in top-3. The model does not discriminate
  trivial from hard. The ~0.3 ceiling is `medgemma-4b-pt` (pretrained base, no
  instruction tuning) + the `Answer:` scaffold.

- **The earlier "clinical exclusion-default" idea (Effect 2) is dead.**
  Head-to-head yes vs no mass: model prefers the correct answer in only ~20/36
  (≈ chance), with a mild "No" lean on _pos prompts. But the SAME lean appears
  on trivial easy prompts with zero clinical content (e.g. `easy_age_001_pos`,
  45 y/o vs criterion ≥18, is a Yes/No tie). It's a generic base-model artifact,
  not clinical reasoning. Do not revive it.

## Implication

Attribution graphs traced toward a target the model assigns ~0.3 to (and often
doesn't even prefer) explain a near-random prediction — they will fail the
CLAUDE.md `completeness ≥ 0.5` / `p(target)` sanity bars, or look confident on
noise. The lever is sharpening the output distribution, not prompt wording.

## The open decision for the next session

Pick how to make MedGemma commit to an answer. Options (none chosen):

- **(a) Few-shot scaffold** — raises confidence, but injects in-context-learning
  / induction circuitry into every graph (contaminates the clinical-reasoning
  signal the project exists to study). The circuit-tracing paper keeps prompts
  zero-shot for this reason.
- **(b) A zero-shot scaffold the base model completes decisively** — no
  demonstrations, so no ICL contamination; needs a re-screen to prove it lifts
  margins. Lowest-risk experimental design. (Prior scaffold flip-flop:
  commits `0f188e2` → `66d04dc`.)
- **(c) Trace only the handful of prompts** where the model genuinely prefers
  the target with margin — very few; limits the scientific scope.
- **(d) Revisit `-pt` vs an instruction-tuned variant** — BLOCKED: the CLT
  (`checkpoints/medgemma-4b-1024/clt_inference.pt`) was trained on `-pt`
  weights; switching invalidates the checkpoint and needs a retrain.

Recommended first move: (b) — design 1–2 candidate zero-shot scaffolds, screen
them against BOTH `CATEGORICAL_PROMPTS` and `EASY_INCLUSION_PROMPTS`, and look
for *separation* between easy and hard (dynamic range), not just higher p_agg.

## How to re-screen (local, ~2 commands, model already cached)

```bash
.venv/bin/python scripts/screen_prompts.py \
  --prompts prompts/categorical_prompts.py --var_name CATEGORICAL_PROMPTS \
  --model_name google/medgemma-4b-pt --device auto --dtype bfloat16 \
  --min_prob 0.2 --output data/categorical_screen_v2.json

.venv/bin/python scripts/screen_prompts.py \
  --prompts prompts/categorical_prompts.py --var_name EASY_INCLUSION_PROMPTS \
  --model_name google/medgemma-4b-pt --device auto --dtype bfloat16 \
  --min_prob 0.2 --output data/easy_screen.json
```

Use `--dtype bfloat16` locally (MPS). `docs/rescreen_checklist.md` has the pod
fallback and the "read the spread, not pass/fail" guidance.

## Working tree state (uncommitted)

- `scripts/screen_prompts.py` — aggregation + MPS guard
- `prompts/categorical_prompts.py` — EASY_INCLUSION_PROMPTS added
- `docs/rescreen_checklist.md`, `docs/prompt_session_handoff.md` — new
- Pre-existing unrelated dirty files: `CLAUDE.md`, `frontend`,
  `scripts/find_top_activations.py`, `scripts/run_pipeline.sh`
- `data/*.json` screen outputs written (data/ is gitignored)
