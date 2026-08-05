# ECOG v0, first run (2026-08-05) — invalid probability columns

The two files `ecog_v0_results_medgemma-4b-it.json` and
`ecog_v0_results_gemma-3-4b-it.json` were produced with a broken readout token.

## What was wrong

`resolve_canonical()` chose the readout token by asking *"is this form a single
token?"* — and `" Yes"` is one (id 8438). But after a chat template ending in
`<start_of_turn>model\n` there is no preceding space, so the model emits the
**bare** `"Yes"` (id 10784). Measured on these very files:

    p(" Yes")  ~ 0.000000    <- single token, scored against, never emitted
    p("Yes")   ~ 0.9999      <- what the model actually produces

Tokenisation and emission are different questions. Picking by the first reads a
token the model never produces.

Grades escaped the bug by accident: `" 0"` is *not* a single token in Gemma-3
(it is `'▁'` + `'0'`), so digits fell back to the bare form, which is correct.
Grade-token mass is 0.996–1.000 throughout.

## Which columns are invalid

Per row, under `eligibility`:

- `p_yes`, `p_no` — ~0.000000, meaningless
- `logit_diff` — computed on the same unemitted tokens, do not use

Everything else is sound:

- `p_yes_agg`, `p_no_agg` sum over both surface variants and DO carry the real
  mass (0.999–1.000). **These are the valid eligibility readout in these files.**
- the whole `grading` block, including `grade_dist` and `pred_grade`
- every stimulus annotation

`says` — and therefore the accuracy and dissociation summaries — survived mostly
intact, because comparing two near-zero numbers happened to preserve the sign.
Corrected values recomputed from the `*_agg` columns:

|                  | eligibility | self-consistent | AUC elig | AUC grade |
|------------------|-------------|-----------------|----------|-----------|
| MedGemma, as-written | 11/18   | 15/18           | 0.644    | 0.950     |
| MedGemma, corrected  | 11/18   | 15/18           | 0.613    | 0.950     |
| Gemma-3, as-written  |  6/18   |  9/18           | 0.400    | 0.887     |
| Gemma-3, corrected   |  5/18   |  8/18           | 0.394    | 0.887     |

The headline result is unchanged: MedGemma applies the rule faithfully to a
wrong intermediate; Gemma-3 computes a good intermediate and does not route it
into the answer (`wrong_answer_right_grade` 8x, self-consistency at chance).

The AUC contrast is the part worth carrying forward — predicted GRADE separates
eligible from ineligible at 0.95/0.89 while the ELIGIBILITY ANSWER manages
0.61/0.39. The eligibility prompt, not the models, is what needs work first.

## Fixed

`scripts/run_ecog_stimuli.py` now calls `calibrate_surface()` — one forward pass
per prompt type, comparing spaced vs bare mass — before choosing canonical ids,
and hard-fails if the chosen tokens carry <1% mass on a probe row. The chosen
forms and the probe mass are recorded under `token_ids.canonical.calibration`.

These files are kept rather than regenerated: they are the raw output that the
diagnosis rests on. **Re-run to get clean files; until then read `*_agg`, never
`p_yes`/`p_no`/`logit_diff`.**

## Check the other sweeps

`prompts/eligibility.py` documents `POS_TOKEN = " Yes"` and
`scripts/sweep_eligibility.py` sums over both variants (`pos_ids`/`neg_ids`), so
that script is probably unaffected — but its `logit_diff` averages raw logits
over BOTH variants per side, which mixes an emitted token with an unemitted one.
Worth re-checking `data/eligibility_sweep_*.json` before quoting its logit
numbers.
