# Writing vignettes

Working guide for the hand-written stimulus sets.

**Current job: bring `ecog_v0.csv` to pilot size.** ECOG goes first because it is furthest
along and already has the paraphrase-set structure the other two lack. mRS and RECIST get
sized *after* the ECOG pilot reports an effect size — see "Why ECOG first" at the bottom.

Read `docs/program/pillars-1-2-intermediate-recovery.md` first. This is the practical
companion, not a substitute.

---

## What the vignettes have to do

We are testing whether the model **computes the clinical abstraction** the protocol is
written in, or merely recognises the vocabulary. A model that has memorised the ECOG table
but not the concept looks competent on anything phrased like the ECOG table, and fails on
real clinical notes — which never use that phrasing. That population is exactly where it
would be deployed.

Everything rests on one property: **the vignette describes the patient's state without ever
naming the scale or the grade.** If the narrative says "ECOG 2", the model can look it up
and we have measured nothing.

---

## The rule that matters most

**Never use the defining vocabulary.** For ECOG: no *ECOG*, no *performance status*, no
*Karnofsky*, and no bare grade number presented as the score.

The instrument enforces this — `--dry-run` warns on any vignette naming its own scale.
ECOG currently has **zero** violations. Keep it that way; it is the set's main advantage
over `mrs_v0.csv`, where 9 of 18 rows name the scale.

Note the existing `definition_verbatim` rows (E001, E002, E003, E012). They quote the ECOG
definition wording *without* naming a grade — a legitimate anchor condition, and different
from mRS's `direct_label` rows, which state the score outright. Don't add more; four is
enough.

---

## The 1/2 boundary is the experiment

The protocol threshold is ECOG ≤ 1, so the decisive contrast is grade 1 against grade 2.

What actually separates them:

| | ECOG 1 | ECOG 2 |
|---|---|---|
| Ambulatory | **yes** | **yes** |
| Manages all self-care | **yes** | **yes** |
| Can do light or sedentary work | **yes** | **no** |
| Up and about >50% of waking hours | yes | yes |

**Both walk. Both self-care.** That is the trap. A vignette that separates 1 from 2 by
describing washing, dressing, or walking difficulty is not testing this boundary — it is
testing grade 3, where self-care genuinely fails. At the 1/2 line, hold mobility and
self-care constant and vary **only capacity for light work**.

E008 is the model to copy: *"Patient able to walk unassisted but not able to work."*
Ambulation fixed, work capacity varied.

The traps that make this hard, and which the set should contain:

- **Not working ≠ grade 2.** Retirement, redundancy, or caring for a relative are
  non-clinical reasons. E016 covers this; write more.
- **Sounding unwell ≠ grade 2.** Fatigue and low mood with unrestricted activity is still
  grade 1. E015 covers this.
- **A walking aid ≠ restriction**, if it predates the disease. E018 covers this.

The other boundaries, for reference:

- **0 vs 1** — 0 has no restriction at all; 1 is restricted in *strenuous* activity only.
- **2 vs 3** — 2 manages all self-care and is up >50% of the day; 3 is limited in self-care
  and confined to bed or chair >50%.

---

## Pilot target

The pilot's job is to estimate the effect size, not to be the definitive study. Once it
reports, mRS and RECIST get sized from evidence.

| Grade | target | status |
|---|---|---|
| 0 | 6 | met |
| **1** | **12** | met |
| **2** | **12** | met |
| 3 | 5 | met |
| 4 | 4 | met |
| **total** | **39** | **met (2026-08-05)** |

The pilot set is written. What it still needs is clinician adjudication of the grades, not
more rows.

A separate "held-out" set was written and then deferred the same day — see
`deferred/stimuli/README.md`. Two reasons: the term was wrong (nothing is trained, so there
is no holdout), and it was not measurably distant. Paired rows shared 4x more vocabulary
than unrelated ones, and the discriminating stems survived the rewrite. What changed was
register, not the words the grade turns on. Same author, same source.

---

## Paraphrase sets carry the generalisation test

A paraphrase set is **one patient described twice**, sharing a `set_id` and a true grade:
once as `paraphrase` (plain clinical restatement) and once as `inferred_symptoms`
(lexically further, the grade must be inferred from described function).

Because the grade is fixed by construction, any change in the model's answer across a set is
caused by wording alone. That is the paraphrase-generalisation measurement, and it is a
paired within-patient comparison — no worrying that one group of patients was simply harder,
because they are the same patients.

All 12 sets (S01–S12) are complete pairs inside `ecog_v0.csv`, 24 rows, spanning grades 0–4.
Any new patient written for the boundary grades should be written as a set, not as a
standalone row.

**Known limitation.** The gap between `paraphrase` and `inferred_symptoms` is a smaller
manipulation than protocol-language versus real-clinical-note language. So this tests that
the model is not keying on one exact phrasing; it does **not** establish that the model
survives genuine clinical notes, which is the deployment-relevant claim. Making that claim
needs either a proscribed-word list for a distant condition or text from a different source
— see `deferred/stimuli/README.md`. Do not write the stronger claim off this set.

---

## `lexical_distance` values

Easy → hard:

- `direct_label` — states the grade outright. ECOG has none. **Don't add any.**
- `definition_verbatim` — quotes the scale's defining wording, names no grade.
- `paraphrase` — plain clinical restatement.
- `inferred_symptoms` — grade must be inferred from described function. **The bulk.**
- `distractor` — baits a wrong route; set `distractor_type`.

Current spread: 5 / 11 / 15 / 8. Any new rows should be mostly `inferred_symptoms`, written
as the second member of a paraphrase set rather than standalone.

---

## Distractors

Have 4, want ~6. Existing types are `symptom_severity_substitution`,
`non_clinical_work_status`, `surface_wellness_bait`, `unrelated_comorbidity`. Worth adding:

- `reversed_threshold` — same vignette, protocol flipped to ECOG ≥ 2 (a supportive-care
  cohort). Tests whether the model reads the criterion or pattern-matches "ECOG ≤ 1 is
  standard". mRS has one of these; ECOG has none.
- `cross_domain_scale_substitution` — an mRS or NYHA value planted in an oncology vignette.
- `temporal_ambiguity` — performance status before treatment versus now.

---

## What makes a bad vignette

- **Sick-versus-well.** If grade correlates with how ill the patient sounds, the probe
  learns illness severity — which the model certainly has, and which is not ECOG. Two
  patients at different grades should be able to sound equally unwell.
- **Varying the protocol.** Hold the criterion text fixed. Only the clinical description
  changes — except for `reversed_threshold` distractors, where flipping it is the point.
- **Self-care or mobility at the 1/2 line.** The most common way to write a pair that looks
  decisive and isn't.
- **Structural tells.** Always listing three deficits for grade 2 gives a countable cue.
- **Silent ambiguity.** If two clinicians would disagree, mark `ambiguous` rather than
  picking one. Three rows are already marked; those are excluded from headline grading
  accuracy and reported separately.

---

## Columns

`id, ecog_true, set_id, lexical_distance, boundary_case, ambiguous, vignette_text,
eligibility_criterion, expected_answer, distractor_type, notes`

- `boundary_case` — `True` for grades 1 and 2, and for distractors near the line.
- `expected_answer` — `eligible` / `excluded`.
- `eligibility_criterion` — `ECOG <= 1 required for enrollment`, fixed except for
  `reversed_threshold` rows.
- `notes` — say why the row exists, especially what a distractor baits.

**Quote any field containing a comma.** An unquoted comma silently shifts every column
after it; one RECIST row was dropping out of scoring entirely before this was caught.

---

## Before calling it done

```bash
python scripts/run_ecog_stimuli.py --dry-run --stimuli specs/stimuli/ecog_v0.csv
```

Check: the row count matches what you wrote; no vocabulary warnings; the grade-4 warning is
gone; paraphrase sets show consistent grades.

---

## Why ECOG first

The honest reason is that we don't yet know which study we're running. If the model
collapses on distant wording, the effect is large and needs few vignettes to establish. If
it merely drifts, the required set is several times larger and the whole measurement may
need rethinking. Those two regimes differ by roughly 5× in required sample size.

So the pilot is sized to tell the regimes apart, not to settle the question. mRS and RECIST
get written once we know which one we're in — which avoids writing 100 vignettes per scale
for a design that may not be the right one.

**Pre-register the pilot as a pilot**, with the rule for sizing the full study stated up
front. Rule 4 forbids retrofitting a PREREG after seeing results, so "run the pilot, then
size by this stated rule" has to be written down before the run, not decided after.
