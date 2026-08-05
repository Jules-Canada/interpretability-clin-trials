# Strata

Project name is Strata, 'ignis' is historical

Keep this file under 150 lines. Docs go in `docs/` with pointer.

---

## What we're doing

High level: advance interpretability for life science and specifically clinical trials

Hypothesis: It's often easy to check if a model got the right answer. In medicine, it can be equally important how the model got the answer. We will use trial eligibility as ground truth for whether a circuit is right for the right reasons.

Read `docs/program/thesis.md` before proposing work. Kill criteria are in there. A
pre-registered null is a real result and does not need rescuing.

---

## Pillars

| | Question | Status |
|---|---|---|
| 1. Completeness | Does the model compute the clinical intermediates the spec is written in? | Active |
| 2. Legitimacy | Does a demographic attribute move the answer without moving the intermediate? | Active |
| 3. Calibration | Does the model know when the spec doesn't settle the answer? | Later |

Pillars 1 and 2 share one object: the **clinical intermediate**. Specs aren't written in raw
patient facts but in abstractions — ECOG grade, creatinine clearance, NYHA class. Pillar 1
asks whether the model has the abstraction or only the vocabulary; pillar 2 asks whether a
demographic attribute routes around it. Same stimuli, same instruments, run together.

The load-bearing test is **paraphrase generalisation** — clinical notes never use protocol
phrasing, so a model holding the ECOG table but not the concept fails silently on exactly the
population it would be deployed against. Probing and patching first, transcoders as a
dictionary before they build graphs. Plan: `docs/program/pillars-1-2-intermediate-recovery.md`.

---

## Rules

1. No experiment without a locked `PREREG.md`. Template in `experiments/`. It won't let you
   start without naming a baseline and a kill criterion. A directory without one isn't part
   of the program.
2. Baselines run the same week as the mechanistic work. Baselines run afterwards get shaped
   by what you already found.
3. Feature labels get adjudicated. Report inter-rater agreement if there's a second rater,
   document your criteria if there isn't. Don't claim reliability you didn't measure.
4. Don't edit a PREREG after lock. Open a new numbered experiment and log the deviation.
5. Check completeness numerically before calling a graph valid. Rendering in the frontend
   proves nothing.
6. Annotate tensor shapes in comments.
7. All figures go through `viz/`.

On rule 3: `data/deferred/feature_labels.jsonl` was once populated from an empty-context bug and the
labeller produced confident nonsense that nearly shipped. Assume an unvalidated label is
wrong.

---

## The bottleneck is clinician time, not compute

Two assets, both written by hand. **Stimulus sets**: ~25 vignettes per grade per
intermediate, describing the clinical state without ever using the defining vocabulary, plus
a deliberately lexically-distant held-out set — that held-out set *is* the experiment.
**Spec items**: the annotated intermediate, threshold, and attribute policy behind each call.

Graphs and transcoders are replaceable. These aren't. When writing and GPU work compete,
writing wins. Don't queue compute that runs ahead of the stimuli it needs.

---

## Not doing

- Training a better transcoder, or domain-adapting one. Stage 2 is killed (ADR-0004).
- Only showing a medical model has medical features. Crowded lane, expected result.
- Generic MedGemma vs Gemma 3 diffing, unless scoped to a pillar.

---

## Layout

- `docs/program/` — thesis, pillar plans
- `docs/decisions/` — ADRs, immutable, superseded rather than edited
- `docs/ops.md` — pod setup, scripts, environment gotchas
- `specs/` — item schema, versioned snapshots, adjudication records
- `experiments/EXP-NNN-name/` — PREREG, run, results, report
- `deferred/` — superseded code and docs, kept not deleted

Spec snapshots freeze once a PREREG points at them.

---

## History

- `docs/decisions/0001-from-scratch-clt.md` — CLT era. Backfilled 2026-08; the session notes
  it cites are authoritative where they disagree.
- `docs/decisions/0002-pretrained-plts.md` — pivot to pretrained PLTs. Staging holds except
  Stage 2, killed by 0004.
- `docs/decisions/0003-specification-grounding.md` — the thesis. Holds; pillar-1 method
  replaced by 0004.
- `docs/decisions/0004-intermediate-recovery.md` — current. Intermediate recovery, not
  criterion-node search; transcoders as dictionary before graph.
- `docs/decisions/0005-medicine-as-testbed-and-application.md` — **Proposed, not in force.**
  Would make medicine both testbed and application domain. Until it is accepted, 0003
  governs: medicine is the testbed, not the application. §What we're doing above is ahead
  of the ADR trail on this point — 0005 is where that gets settled, not CLAUDE.md.

---

## Where things stand

Stage 1 answered its question and closed a door. 10 eligibility graph pairs on gemma-3-4b-it
(`frontend/graph_data/`) and medgemma-4b-it (`.../medgemma/`), completeness 0.80–0.82 and
0.84–0.85. The off-distribution penalty is absent — MedGemma scores *higher* — so Stage 2 lost
its trigger. Record that transfer result, don't headline it; it also tightens what
MedGemma-specific circuit claims can say.

Under 0004 nothing is built: no stimulus sets, no `intermediate` block in the schema, no spec
snapshot, no PREREG. **The blocker is vignette writing, not GPU.** First compute is the 4–6
boundary-pair graph batch, then probes and feature search — forward passes, not a big pod run.

Two Round-1 claims failed re-verification (`scripts/reproduce_round1.py`, offline, ~3 min):
MedGemma is *more* early-layer weighted, not less (late influence 11% vs Gemma's 16%), and
the cross-model jaccard of 0.13–0.19 is size-confounded — ~1780 unique Gemma features against
~1200 MedGemma ones caps jaccard near 0.67. Don't repeat "reorganized ~85%" or "more
late-layer features" without a size-matched comparator. Separately,
`frontend/graph_data/elig_age_pos.json` is 0 bytes (regenerable) and the MedGemma sweep JSON
was never copied back.

## Stack

The active path installs nothing from `pyproject.toml`: `setup_pod_circuit_tracer.sh` pulls
`circuit-tracer` from git plus `nnsight` and cu121 torch, all unpinned. `pyproject.toml`
still describes the CLT era (`transformer_lens`, `h5py`, `datasets`, `wandb`, `torch>=2.2`)
and still names the package `ignis`. Python >=3.10, pytest `-m slow` opt-in.

## Layout on disk

`clt/ graphs/ interventions/` — CLT era, superseded, now imported by nothing. `prompts/` —
eligibility.py (617 lines, live), categorical_prompts.py, medical_knowledge.py, plus
adverse_events.py and endpoints.py (both real, both ~58-line stubs). `tests/` is empty: all
five tests were CLT-era and moved to `deferred/tests/` (2026-08-05), so there is no live test
suite. Root `graph_data/` — an earlier 2026-05-31 run superseded by the 2026-06-01 regeneration
in `frontend/graph_data/`, not a byte-identical copy — moved to `deferred/graph_data_2026-05-31/`.

## Entry points

`scripts/` is now the live set only: `run_graphs_ct.py` (`--probe`, `--smoke`, then batch),
`sweep_eligibility.py`, `analyze_graphs.py`, `compare_graphs.py`, `reproduce_round1.py`,
`profile_criteria.py`, `run_ecog_stimuli.py`, `setup_pod_circuit_tracer.sh`, plus
`stage0_tokenizer_check.py` (kept live — `prompts/eligibility.py` cites it as the provenance
for the canonical-token choice, which 0004 still depends on). The 21 CLT-path scripts moved to
`deferred/scripts/` on 2026-08-05. Among them `find_top_activations.py` still has the
2026-06-01 RMS-scale fix; without it it silently returns zeros. Pod recipes and gotchas:
`docs/ops.md`.

## Writing spec items

Schema: `specs/schema/spec_item.json`, which still needs the 0004 `intermediate` block
(variable, true value, defining threshold, whether the narrative uses the defining
vocabulary). Cut `specs/v0.1/` before the first PREREG points at it.

Vignettes vary only the clinical description — hold the protocol and everything else fixed,
or the probe learns sick-versus-well, which the model certainly has and which is not ECOG.

One canonical token for attribution (leading-space variant on IT models); aggregate case
variants for evaluation only, never for attribution — mixing them corrupts completeness. The
`TrialPrompt` lists in `prompts/` are seed material, not spec items.
