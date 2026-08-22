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
| 1. Completeness | Does the model compute the correct guideline intermediate score? | Active |
| 2. Application | Is the inclusion decision guided by the guideline score? | Active |
| 3. Calibration | Does the model know when the spec doesn't settle the answer? | Later |

Pillar 1 tests if model calculates the guideline scores.
Pillar 2 tests if the model correctly applies the guideline score to the answer of patient inclusion
Pillar 3 tests if the model identifies when the guideline does not settle the answer

Pillar 2 measures the *coupling* between the two turns, not the accuracy of either — the
informative cell is a right answer over a wrong grade (ADR-0007). Pillar 3 is Later but not
dropped; `recist_v0.csv` already carries an `indeterminate` row that serves only it.

Plan: `docs/program/pillars-1-2-intermediate-recovery.md`. Next pod session:
`docs/program/pod-run-2026-08-plan.md` — threshold sweep at 4B/12B/27B plus
contrastive 4B graphs. A transcoder set gates *attribution graphs only*: behavioural work
and activation patching need none and run at any size. The size ladder is a control for
"the model was too small", not a route to better circuits.

---
## Rules

1. **Formal runs** need a locked `PREREG.md` in `experiments/EXP-NNN-name/`. Template
   in `experiments/`; it won't let you start without naming a baseline and a kill criterion.
   A run is formal if its result is meant to support a pillar claim externally.
   **Exploratory runs need none** — they live outside `experiments/`, iterate freely, and
   may not be cited for a pillar claim however good they look (ADR-0006). Two things carry
   into the exploratory tier: log every run in `docs/run-log.md`, and write the rule for
   sizing a confirmatory run *before* expanding the stimuli it will use.
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

Two assets, both written by hand. **Stimulus sets**: vignettes per grade per
intermediate, describing the clinical state without ever using the defining vocabulary


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
- `docs/decisions/` — ADRs. Substance is immutable: a decision, its rationale, date or
  status changes only by a new superseding ADR. Wording, terminology and typos can be fixed
  in place — git holds the prior text, so the record survives either way.
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
- `docs/decisions/0006-prereg-scope.md` — current. Rule 1 applies to formal runs only;
  exploratory runs are logged, not pre-registered.
- `docs/decisions/0007-pillar-2-application.md` — **Proposed.** Pillar 2 becomes
  Application; Legitimacy withdrawn; pillar 3 retained as Later. The table above already
  reflects it.
- `docs/decisions/0005-medicine-as-testbed-and-application.md` — **Proposed, not in force.**
  Would make medicine both testbed and application domain. Until it is accepted, 0003
  governs: medicine is the testbed, not the application. §What we're doing above is ahead
  of the ADR trail on this point — 0005 is where that gets settled, not CLAUDE.md.

---

## Where things stand


ECOG pilot ran 2026-08-06 (39 vignettes, both models). The paraphrase-generalisation gradient
is real and is the pillar-1 result: Gemma falls 100→82→64→50% with lexical distance, MedGemma
flat at ~60%. **The pillar-2 numbers from that run do not clear a constant-answer baseline**
— Gemma's eligibility 22/39 *is* the always-No baseline exactly, and coupling reaches only
p=0.21 (Gemma) and p=0.12 (MedGemma). The models carry opposite response biases (Gemma says
No 31/39, MedGemma Yes 23/39), so their similar totals come from different failures. Run
`scripts/pillar2_baselines.py` before quoting any pillar-2 number. Sizing rule for the
confirmatory run is in the pillar plan: balance the answer key, then n=61 per model.

Stage 1 answered its question and closed a door. 10 eligibility graph pairs on gemma-3-4b-it
(`frontend/graph_data/`) and medgemma-4b-it (`.../medgemma/`), completeness 0.80–0.82 and
0.84–0.85. The off-distribution penalty is absent — MedGemma scores *higher* — so Stage 2 lost
its trigger. Record that transfer result, don't headline it; it also tightens what
MedGemma-specific circuit claims can say.

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
adverse_events.py and endpoints.py (both real, both ~58-line stubs). `tests/` holds 107 offline
tests (`.venv/bin/python -m pytest`, ~0.5s, no weights, no network) over the threshold rule,
`score_rows`, the end-of-run assembly, `load_stimuli`, the sweep answer key, the readout gates,
the sizing statistics and flip-point scoring; plus one `slow` test (`pytest -m slow`, ~11s)
that checks the patching invariants against a real forward pass. The five CLT-era tests moved
to `deferred/tests/` (2026-08-05) and are not revivable. **Do not `pip install -e .` to run
them** — that pulls the CLT dependency list; `tests/conftest.py` puts `scripts/` on the path
instead. Root `graph_data/` — an earlier 2026-05-31 run superseded by the 2026-06-01 regeneration
in `frontend/graph_data/`, not a byte-identical copy — moved to `deferred/graph_data_2026-05-31/`.

## Entry points

`scripts/` is the live set only: `run_ecog_stimuli.py` (the instrument — vignettes in,
grading and eligibility out), `run_graphs_ct.py` (`--probe`, `--smoke`, then batch),
`analyze_graphs.py`, `compare_graphs.py`, `reproduce_round1.py`, `profile_criteria.py`,
`pillar2_baselines.py` (constant-answer comparator — pillar-2 numbers are meaningless
without it), `rescore_results.py` (recompute derived verdicts in a results JSON offline),
`make_sweep_stimuli.py` (crosses ecog_v0 vignettes with criteria — no new writing),
`flip_point.py` (scores the sweep; `--selftest` before trusting it),
`patch_grade.py` (residual-stream patching — needs no transcoders, so it reaches
12B/27B where graphs cannot; `--selftest` runs offline in 10s),
`setup_pod_circuit_tracer.sh`, plus `stage0_tokenizer_check.py` (kept live —
`prompts/eligibility.py` cites it as the provenance for the canonical-token choice, which
0004 still depends on).

22 CLT-path scripts are in `deferred/scripts/`. `sweep_eligibility.py` joined them
2026-08-06: it is the Stage 1 behavioural gate, and Stage 1 is closed, so it gates a
decision already made. Among the rest `find_top_activations.py` still has the 2026-06-01
RMS-scale fix; without it it silently returns zeros. Pod recipes and gotchas: `docs/ops.md`.

## Writing spec items

Schema: `specs/schema/spec_item.json`, which carries the 0004 `intermediate` block
(variable, true value, defining threshold, whether the narrative uses the defining
vocabulary) plus `attribute_policy`, since 61d8b36. Stimuli live in `specs/stimuli/`;
there is no `specs/v0.1/`.

Vignettes vary only the clinical description — hold the protocol and everything else fixed,
or the probe learns sick-versus-well, which the model has and which is not ECOG.

One canonical token for attribution (leading-space variant on IT models); aggregate case
variants for evaluation only, never for attribution — mixing them corrupts completeness. The
`TrialPrompt` lists in `prompts/` are seed material, not spec items.
