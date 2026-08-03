# Where we were — re-entry note (last active 2026-06-20)

Written to let you (or a fresh session) restart cold after a ~3-week gap. For the
big picture see `CLAUDE.md`; for the Round-1 result see
`docs/round1_eligibility_summary.md`. This note is the "what's the state and what
do I do next" layer between them.

---

## The one-paragraph state

The active pipeline works and produced a real, defensible finding. We're **between
Round 1 (done, validated) and Round 2 (designed, committed, not yet run)**. Round 2
needs one pod session. Everything up to that session is done and on `main`; nothing
is half-finished in the tree (`git status` is clean apart from three untracked
scratch docs).

---

## What the project is (30-second reminder)

- **Goal:** replicate Anthropic's circuit-tracing method, apply it to clinical-trial
  **eligibility reasoning**, find a causally-validated clinical finding. Dual purpose:
  mech-interp contribution + portfolio for a clinical-AI career pivot. Target ML4H/CHIL.
- **Active method (the "Stage 1" path):** `circuit-tracer` (nnsight backend) +
  **Gemma Scope 2** pretrained per-layer transcoders
  (`mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine`), run on
  **gemma-3-4b-it** (general) and **medgemma-4b-it** (medical). From-scratch CLT training
  is deferred, not the path. Rule: IT model → IT transcoders → chat template.

---

## What's DONE and solid (Round 1)

1. **Pipeline validated** — first completeness-passing attribution graphs after the
   CLT era. 10 gemma + 10 medgemma eligibility graphs on disk
   (`frontend/graph_data/elig_*.json`, `frontend/graph_data/medgemma/*.json`),
   completeness 0.80–0.85 (all above the 0.5 gate).

2. **The finding — over-exclusion, robust across a general AND a medical model.**
   Both models **wrongly exclude eligible patients** on knowledge-dependent exclusion
   criteria (pemetrexed→No, stage III→No, when they should be Yes), while correctly
   excluding the true-exclusion arms. Visible behaviorally (the sweep) AND in the
   circuits (`compare_graphs.py`: the deciding token barely moves the graph for
   knowledge pairs vs controls). Medical fine-tuning **reorganized ~85% of the circuit
   yet did not fix the bug** (cross-model overlap only ~0.13–0.17 jaccard). That
   cross-model robustness is the headline.

3. **Stage 0 age-bound lead was killed** — it was a raw-cloze/OOD artifact; under the
   real chat template the model reads age fine. Do NOT cite it.

Full detail: `docs/round1_eligibility_summary.md`.

---

## The open problem Round 1 left (why Round 2 exists)

Round 1's finding has **one honest confound**: the clean pairs were all *inclusion*
criteria and the failing pairs all *exclusion*. So "the model fails on **knowledge**"
and "the model has an **exclusion-phrasing → No** bias" were not separated. That's the
first question a reviewer asks, so it has to be settled.

---

## What's READY but NOT YET RUN (Round 2 — designed & committed 2026-06-20)

The last three commits set this up. Nothing to redesign — just run it.

- **Redesigned prompts** (`prompts/eligibility.py`), corpus-grounded (profiled 2,545
  real ClinicalTrials.gov eligibility sections — carve-outs in ~44% of protocols,
  numeric thresholds ~76%; our old prompts hit ~0% on these axes). Three tiers:
  - `ELIGIBILITY_PAIRS` — **8 core graph pairs = a complete 2×2**
    {inclusion,exclusion}×{surface,knowledge} (the two new cells break the confound),
    plus two headline **exception/carve-out** pairs: `gilbert` (STATED carve-out, no
    knowledge) vs `malignancy` (KNOWLEDGE carve-out). All ≤88 tokens → graphable.
  - `ELIGIBILITY_PAIRS_EXTENDED` — ecog, creatinine (×ULN), temporal window.
  - `ELIGIBILITY_BEHAVIORAL` — conjunction-aggregation, nested liver-mets carve-out;
    **forward-pass only** (too long to graph — length dilutes completeness).
- **Sweep wired** (`scripts/sweep_eligibility.py`) — runs `ELIGIBILITY_ALL` grouped by
  2×2 cell, prints a **phrasing×inference confound read**, and now has a **contrastive
  `logit(Yes)−logit(No)` column**.
- **Corpus profiler** committed (`scripts/profile_criteria.py`).

---

## DO THIS FIRST when you next have pod time

Order matters — the cheap step may answer the question before you spend GPU on graphs.

1. **Rent H100, set up:** `bash scripts/setup_pod_circuit_tracer.sh` (see CLAUDE.md
   Pod Setup — active path). No corpus/HDF5/checkpoint scp needed on this path.
2. **Behavioral sweep, both models (cheap, forward-pass):**
   `python scripts/sweep_eligibility.py --model google/gemma-3-4b-it` then
   `--model google/medgemma-4b-it`. **Read the confound line at the end:**
   - `prior_chemo`/`gilbert` (exclusion×surface) clean but `prior_tx`/`stage`/
     `malignancy` failing → **knowledge gap**.
   - exclusion×surface *also* failing → **exclusion-phrasing bias**.
   This may settle the confound **without new graphs.**
3. **Graph the 16 core pairs** (only if the sweep says graphing adds signal):
   `python scripts/run_graphs_ct.py --model google/gemma-3-4b-it --transcoders mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine`,
   then repeat for `google/medgemma-4b-it`. The `gilbert` vs `malignancy` contrast
   isolates "can't apply a stated exception" from "lacks the knowledge to match it."
4. **scp back** `frontend/graph_data/*.json` + `data/eligibility_sweep_*.json`.
   ⚠️ paste pod output **to Claude, not into your local shell** (an accidental paste
   once truncated a local graph file to 0B).

## No-GPU work available anytime (doesn't need a pod)

- **Feature labeling** of the shared "exclude" scaffold + the candidate Yes-feature
  (gemma L14 f7595238) and MedGemma's late-layer features — turns the structural
  circuit claims into semantic ones. Needs the graph JSONs (already on disk).
- **Draft the writeup** — the cross-model-robust over-exclusion is the result.
- **Visual abstract** for a non-ML clinical audience.

---

## Gotchas that cost time last time (so they don't again)

- Pod HF cache must go to `/workspace` (20GB root disk fills) — the scripts now
  `setdefault` `HF_HOME`, but verify in an interactive shell.
- `transcoders` is a **required positional** arg to `ReplacementModel.from_pretrained`,
  and needs the full `/transcoder_all/width_16k_l0_small_affine` subpath.
- `--max_feature_nodes 8192` is set to avoid the `torch.sort` INT_MAX prune crash.
- We deliberately do NOT write `graph.to_pt` (fills root disk) — only the frontend JSON.
- Pod cost/time estimates run 2–3× over raw GPU time (setup + scp + debugging).

---

## Untracked scratch (not committed, safe to ignore or clean)

`docs/clt_feature_count_handoff.md`, `docs/corpus_experiment.md`,
`docs/ignis_L0_diagnosis_note.md` — CLT-era notes, superseded by the active path.
