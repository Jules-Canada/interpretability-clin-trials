# Pipeline Lessons — Pythia-410m Run 1

Issues discovered after the first full training run (April 2026, Lambda H100).

---

## 1. n_features was hardcoded wrong in run_pipeline.sh

`run_pipeline.sh` had `n_features=4096` throughout, but the actual run used 2048 because 4096 exceeded H100 VRAM (81GB needed vs 79GB available). The script would have failed on re-run.

**Fix:** Added `N_FEATURES=2048` as a single variable at the top of `run_pipeline.sh`. All flags now reference `$N_FEATURES`.

---

## 2. The pipeline stopped too early — feature labeling prep was missing

`run_pipeline.sh` ended after graph generation. But `find_top_activations.py` requires the HDF5 file (which lives on the instance and is not scp'd home), so it *must* run before termination. This step was not in the script.

**Result:** `feature_activations.jsonl` was never generated before the instance was terminated.

**Fix:** Added Step 4 to `run_pipeline.sh`: strips optimizer state → `clt_inference.pt`, runs `collect_graph_features.py`, then runs `find_top_activations.py`. All three happen on the instance while HDF5 is available.

---

## 3. graph_features.json was stale

`data/graph_features.json` contained features from only `france_capital` and `water_boil` (the two test graphs used during development). It was never regenerated after the 14 clinical trial graphs were produced.

**Result:** Even if `find_top_activations.py` had run, it would have scanned the wrong features.

**Fix:** `collect_graph_features.py` is now called in Step 4 of `run_pipeline.sh` (after graph generation), so `graph_features.json` is always regenerated from the current graph directory before find_top_activations runs.

---

## 4. No executable pre-termination script

The pre-termination checklist was markdown in CLAUDE.md — manual, no enforcement. Easy to skip steps under time pressure.

**Fix:** Added `scripts/pre_terminate.sh`. Run it on the instance if the pipeline was interrupted or run in parts. It idempotently handles: strip checkpoint, collect graph features, find top activations, then prints the exact scp commands to run. One script, one invocation, ready-to-copy output.

---

## 5. Checkpoint files landed in wrong locations

`clt_inference.pt` ended up in `data/` rather than `checkpoints/pythia-410m-2048/`. The two locations existed in parallel with no clear authority.

**Fix:** `run_pipeline.sh` and `pre_terminate.sh` both write `clt_inference.pt` to `$CHECKPOINT_DIR/`. The scp command in the end message reflects this path.

---

## What's still manual (intentionally)

- **label_features.py runs locally** — it only needs `feature_activations.jsonl` and the Claude API. No GPU or HDF5 required. Run with `--resume` to restart safely after interruption.
- **HDF5 stays on the instance** — re-extracting is cheap (~$0.10, ~1 min on A10). Not worth the bandwidth to scp 20GB home.
