# Run log

Append-only. One line per run, exploratory or otherwise.

Required by rule 1 / ADR-0006. Exploratory runs do not need a PREREG, but they do need to be
visible: a single unregistered run is not the threat, an unrecorded *number* of them is.
Without this the multiplicity behind any later registered result is unrecoverable.

Log the run even when it fails, and especially when the result is uninteresting — those are
the ones that go unrecorded and inflate everything else.

| Date | Model(s) | Stimuli | What was run | Outcome (one line) |
|---|---|---|---|---|
| 2026-08-05 | — | ecog_v0 (39), mrs_v0 (18), recist_v0 (15) | `--dry-run`, `--check-tokens`, offline readout-safety checks | No model. ECOG/mRS separable bare-only; RECIST separable both conventions. No vocabulary leaks in ECOG. |
| 2026-08-05 | gemma-3-4b-it, medgemma-4b-it | Round-1 eligibility graphs | `reproduce_round1.py` (re-verification, offline) | Reproduced: cross-model jaccard 0.13–0.19, `age_pos` missing (0-byte graph). |
| 2026-08-06 | medgemma-4b-it, gemma-3-4b-it | ecog_v0 (39), mrs_v0 (18) | `run_ecog_stimuli.py`, RTX 5090 pod — **crashed** | NameError in `build_results` (`GRADE_QUESTION`, stale after 6ec0484). Scored and printed, wrote no JSON. Same deterministic computation as the row below, not an independent look. |
| 2026-08-06 | medgemma-4b-it, gemma-3-4b-it | ecog_v0 (39) | `run_ecog_stimuli.py`, RTX 5090 pod, both models same session | **Gemma grading falls monotonically with lexical distance: 100% verbatim → 82% paraphrase → 64% inferred → 50% distractor. MedGemma flat at ~60%.** Eligibility 22/39 and 25/39. Self-consistency 24/39 and 29/39 — the pillar-2 null is high. Readout clean, `probe_canonical_mass` ≥ 0.999, both surfaces calibrated `bare`. Grades not yet clinician-adjudicated. |
| 2026-08-06 | medgemma-4b-it, gemma-3-4b-it | mrs_v0 (18) | `run_ecog_stimuli.py`, same session | Smoke test of the per-intermediate config on real weights — it works. Eligibility 13/18 and 15/18. **Not a measurement**: 18 rows, no paraphrase sets, 9 rows name the scale. |
