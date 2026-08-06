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
