"""Locks the end-of-run assembly: summarise, build_results, write_scored_csv.

These run only after every prompt has been scored, which is exactly why they
need a test. On 2026-08-06 a stale `GRADE_QUESTION` reference in build_results
raised NameError on a live pod run: both models had already been loaded and
scored, the results were printed, and no JSON was written. The run-log records
it as a crash that produced nothing. Calling these on synthetic rows costs
milliseconds and would have caught it before the pod was booked.

No model, no tokenizer — just the shapes.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from run_ecog_stimuli import (INTERMEDIATES, build_results, score_rows, summarise,
                              write_scored_csv)

CFG = INTERMEDIATES["ecog"]


@pytest.fixture
def tok():
    grades = CFG["values"]
    return {
        "yes": 1, "no": 2,
        "grade": {g: 100 + g for g in grades},
        "yes_form": " Yes", "no_form": " No",
        "grade_form": {g: f" {g}" for g in grades},
        "yes_agg": [1, 3], "no_agg": [2, 4],
        "grade_agg": {g: [100 + g, 200 + g] for g in grades},
    }


def test_build_results_assembles(tok, make_row):
    """The NameError regression: this must return a dict, not raise."""
    rows = [make_row("R1"), make_row("R2", says="No", expected_eligible="No")]
    score_rows(rows)
    summary = summarise(rows)
    out = build_results(argparse.Namespace(model="test/model"), Path("stim.csv"),
                        rows, tok, ["a warning"], summary, {"yesno": "bare"}, CFG)

    assert out["model"] == "test/model"
    assert out["n_rows"] == 2
    assert out["intermediate"] == CFG["name"]
    assert out["warnings"] == ["a warning"]
    assert out["rows"] is rows
    # the readout provenance a later reader needs to trust the numbers
    assert out["token_ids"]["canonical"]["yes"] == 1
    assert out["token_ids"]["canonical"]["surface_forms"]["yes"] == " Yes"
    assert set(out["token_ids"]["canonical"]["grades"]) == {"0", "1", "2", "3", "4"}
    assert "eligibility" in out["prompts"] and "grading" in out["prompts"]


def test_build_results_is_json_serialisable(tok, make_row):
    """It is written straight to disk; a stray non-serialisable value loses the run."""
    import json
    rows = [make_row()]
    score_rows(rows)
    out = build_results(argparse.Namespace(model="m"), Path("s.csv"), rows, tok,
                        [], summarise(rows), {}, CFG)
    json.dumps(out)


def test_summarise_counts(make_row):
    rows = [
        make_row("A", says="Yes", expected_eligible="Yes", pred_grade=1, expected_grade=1),
        make_row("B", says="No", expected_eligible="Yes", pred_grade=1, expected_grade=1),
        make_row("C", says="Yes", expected_eligible="Yes", pred_grade=3, expected_grade=1),
        make_row("D", says="Yes", expected_eligible="Yes", pred_grade=2,
                 expected_grade=2, ambiguous=True),
    ]
    score_rows(rows)
    s = summarise(rows)
    assert s["eligibility_correct"] == 3 and s["eligibility_n"] == 4
    # ambiguous rows are held out of the headline grade count and reported apart
    assert s["grade_correct"] == 2 and s["grade_n"] == 3
    assert s["grade_ambiguous_correct"] == 1 and s["grade_ambiguous_n"] == 1
    # dissociation spans ALL rows, ambiguous included — unlike grade_n above.
    # A and D are both_right; D is the ambiguous one, held out of grade_n but
    # still counted here. Worth knowing before quoting the two side by side.
    assert s["dissociation"]["both_right"] == 2
    assert s["dissociation"]["wrong_answer_right_grade"] == 1
    assert s["dissociation"]["right_answer_wrong_grade"] == 1
    assert sum(s["dissociation"].values()) == 4


def test_summarise_paraphrase_set_drift(make_row):
    rows = [make_row("A", set_id="S1", pred_grade=1, lexical_distance="paraphrase"),
            make_row("B", set_id="S1", pred_grade=3, lexical_distance="inferred_symptoms")]
    score_rows(rows)
    sets = summarise(rows)["paraphrase_sets"]
    assert sets["S1"]["grade_consistent"] is False
    assert sets["S1"]["grade_drift"] == 2


def test_summarise_ignores_singleton_sets(make_row):
    rows = [make_row("A", set_id="S1")]
    score_rows(rows)
    assert summarise(rows)["paraphrase_sets"] == {}


def test_write_scored_csv_fills_outputs_and_keeps_handwritten_columns(tmp_path, make_row):
    stim = tmp_path / "stim.csv"
    with stim.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "ecog_true", "vignette_text", "notes"])
        w.writeheader()
        w.writerow({"id": "R1", "ecog_true": "1", "vignette_text": "v", "notes": "keep me"})

    rows = [make_row("R1", says="No", pred_grade=3)]
    score_rows(rows)
    out = tmp_path / "scored.csv"
    write_scored_csv(stim, rows, out)

    got = list(csv.DictReader(out.open()))
    assert len(got) == 1
    assert got[0]["model_inclusion"] == "No"
    assert got[0]["model_ecog"] == "3"
    assert got[0]["notes"] == "keep me"        # hand-written columns pass through
    assert got[0]["ecog_true"] == "1"


def test_write_scored_csv_leaves_unmatched_rows_blank(tmp_path, make_row):
    stim = tmp_path / "stim.csv"
    with stim.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "vignette_text"])
        w.writeheader()
        w.writerow({"id": "OTHER", "vignette_text": "v"})

    rows = [make_row("R1")]
    score_rows(rows)
    out = tmp_path / "scored.csv"
    write_scored_csv(stim, rows, out)
    got = list(csv.DictReader(out.open()))
    assert got[0].get("model_inclusion") in ("", None)
