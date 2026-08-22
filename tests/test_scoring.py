"""Locks the derived per-row verdicts (run_ecog_stimuli.score_rows).

Every pillar-2 number the programme quotes is a count over these four fields.
`dissociation` in particular is ADR-0007's quantity: the informative cell is a
right answer over a wrong grade, and mislabelling it silently changes the claim.

`self_consistent` asks whether the answer agrees with the model's OWN grade, not
with the truth — it is meaningful even when both turns are wrong, which is the
whole point of measuring coupling separately from accuracy.
"""
from __future__ import annotations

import pytest

from run_ecog_stimuli import score_rows


@pytest.mark.parametrize("says,expected_eligible,pred,expected_grade,cell", [
    ("Yes", "Yes", 1, 1, "both_right"),
    ("No",  "Yes", 1, 1, "wrong_answer_right_grade"),
    ("Yes", "Yes", 3, 1, "right_answer_wrong_grade"),
    ("No",  "Yes", 3, 1, "both_wrong"),
])
def test_dissociation_cells(make_row, says, expected_eligible, pred, expected_grade, cell):
    row = make_row(says=says, expected_eligible=expected_eligible,
                   pred_grade=pred, expected_grade=expected_grade)
    score_rows([row])
    assert row["dissociation"] == cell
    assert row["eligibility_correct"] == (says == expected_eligible)
    assert row["grade_correct"] == (pred == expected_grade)


def test_self_consistency_tracks_the_models_own_grade(make_row):
    """Both turns wrong, but coupled — self_consistent must still be True."""
    row = make_row(says="No", implied="No", expected_eligible="Yes",
                   pred_grade=4, expected_grade=0)
    score_rows([row])
    assert row["self_consistent"] is True
    assert row["dissociation"] == "both_wrong"


def test_self_consistency_is_false_when_the_answer_leaves_the_grade(make_row):
    row = make_row(says="Yes", implied="No")
    score_rows([row])
    assert row["self_consistent"] is False


def test_indeterminate_answer_is_not_scored(make_row):
    """Pillar 3's `indeterminate` rows have no answer key; they must not count."""
    row = make_row(expected_eligible=None)
    score_rows([row])
    assert row["eligibility_correct"] is None
    assert row["dissociation"] == "incomplete"


def test_missing_true_grade_is_not_scored(make_row):
    row = make_row(expected_grade=None)
    score_rows([row])
    assert row["grade_correct"] is None
    assert row["dissociation"] == "incomplete"


def test_scoring_is_idempotent(make_row):
    row = make_row()
    score_rows([row])
    first = dict(row)
    score_rows([row])
    assert row == first
