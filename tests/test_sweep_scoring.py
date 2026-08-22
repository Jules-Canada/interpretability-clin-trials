"""Locks flip-point scoring (scripts/flip_point.py).

The sweep's whole claim rests on one property: a constant responder cannot score.
always-No never flips and gets 0; always-Yes flips at t=0 and collects exactly the
true-grade-0 vignettes, because the scale floors there and an early flip is
indistinguishable from a correct one. That is why the baseline is 6/39 and why
the score is meaningless read on its own.

Mirrors flip_point.py --selftest so the same cases run with the rest of the suite.
"""
from __future__ import annotations

import pytest

from flip_point import collect, flip_point

GRADES = [0] * 6 + [1] * 12 + [2] * 12 + [3] * 5 + [4] * 4      # ecog_v0's distribution
THRESHOLDS = [0, 1, 2, 3, 4]


def sweep_rows(answer_fn, grades=GRADES, ambiguous=False):
    return [{"id": f"V{i:03d}_LE{t}", "criterion_text": f"ECOG <= {t} required",
             "expected_grade": g, "ambiguous": ambiguous,
             "eligibility": {"says": answer_fn(g, t)},
             "grading": {"pred_grade": g}}
            for i, g in enumerate(grades) for t in THRESHOLDS]


def score(answer_fn):
    """(flip == true grade, flip == own grade) over the synthetic corpus."""
    hit_true = hit_own = 0
    for e in collect(sweep_rows(answer_fn)).values():
        fp, mono = flip_point(e["le"])
        if mono and fp is not None:
            hit_true += fp == e["true"]
            hit_own += fp == next(iter(e["grades"]))
    return hit_true, hit_own


# --------------------------------------------------------------------------- #
# flip_point
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("answers,want", [
    ({0: "No", 1: "Yes", 2: "Yes", 3: "Yes", 4: "Yes"}, (1, True)),
    ({0: "Yes", 1: "Yes", 2: "Yes", 3: "Yes", 4: "Yes"}, (0, True)),
    ({0: "No", 1: "No", 2: "No", 3: "No", 4: "Yes"}, (4, True)),
    ({0: "No", 1: "No", 2: "No", 3: "No", 4: "No"}, (None, True)),
])
def test_step_functions_resolve(answers, want):
    assert flip_point(answers) == want


def test_non_monotone_is_flagged_not_scored():
    """Flipping back and forth is not threshold-reading; the flip point is not meaningful."""
    fp, mono = flip_point({0: "Yes", 1: "No", 2: "Yes", 3: "Yes", 4: "Yes"})
    assert fp == 0 and mono is False


def test_a_late_relapse_is_also_non_monotone():
    fp, mono = flip_point({0: "No", 1: "Yes", 2: "Yes", 3: "No", 4: "Yes"})
    assert fp == 1 and mono is False


# --------------------------------------------------------------------------- #
# responder archetypes — the baseline argument
# --------------------------------------------------------------------------- #
def test_perfect_responder_scores_everything():
    assert score(lambda g, t: "Yes" if g <= t else "No") == (39, 39)


def test_always_no_scores_zero():
    """The baseline the 2026-08-06 between-item design could not beat."""
    assert score(lambda g, t: "No") == (0, 0)


def test_always_yes_collects_only_the_floored_grade_zero_rows():
    assert score(lambda g, t: "Yes") == (6, 6)
    assert sum(1 for g in GRADES if g == 0) == 6


def test_uniformly_lenient_responder_gets_the_same_six_for_free():
    """Off by one everywhere, yet correct at grade 0 — the scale floors there."""
    assert score(lambda g, t: "Yes" if g <= t + 1 else "No") == (6, 6)


def test_uniformly_strict_responder_scores_nothing_extra():
    assert score(lambda g, t: "Yes" if g <= t - 1 else "No") == (0, 0)


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def test_collect_groups_by_source_vignette():
    by = collect(sweep_rows(lambda g, t: "Yes" if g <= t else "No"))
    assert len(by) == len(GRADES)
    entry = by["V000"]
    assert set(entry["le"]) == set(THRESHOLDS)
    assert entry["true"] == 0

def test_collect_separates_the_reversed_direction_rows():
    rows = [{"id": "V1_LE1", "criterion_text": "ECOG <= 1 required", "expected_grade": 1,
             "ambiguous": False, "eligibility": {"says": "Yes"},
             "grading": {"pred_grade": 1}},
            {"id": "V1_GE2", "criterion_text": "ECOG >= 2 required", "expected_grade": 1,
             "ambiguous": False, "eligibility": {"says": "No"},
             "grading": {"pred_grade": 1}}]
    entry = collect(rows)["V1"]
    assert entry["le"] == {1: "Yes"} and entry["ge"] == {2: "No"}


def test_collect_records_grade_disagreement_across_identical_prompts():
    """The determinism check: the grading prompt never mentions the criterion."""
    rows = [{"id": "V1_LE1", "criterion_text": "ECOG <= 1 required", "expected_grade": 1,
             "ambiguous": False, "eligibility": {"says": "Yes"},
             "grading": {"pred_grade": 1}},
            {"id": "V1_LE2", "criterion_text": "ECOG <= 2 required", "expected_grade": 1,
             "ambiguous": False, "eligibility": {"says": "Yes"},
             "grading": {"pred_grade": 3}}]
    assert collect(rows)["V1"]["grades"] == {1, 3}


def test_collect_skips_rows_without_a_parseable_criterion():
    rows = [{"id": "V1_X", "criterion_text": "no progressive disease required",
             "expected_grade": 1, "ambiguous": False,
             "eligibility": {"says": "Yes"}, "grading": {"pred_grade": 1}}]
    assert collect(rows) == {}
