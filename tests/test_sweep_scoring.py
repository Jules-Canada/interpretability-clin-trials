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

from flip_point import collect, flip_point, own_grade, score as score_run, surv

GRADES = [0] * 6 + [1] * 12 + [2] * 12 + [3] * 5 + [4] * 4      # ecog_v0's distribution
THRESHOLDS = [0, 1, 2, 3, 4]


def sweep_rows(answer_fn, grades=GRADES, ambiguous=False, grade_fn=lambda g: g):
    """`grade_fn` is the model's grade; by default it agrees with the key.

    A model whose grades differ from the key is the case the own-grade baseline
    exists for, so the archetypes below need to be able to build one.
    """
    return [{"id": f"V{i:03d}_LE{t}", "criterion_text": f"ECOG <= {t} required",
             "expected_grade": g, "ambiguous": ambiguous,
             "eligibility": {"says": answer_fn(g, t)},
             "grading": {"pred_grade": grade_fn(g)}}
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


# --------------------------------------------------------------------------- #
# the two baselines — flip==true and flip==own are not held to the same bar
# --------------------------------------------------------------------------- #
# always-Yes flips at t=0, so it hits wherever the *target* grade is 0. That
# target is the key's grade for flip==true but the model's own grade for
# flip==own, and a lenient grader moves only the second. Scoring coupling
# against the true-grade baseline credits grading leniency as application.
def baselines(answer_fn, grade_fn=lambda g: g):
    s = score_run(collect(sweep_rows(answer_fn, grade_fn=grade_fn)))
    return s["base_true"], s["base_own"]


def test_baselines_agree_when_the_model_grades_like_the_key():
    assert baselines(lambda g, t: "Yes" if g <= t else "No") == (6, 6)


def test_a_lenient_grader_raises_only_its_own_grade_baseline():
    """Calling every grade-1 vignette a 0 hands always-Yes 12 more free hits."""
    bt, bo = baselines(lambda g, t: "Yes", grade_fn=lambda g: 0 if g <= 1 else g)
    assert bt == 6                      # the key is untouched
    assert bo == 18                     # the 6 real zeros plus 12 downgraded ones


def test_a_strict_grader_lowers_its_own_grade_baseline():
    bt, bo = baselines(lambda g, t: "Yes", grade_fn=lambda g: max(g, 1))
    assert bt == 6
    assert bo == 0                      # it never says 0, so always-Yes never hits


def test_grade_zero_collapse_scores_perfectly_and_means_nothing():
    """Grade everything 0, answer Yes to everything: 39/39 coupling, no computation.

    The whole reason the own-grade baseline is computed separately. Against the
    true-grade baseline this reads as overwhelming significance.
    """
    s = score_run(collect(sweep_rows(lambda g, t: "Yes", grade_fn=lambda g: 0)))
    assert s["hit_own"] == 39                       # looks perfect
    assert s["base_own"] == 39                      # so does the constant responder
    assert surv(s["hit_own"], s["m"], s["base_own"] / s["m"]) > 0.05
    assert surv(s["hit_own"], s["m"], s["base_true"] / s["m"]) < 1e-6


def test_own_grade_is_none_when_the_run_is_not_deterministic():
    """No single own-grade means the vignette cannot count for or against coupling."""
    rows = sweep_rows(lambda g, t: "Yes")
    rows[0]["grading"]["pred_grade"] = 4            # one copy disagrees
    e = collect(rows)["V000"]
    assert own_grade(e) is None
