"""Locks the per-row threshold rule (scripts/run_ecog_stimuli.eligible_rule).

The regression this exists to prevent, found 2026-08-21: the threshold came from
the intermediate's config and was applied to every row, ignoring that row's own
`criterion_text`. ecog_v0 and mrs_v0 each carry a reversed-threshold distractor
written to catch a model that pattern-matches "ECOG <= 1" instead of reading the
criterion — and the scorer was committing that identical error, so those rows
scored self_consistent whatever the model answered. The one check on threshold
direction could not fail.

Offline; no tokenizer, no weights.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from run_ecog_stimuli import INTERMEDIATES, eligible_rule

ROOT = Path(__file__).resolve().parent.parent
ECOG = INTERMEDIATES["ecog"]
MRS = INTERMEDIATES["mrs"]


@pytest.mark.parametrize("criterion,cfg,eligible,excluded", [
    ("ECOG <= 1 required for enrollment", ECOG, [0, 1], [2, 3, 4]),
    ("ECOG <= 0 required for enrollment", ECOG, [0], [1, 2, 3, 4]),
    ("ECOG >= 2 required for enrollment (supportive care cohort)", ECOG,
     [2, 3, 4], [0, 1]),
    ("mRS <= 2 required for enrollment", MRS, [0, 1, 2], [3, 4, 5, 6]),
    ("mRS >= 3 required for enrollment (palliative/severe-disability cohort)", MRS,
     [3, 4, 5, 6], [0, 1, 2]),
])
def test_threshold_read_from_the_row(criterion, cfg, eligible, excluded):
    rule, source = eligible_rule({"criterion_text": criterion}, cfg)
    assert source == "criterion"
    assert [g for g in cfg["values"] if rule(g)] == eligible
    assert [g for g in cfg["values"] if not rule(g)] == excluded


def test_reversed_criterion_is_not_scored_as_the_default():
    """The bug in one assertion: grade 1 under `>= 2` is excluded, not eligible."""
    rule, _ = eligible_rule(
        {"criterion_text": "ECOG >= 2 required for enrollment (supportive care cohort)"},
        ECOG)
    assert rule(1) is False
    default = ECOG["eligible"]
    assert default(1) is True          # the config default disagrees — that was the bug


@pytest.mark.parametrize("criterion", [
    "no progressive disease required for continuation",
    "no progressive disease required for continuation (protocol specifies RECIST 1.1)",
    "",
])
def test_prose_criterion_falls_back_to_the_config(criterion):
    rule, source = eligible_rule({"criterion_text": criterion}, ECOG)
    assert source == "config"
    assert [g for g in ECOG["values"] if rule(g)] == [0, 1]


def test_missing_criterion_key_does_not_raise():
    rule, source = eligible_rule({}, ECOG)
    assert source == "config" and rule(0) is True


@pytest.mark.parametrize("stem,cfg", [("ecog_v0", ECOG), ("mrs_v0", MRS)])
def test_every_criterion_in_the_corpus_parses(stem, cfg):
    """A criterion silently falling back to the config is how the bug hid."""
    path = ROOT / "specs" / "stimuli" / f"{stem}.csv"
    criteria = {r["eligibility_criterion"] for r in csv.DictReader(path.open())
                if r.get("vignette_text")}
    assert criteria, f"no criteria read from {stem}"
    for c in criteria:
        _, source = eligible_rule({"criterion_text": c}, cfg)
        assert source == "criterion", f"{c!r} fell back to the config default"
