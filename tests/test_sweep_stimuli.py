"""Locks the sweep set's derived answer key (specs/stimuli/ecog_sweep_v0.csv).

Unlike the hand-written sets, this file's `expected_answer` is computed — 234
rows of ground truth produced by scripts/make_sweep_stimuli.py from each
vignette's true grade and each criterion. Nothing else checks it, and an error
here does not look like an error: it looks like a model result.

Also checks the committed file still matches the generator, so the two cannot
drift without someone noticing.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "specs" / "stimuli" / "ecog_sweep_v0.csv"
SOURCE = ROOT / "specs" / "stimuli" / "ecog_v0.csv"


@pytest.fixture(scope="module")
def rows():
    return list(csv.DictReader(SWEEP.open(encoding="utf-8-sig")))


@pytest.fixture(scope="module")
def source_rows():
    return {r["id"]: r for r in csv.DictReader(SOURCE.open(encoding="utf-8-sig"))
            if r.get("vignette_text")}


def test_shape(rows, source_rows):
    assert len(source_rows) == 39
    assert len(rows) == 234                       # 39 vignettes x 6 criteria
    assert len({r["id"] for r in rows}) == 234    # ids unique


def test_answer_key_follows_the_criterion(rows):
    """The whole point of the file: the label must track the swept threshold."""
    for r in rows:
        grade = int(r["ecog_true"])
        op, val = r["criterion_op"], int(r["criterion_value"])
        eligible = grade <= val if op == "<=" else grade >= val
        want = "eligible" if eligible else "excluded"
        assert r["expected_answer"] == want, (
            f"{r['id']}: grade {grade} under {op} {val} should be {want}")


def test_criterion_text_matches_its_columns(rows):
    """The parser reads criterion_text; the columns are for analysis. Keep them agreed."""
    for r in rows:
        assert f"ECOG {r['criterion_op']} {r['criterion_value']} " in r["eligibility_criterion"]


def test_boundary_case_is_the_pairing_not_the_vignette(rows):
    for r in rows:
        want = int(r["ecog_true"]) == int(r["criterion_value"])
        assert (r["boundary_case"] == "True") is want, r["id"]


def test_vignette_text_is_carried_through_unchanged(rows, source_rows):
    """Vignettes vary only the clinical description; the sweep must not touch it."""
    for r in rows:
        src = source_rows[r["source_id"]]
        assert r["vignette_text"] == src["vignette_text"]
        assert r["ecog_true"] == src["ecog_true"]
        assert r["lexical_distance"] == src["lexical_distance"]


def test_every_vignette_flips_inside_the_sweep(rows, source_rows):
    """A vignette with no flip point contributes nothing to the primary metric."""
    forward = [r for r in rows if r["criterion_op"] == "<="]
    by_src: dict[str, list] = {}
    for r in forward:
        by_src.setdefault(r["source_id"], []).append(r)
    assert len(by_src) == 39
    for src, group in by_src.items():
        answers = {int(r["criterion_value"]): r["expected_answer"] for r in group}
        assert "eligible" in answers.values(), f"{src} is never eligible at any threshold"
        flip = min(t for t, a in answers.items() if a == "eligible")
        assert flip == int(group[0]["ecog_true"]), f"{src} flips at {flip}"


def test_reversed_direction_row_is_present(rows):
    """The check that separates reading the criterion from pattern-matching `<= 1`."""
    ge = [r for r in rows if r["criterion_op"] == ">="]
    assert len(ge) == 39
    assert all(r["distractor_type"] == "reversed_threshold" for r in ge)


def test_tiers_are_declared(rows):
    tiers = {r["tier"] for r in rows}
    assert tiers == {"primary", "exploratory", "ceiling", "direction"}
    primary = {int(r["criterion_value"]) for r in rows if r["tier"] == "primary"}
    assert primary == {1, 2}, "the pre-registered contrast is <=1 vs <=2"


def test_committed_file_matches_the_generator(tmp_path):
    """Regenerate and diff: the CSV and its generator must not drift apart."""
    out = tmp_path / "regen.csv"
    subprocess.run([sys.executable, "scripts/make_sweep_stimuli.py", "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    assert out.read_text(encoding="utf-8") == SWEEP.read_text(encoding="utf-8")
