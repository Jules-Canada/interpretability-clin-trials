"""Locks the stimulus reader (run_ecog_stimuli.load_stimuli).

Two things here rot quietly. The pre-2026-08-05 header set (`patient_detail`,
`inclusion_rule`, `expected_ecog`, `expected_inclusion`, `decisive`) is still
tolerated so older copies and the scored CSVs written from them keep loading,
and nothing in the live path exercises it. And the derived fields — leaks_vocab,
decisive, expected_eligible — are computed here, so a change reaches every
downstream number without touching the code that reports them.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from run_ecog_stimuli import load_stimuli


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


CURRENT = ["id", "ecog_true", "set_id", "lexical_distance", "boundary_case",
           "ambiguous", "vignette_text", "eligibility_criterion", "expected_answer",
           "distractor_type", "notes"]


def base_row(**kw):
    row = {"id": "E001", "ecog_true": "1", "set_id": "", "lexical_distance": "paraphrase",
           "boundary_case": "False", "ambiguous": "False", "vignette_text": "walks a lot",
           "eligibility_criterion": "ECOG <= 1 required for enrollment",
           "expected_answer": "eligible", "distractor_type": "none", "notes": ""}
    row.update(kw)
    return row


def test_current_schema(tmp_path):
    p = write_csv(tmp_path / "s.csv", CURRENT, [base_row()])
    (row,) = load_stimuli(p)
    assert row["id"] == "E001"
    assert row["expected_grade"] == 1
    assert row["expected_eligible"] == "Yes"
    assert row["expected_answer"] == "eligible"
    assert row["intermediate"] == "ecog"
    assert row["criterion_text"] == "ECOG <= 1 required for enrollment"


def test_excluded_maps_to_no_and_is_decisive(tmp_path):
    """`decisive` is no longer stored — it is derived as 'the grade settles it'."""
    p = write_csv(tmp_path / "s.csv", CURRENT,
                  [base_row(expected_answer="excluded", ecog_true="3")])
    (row,) = load_stimuli(p)
    assert row["expected_eligible"] == "No"
    assert row["decisive"] is True


def test_eligible_is_not_decisive(tmp_path):
    p = write_csv(tmp_path / "s.csv", CURRENT, [base_row(expected_answer="eligible")])
    (row,) = load_stimuli(p)
    assert row["decisive"] is False


def test_indeterminate_has_no_answer_key(tmp_path):
    """Pillar 3's row: present in the file, deliberately unscoreable."""
    p = write_csv(tmp_path / "s.csv", CURRENT,
                  [base_row(expected_answer="indeterminate")])
    (row,) = load_stimuli(p)
    assert row["expected_eligible"] is None
    assert row["expected_answer"] == "indeterminate"


@pytest.mark.parametrize("text,leaks", [
    ("walks a lot, tires easily", False),
    ("ECOG 1 by the notes", True),
    ("performance status is fair", True),
    ("Karnofsky 80", True),
])
def test_leaks_vocab_flags_the_defining_words(tmp_path, text, leaks):
    """A vignette naming the scale tests lookup, not recovery."""
    p = write_csv(tmp_path / "s.csv", CURRENT, [base_row(vignette_text=text)])
    (row,) = load_stimuli(p)
    assert row["leaks_vocab"] is leaks


def test_mrs_uses_its_own_vocabulary(tmp_path):
    fields = ["id", "mrs_true", "lexical_distance", "vignette_text",
              "eligibility_criterion", "expected_answer"]
    p = write_csv(tmp_path / "m.csv", fields, [
        {"id": "M1", "mrs_true": "2", "lexical_distance": "paraphrase",
         "vignette_text": "modified Rankin of 2", "eligibility_criterion": "mRS <= 2",
         "expected_answer": "eligible"}])
    (row,) = load_stimuli(p)
    assert row["intermediate"] == "mrs"
    assert row["leaks_vocab"] is True


def test_blank_rows_are_skipped(tmp_path):
    p = write_csv(tmp_path / "s.csv", CURRENT,
                  [base_row(), base_row(id="E002", vignette_text="")])
    assert [r["id"] for r in load_stimuli(p)] == ["E001"]


def test_legacy_headers_still_load(tmp_path):
    """The pre-2026-08-05 schema. Nothing in the live path covers this."""
    fields = ["id", "expected_ecog", "patient_detail", "inclusion_rule",
              "expected_inclusion", "decisive", "lexical_distance"]
    p = write_csv(tmp_path / "old.csv", fields, [
        {"id": "E001", "expected_ecog": "2", "patient_detail": "up and about",
         "inclusion_rule": "ECOG <= 1", "expected_inclusion": "no",
         "decisive": "true", "lexical_distance": "paraphrase"}])
    (row,) = load_stimuli(p)
    assert row["patient"] == "up and about"
    assert row["criterion_text"] == "ECOG <= 1"
    assert row["expected_grade"] == 2
    assert row["expected_eligible"] == "No"
    assert row["decisive"] is True


def test_headers_are_normalised(tmp_path):
    """Stray spaces and capitals in a hand-edited CSV must not silently drop a column."""
    fields = [" ID ", "ECOG_True", "Vignette Text", "Expected Answer"]
    p = write_csv(tmp_path / "s.csv", fields, [
        {" ID ": "E001", "ECOG_True": "0", "Vignette Text": "fully active",
         "Expected Answer": "eligible"}])
    (row,) = load_stimuli(p)
    assert row["id"] == "E001" and row["expected_grade"] == 0


def test_real_corpus_loads(tmp_path):
    root = Path(__file__).resolve().parent.parent / "specs" / "stimuli"
    for stem, n in (("ecog_v0", 39), ("mrs_v0", 18), ("recist_v0", 15)):
        rows = load_stimuli(root / f"{stem}.csv")
        assert len(rows) == n, f"{stem} changed size — update the test deliberately"
        assert all(r["patient"] for r in rows)
