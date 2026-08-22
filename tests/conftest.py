"""Shared path setup and row factory.

The scripts are imported by path, not installed: `pip install -e .` would pull
the CLT-era dependency list out of pyproject.toml (transformer_lens, h5py, wandb,
datasets) that the active path deliberately does not use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "scripts", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def make_row():
    """Build a scored-shaped stimulus row. Override any field per call."""
    def _make(rid="R001", *, expected_grade=1, pred_grade=1,
              says="Yes", implied="Yes", expected_eligible="Yes",
              ambiguous=False, set_id=None, lexical_distance="paraphrase",
              decisive=False, criterion_text="ECOG <= 1 required for enrollment",
              logit_diff=1.0):
        return {
            "id": rid,
            "set_id": set_id,
            "criterion_text": criterion_text,
            "patient": "a patient",
            "expected_grade": expected_grade,
            "expected_eligible": expected_eligible,
            "expected_answer": {"Yes": "eligible", "No": "excluded"}.get(expected_eligible),
            "lexical_distance": lexical_distance,
            "boundary_case": False,
            "decisive": decisive,
            "ambiguous": ambiguous,
            "distractor_type": "none",
            "notes": "",
            "leaks_vocab": False,
            "intermediate": "ecog",
            "eligibility": {"says": says, "logit_diff": logit_diff},
            "grading": {"pred_grade": pred_grade, "pred_eligible_from_grade": implied},
        }
    return _make
