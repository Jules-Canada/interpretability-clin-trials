"""
tests/test_tokenizer_resolve.py — regression lock for scripts/_tokenizer_resolve.

Locks in the five-case precedence policy that decides which tokenizer decodes
HDF5 token ids back to context strings. Reintroducing the Phase 4 footgun (a
mismatched tokenizer silently decoding into garbage) must fail loudly here.

  HDF5 attr | --model_name | expected
  ----------+--------------+-----------------------------------
  present   | none         | return attr
  present   | == attr      | return attr
  present   | != attr      | SystemExit (wrong-value caught)
  absent    | given        | return flag (legacy fallback)
  absent    | none         | SystemExit (legacy, no truth)

No GPU/model needed — uses empty in-memory-sized HDF5 files, sub-second.
"""

import sys
from pathlib import Path

import h5py
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._tokenizer_resolve import resolve_model_name

MODEL = "google/medgemma-4b-pt"
OTHER = "EleutherAI/pythia-410m"


@pytest.fixture
def with_attr(tmp_path) -> str:
    """A self-describing HDF5 (written by current extract_activations.py)."""
    p = tmp_path / "withattr.h5"
    with h5py.File(p, "w") as f:
        f.attrs["model_name"] = MODEL
    return str(p)


@pytest.fixture
def legacy(tmp_path) -> str:
    """A pre-attr HDF5 (written before the self-describing change)."""
    p = tmp_path / "legacy.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("token_ids", data=[1, 2, 3])
    return str(p)


def test_attr_present_no_flag(with_attr):
    assert resolve_model_name(with_attr, None) == MODEL


def test_attr_present_flag_agrees(with_attr):
    assert resolve_model_name(with_attr, MODEL) == MODEL


def test_attr_present_flag_conflicts_is_hard_error(with_attr):
    with pytest.raises(SystemExit) as e:
        resolve_model_name(with_attr, OTHER)
    assert "conflicts" in str(e.value)


def test_legacy_with_flag_falls_back_to_flag(legacy):
    assert resolve_model_name(legacy, OTHER) == OTHER


def test_legacy_without_flag_is_hard_error(legacy):
    with pytest.raises(SystemExit) as e:
        resolve_model_name(legacy, None)
    assert "predates" in str(e.value)
