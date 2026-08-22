"""Locks the patching harness (scripts/patch_grade.py).

Split deliberately. The module-tree walk and the tuple/tensor handling are pure
Python and run in the fast suite — they are what breaks when a transformers
release changes where the text stack lives or what a decoder layer returns.

The end-to-end invariants need a real forward pass, so they are marked `slow`:

    pytest -m slow tests/test_patching.py

That path builds a randomly initialised 3-layer Gemma 3 from config (no weights,
no network) and asserts patching is exact — identity, and last-layer dominance.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

import patch_grade
from patch_grade import _rejoin, _split, decoder_layers


# --------------------------------------------------------------------------- #
# module tree — where the decoder layers live
# --------------------------------------------------------------------------- #
def test_text_only_layout():
    layers = ["a", "b"]
    assert decoder_layers(NS(model=NS(layers=layers))) is layers


def test_multimodal_layout_nests_a_level_deeper():
    """`AutoModelForCausalLM` on a multimodal checkpoint — the Gemma 3 4B case."""
    layers = ["a"]
    assert decoder_layers(NS(model=NS(language_model=NS(layers=layers)))) is layers


def test_language_model_first_layout():
    layers = ["a"]
    assert decoder_layers(NS(language_model=NS(model=NS(layers=layers)))) is layers


def test_gpt2_style_layout():
    layers = ["a"]
    assert decoder_layers(NS(transformer=NS(h=layers))) is layers


def test_an_empty_list_is_not_accepted_as_the_stack():
    """A wrapper exposing an empty `.layers` must not shadow the real one."""
    real = ["a", "b"]
    model = NS(model=NS(layers=[], language_model=NS(layers=real)))
    assert decoder_layers(model) is real


def test_unknown_layout_raises_rather_than_patching_nothing():
    with pytest.raises(RuntimeError, match="could not locate decoder layers"):
        decoder_layers(NS(something_else=NS(layers=["a"])))


# --------------------------------------------------------------------------- #
# layer output shape — tensor on some versions, tuple on others
# --------------------------------------------------------------------------- #
def test_split_and_rejoin_a_bare_tensor():
    hidden, rest = _split("H")
    assert (hidden, rest) == ("H", None)
    assert _rejoin("H2", rest) == "H2"


def test_split_and_rejoin_a_tuple_preserves_the_tail():
    hidden, rest = _split(("H", "attn", "cache"))
    assert hidden == "H" and rest == ("attn", "cache")
    assert _rejoin("H2", rest) == ("H2", "attn", "cache")


def test_rejoin_keeps_a_single_element_tuple_a_tuple():
    hidden, rest = _split(("H",))
    assert _rejoin("H2", rest) == ("H2",)


# --------------------------------------------------------------------------- #
# end-to-end — needs a forward pass
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_patching_invariants_on_a_real_forward_pass():
    """identity, last-layer dominance, non-degeneracy, no leaked hooks."""
    patch_grade.selftest()          # exits non-zero on failure
