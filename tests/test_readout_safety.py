"""Locks the offline readout checks (surface_gate, separability).

These decide whether a readout means anything before any weights load. The
failure they exist to prevent is `data/ecog_v0_results_READOUT_BUG.md`: a token
id resolves cleanly, calibrates fine, and measures the wrong thing — completeness
will not catch it later, and the numbers look ordinary.

`separability` takes `encode(text) -> list[int]` rather than a tokenizer
specifically so it can be exercised without transformers. Nothing used that seam
until this file.
"""
from __future__ import annotations

import pytest

from run_ecog_stimuli import IMPLEMENTED_SURFACES, READOUTS, separability, surface_gate


def encoder(table: dict[str, list[int]]):
    """Deterministic fake tokenizer. Unlisted strings encode to nothing."""
    return lambda text: table.get(text, [])


# --------------------------------------------------------------------------- #
# surface_gate
# --------------------------------------------------------------------------- #
def test_implemented_surface_passes():
    assert surface_gate("ecog", {"answer_surface": "single_token_per_value"}) is None


@pytest.mark.parametrize("decl", [
    {"answer_surface": "free_text"},
    {"answer_surface": None},
    {},                                    # missing key must not pass silently
])
def test_unimplemented_surface_is_refused(decl):
    msg = surface_gate("thing", decl)
    assert msg is not None
    assert "not implemented" in msg and "thing" in msg


def test_every_declared_intermediate_passes_the_gate():
    for name, decl in READOUTS.items():
        assert surface_gate(name, decl) is None, name
    assert IMPLEMENTED_SURFACES == {"single_token_per_value"}


# --------------------------------------------------------------------------- #
# separability
# --------------------------------------------------------------------------- #
def test_distinct_first_tokens_are_separable():
    enc = encoder({"0": [10], "1": [11], " 0": [20], " 1": [21]})
    rep = separability(enc, ["0", "1"], {"0": ["0"], "1": ["1"]})
    assert rep["separable_under"] == ["bare", "spaced"]
    assert rep["conventions"]["bare"]["collisions"] == {}


def test_collision_under_one_convention_only():
    """The real ECOG/mRS finding: separable bare, not spaced."""
    enc = encoder({"0": [10], "1": [11], " 0": [99], " 1": [99]})
    rep = separability(enc, ["0", "1"], {"0": ["0"], "1": ["1"]})
    assert rep["separable_under"] == ["bare"]
    assert rep["conventions"]["spaced"]["separable"] is False
    assert rep["conventions"]["spaced"]["collisions"] == {99: ["0", "1"]}


def test_value_that_does_not_encode_is_unresolved():
    enc = encoder({"0": [10], " 0": [20]})          # "1" missing entirely
    rep = separability(enc, ["0", "1"], {"0": ["0"], "1": ["1"]})
    assert rep["conventions"]["bare"]["unresolved"] == ["1"]
    assert rep["conventions"]["bare"]["separable"] is False


def test_multitoken_form_is_fine_when_its_first_token_is_unique():
    """"Complete Response" is a legal target: "Complete" is unique among CR/PR."""
    enc = encoder({"Complete Response": [50, 51], "Partial Response": [60, 51],
                   " Complete Response": [150, 51], " Partial Response": [160, 51]})
    rep = separability(enc, ["CR", "PR"],
                       {"CR": ["Complete Response"], "PR": ["Partial Response"]})
    assert rep["conventions"]["bare"]["separable"] is True
    assert rep["values"]["CR"][0]["n_tokens"] == 2


def test_shared_first_token_across_abbreviations_collides():
    """If "CR" and "PR" both start with the same id, the readout cannot tell them apart."""
    enc = encoder({"CR": [7, 8], "PR": [7, 9], " CR": [17, 8], " PR": [17, 9]})
    rep = separability(enc, ["CR", "PR"], {"CR": ["CR"], "PR": ["PR"]})
    assert rep["separable_under"] == []
    assert rep["conventions"]["bare"]["collisions"] == {7: ["CR", "PR"]}


def test_one_bad_form_poisons_the_value():
    """Every admissible form is checked — a colliding alias is not excused by a good one."""
    enc = encoder({"CR": [1], "Complete Response": [5, 6],
                   "PR": [2], "Partial Response": [5, 7]})
    rep = separability(enc, ["CR", "PR"],
                       {"CR": ["CR", "Complete Response"],
                        "PR": ["PR", "Partial Response"]})
    assert rep["conventions"]["bare"]["collisions"] == {5: ["CR", "PR"]}


def test_missing_surface_forms_falls_back_to_the_value():
    enc = encoder({"0": [10], "1": [11]})
    rep = separability(enc, ["0", "1"], {})
    assert rep["conventions"]["bare"]["separable"] is True


def test_real_ecog_declaration_separates_with_a_digit_tokenizer():
    """Digits as their own tokens: the assumption the grading prompt relies on."""
    decl = READOUTS["ecog"]
    enc = encoder({v: [100 + int(v)] for v in decl["values"]}
                  | {f" {v}": [200 + int(v)] for v in decl["values"]})
    rep = separability(enc, decl["values"], decl["surface_forms"])
    assert rep["separable_under"] == ["bare", "spaced"]
