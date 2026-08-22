"""Locks the statistics behind the pillar-2 baseline and the sizing rule.

These are small functions carrying disproportionate weight: `surv` decides
whether a result clears its baseline, and `required_n` sets how many vignettes a
confirmatory run needs — which spends the one resource CLAUDE.md names as the
bottleneck, clinician time. An error here is not a wrong number on a page, it is
a wrongly sized experiment.

`surv` and `wilson` exist in two copies (pillar2_baselines and flip_point). The
last test pins them together so they cannot drift.
"""
from __future__ import annotations

import math

import pytest

import flip_point as F
import pillar2_baselines as P


# --------------------------------------------------------------------------- #
# surv — P(X >= k) for X ~ Binom(n, p)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,n,p,want", [
    (1, 1, 0.5, 0.5),
    (2, 2, 0.5, 0.25),
    (1, 2, 0.5, 0.75),
    (0, 5, 0.3, 1.0),          # P(X >= 0) is certain
    (3, 3, 1.0, 1.0),
    (1, 3, 0.0, 0.0),
])
def test_surv_known_values(k, n, p, want):
    assert P.surv(k, n, p) == pytest.approx(want)


def test_surv_is_monotone_decreasing_in_k():
    vals = [P.surv(k, 20, 0.4) for k in range(21)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_surv_is_monotone_increasing_in_p():
    vals = [P.surv(10, 20, p / 20) for p in range(21)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_the_pilot_numbers_do_not_clear_their_baseline():
    """The 2026-08-21 finding, pinned. If this starts passing, something changed."""
    assert P.surv(23, 39, 20 / 39) == pytest.approx(0.212, abs=5e-3)   # gemma coupling
    assert P.surv(28, 39, 24 / 39) == pytest.approx(0.124, abs=5e-3)   # medgemma coupling
    assert P.surv(22, 39, 22 / 39) > 0.05                              # gemma eligibility


def test_flip_point_baseline_clears_at_eleven_of_thirty_nine():
    """The sweep's power claim: 11/39 beats the always-Yes baseline of 6/39."""
    assert P.surv(11, 39, 6 / 39) < 0.05
    assert P.surv(10, 39, 6 / 39) > 0.05


# --------------------------------------------------------------------------- #
# wilson
# --------------------------------------------------------------------------- #
def test_wilson_brackets_the_point_estimate():
    # tolerance because the interval touches the point estimate exactly at k=0
    # and k=n, where the arithmetic lands a float ulp shy of it
    for k in range(0, 21):
        lo, hi = P.wilson(k, 20)
        assert lo <= k / 20 + 1e-12 and k / 20 - 1e-12 <= hi


def test_wilson_stays_inside_zero_one():
    for k in range(0, 11):
        lo, hi = P.wilson(k, 10)
        assert 0.0 <= lo and hi <= 1.0


def test_wilson_is_symmetric_about_a_half():
    lo, hi = P.wilson(5, 10)
    assert lo + hi == pytest.approx(1.0)


def test_wilson_narrows_as_n_grows():
    widths = [(lambda t: t[1] - t[0])(P.wilson(n // 2, n)) for n in (20, 100, 500)]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_handles_zero_n():
    assert F.wilson(0, 0) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# required_n
# --------------------------------------------------------------------------- #
def test_required_n_matches_the_documented_sizing_rule():
    """docs/program/pillars-1-2-intermediate-recovery.md: 0.15 lift over 0.60 -> n=61."""
    assert math.ceil(P.required_n(0.60, 0.75)) == 61


def test_required_n_for_the_flip_point_design():
    """The sweep is powered by the 39 vignettes already written."""
    assert P.required_n(6 / 39, 0.50) < 39


def test_no_lift_needs_infinite_n():
    assert P.required_n(0.5, 0.5) == float("inf")
    assert P.required_n(0.5, 0.4) == float("inf")


def test_required_n_falls_as_the_lift_grows():
    ns = [P.required_n(0.6, 0.6 + d) for d in (0.05, 0.10, 0.20, 0.30)]
    assert all(a > b for a, b in zip(ns, ns[1:]))


# --------------------------------------------------------------------------- #
def test_the_two_copies_of_the_statistics_agree():
    """surv and wilson are duplicated across two scripts; keep them identical."""
    for k in range(21):
        assert P.surv(k, 20, 0.4) == pytest.approx(F.surv(k, 20, 0.4), abs=1e-15)
        assert P.wilson(k, 20) == pytest.approx(F.wilson(k, 20))
