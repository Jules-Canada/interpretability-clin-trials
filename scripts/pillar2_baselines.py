#!/usr/bin/env python3
"""Pillar-2 numbers against the comparator that must lose: a constant answer.

Rule 2 wants the baseline run alongside the mechanistic work. For pillar 2 the
cheapest comparator is a model that ignores the vignette entirely and always says
the same thing, and on a set that is not answer-balanced it is a strong one — so
`eligibility_correct` and `self_consistent` mean nothing until they clear it.

Two baselines, because the two metrics are scored against different things:

  eligibility  vs the majority ground-truth answer (best constant responder)
  coupling     vs a constant answer scored against the model's own grades,
               i.e. max(rows the grade implies Yes, rows it implies No)

Offline; reads the results JSONs only.

    python scripts/pillar2_baselines.py data/ecog_v0_results_*.json
"""
from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path

Z_ALPHA, Z_BETA = 1.6449, 0.8416   # one-sided 0.05, power 0.80


def surv(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binom(n, p) — the one-sided p-value against the baseline."""
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * sqrt((ph * (1 - ph) + z * z / (4 * n)) / n) / d
    return centre - half, centre + half


def required_n(p0: float, p1: float) -> float:
    """Rows needed to call a lift from p0 to p1 at one-sided 0.05, power 0.80."""
    if p1 <= p0:
        return float("inf")
    return ((Z_ALPHA * sqrt(p0 * (1 - p0)) + Z_BETA * sqrt(p1 * (1 - p1)))**2
            / (p1 - p0)**2)


def report(path: Path) -> None:
    doc = json.loads(path.read_text())
    rows = doc["rows"]
    n = len(rows)

    yes = sum(r["eligibility"]["says"] == "Yes" for r in rows)
    true_yes = sum(r["expected_eligible"] == "Yes" for r in rows)

    correct = sum(bool(r["eligibility_correct"]) for r in rows)
    majority = max(true_yes, n - true_yes)

    coupled = sum(r["self_consistent"] for r in rows)
    implies_yes = sum(r["grading"]["pred_eligible_from_grade"] == "Yes" for r in rows)
    constant = max(implies_yes, n - implies_yes)

    print(f"\n{doc['model']}  —  {path.name}   n={n}")
    print(f"  answer split      model Yes={yes:>3} No={n-yes:<3}   truth Yes={true_yes:>3} No={n-true_yes:<3}")

    for label, k, base in (("eligibility", correct, majority),
                           ("coupling", coupled, constant)):
        lo, hi = wilson(k, n)
        p = surv(k, n, base / n)
        need = required_n(base / n, k / n)
        verdict = "clears baseline" if p < 0.05 else "NOT distinguishable from baseline"
        print(f"  {label:<12} {k:>3}/{n} = {k/n:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  {'':<12} baseline {base:>3}/{n} = {base/n:.2f}   p={p:.3f}   {verdict}")
        if need != float("inf"):
            print(f"  {'':<12} n for 80% power at this lift: {need:.0f}")
        else:
            print(f"  {'':<12} n for 80% power at this lift: n/a (no lift)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    for p in args.paths:
        if p.exists():
            report(p)
        else:
            print(f"{p}: missing")
    print()


if __name__ == "__main__":
    main()
