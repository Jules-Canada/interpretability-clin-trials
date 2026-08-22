#!/usr/bin/env python3
"""Flip-point scoring for the pillar-2 threshold sweep.

Each vignette appears under several `ECOG <= t` criteria. A model that applies
the stated threshold answers Yes while t >= its grade and No below that, so its
answers across t form a step function and the step location — the flip point —
is the quantity of interest. The correct flip point is the vignette's true grade.

Three readings, deliberately separated:

  vs true grade   application AND completeness together; fails if either does
  vs own grade    application alone — ADR-0007's coupling, and the informative
                  one, because it holds regardless of whether the grade is right
  monotonicity    whether the answers form a step function at all; a model that
                  flips back and forth is not reading the criterion as a
                  threshold and its flip point is not meaningful

The constant-responder baseline is computed from the data rather than assumed:
always-Yes flips at t=0 and so is correct on exactly the true-grade-0 vignettes,
always-No never flips and scores zero. Grade-0 vignettes are floored — an
early flip there looks correct — so any lenient responder collects them for
free. Read the score against the baseline, never on its own.

    python scripts/flip_point.py data/ecog_sweep_v0_results_*.json
    python scripts/flip_point.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

CRIT_RE = re.compile(r"ECOG\s*(<=|>=)\s*(\d+)")


def surv(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * sqrt((ph * (1 - ph) + z * z / (4 * n)) / n) / d
    return c - h, c + h


def flip_point(answers: dict[int, str]) -> tuple[int | None, bool]:
    """Step location of Yes-above / No-below over ascending threshold.

    answers: {threshold: "Yes"|"No"}
    returns: (flip point, monotone). Flip point is the lowest t answered Yes;
    None if the model never says Yes. Monotone means every t above the flip is
    also Yes — i.e. the answers really are a step.
    """
    ts = sorted(answers)
    yes = [t for t in ts if answers[t] == "Yes"]
    if not yes:
        return None, True                      # never eligible: a step at +inf
    fp = min(yes)
    monotone = all(answers[t] == "Yes" for t in ts if t >= fp)
    return fp, monotone


def collect(rows: list[dict]) -> dict:
    """Group sweep rows by source vignette. returns: {source_id: {...}}"""
    by_src: dict[str, dict] = defaultdict(
        lambda: {"le": {}, "ge": {}, "grades": set(), "true": None, "ambiguous": False})
    for r in rows:
        m = CRIT_RE.search(r["criterion_text"] or "")
        if not m:
            continue
        op, val = m.group(1), int(m.group(2))
        src = r["id"].rsplit("_", 1)[0]
        e = by_src[src]
        e["le" if op == "<=" else "ge"][val] = r["eligibility"]["says"]
        e["grades"].add(r["grading"]["pred_grade"])
        e["true"] = r["expected_grade"]
        e["ambiguous"] = bool(r.get("ambiguous"))
    return dict(by_src)


def report(path: Path) -> None:
    doc = json.loads(path.read_text())
    by_src = collect(doc["rows"])
    n = len(by_src)
    if not n:
        print(f"{path.name}: no sweep rows (is this a sweep results file?)")
        return

    nondet = {s: e["grades"] for s, e in by_src.items() if len(e["grades"]) > 1}
    solid = {s: e for s, e in by_src.items() if not e["ambiguous"]}

    hit_true = hit_own = monotone = never = 0
    for e in solid.values():
        fp, mono = flip_point(e["le"])
        monotone += mono
        never += fp is None
        own = next(iter(e["grades"])) if len(e["grades"]) == 1 else None
        if mono and fp is not None:
            hit_true += fp == e["true"]
            hit_own += own is not None and fp == own
    m = len(solid)

    # constant responders, scored the same way
    const_yes = sum(e["true"] == 0 for e in solid.values())   # flips at t=0
    const_no = 0                                              # never flips

    print(f"\n{doc['model']}  —  {path.name}")
    print(f"  {n} vignettes swept ({n - m} ambiguous, excluded from the primary metric)")
    if nondet:
        print(f"  !! {len(nondet)} vignette(s) gave different grades across identical "
              f"grading prompts — the run is not deterministic; stop and investigate")
        for s, g in list(nondet.items())[:5]:
            print(f"       {s}: grades {sorted(g)}")
    print(f"  monotone (answers form a step)  {monotone}/{m} = {monotone/m:.2f}")
    print(f"  never eligible at any threshold  {never}/{m}")

    base = max(const_yes, const_no)
    for label, k in (("flip point == true grade", hit_true),
                     ("flip point == own grade ", hit_own)):
        lo, hi = wilson(k, m)
        p = surv(k, m, base / m) if m else 1.0
        verdict = "clears baseline" if p < 0.05 else "NOT distinguishable from baseline"
        print(f"  {label}  {k}/{m} = {k/m:.2f}  95% CI [{lo:.2f}, {hi:.2f}]   p={p:.4f}  {verdict}")
    print(f"  constant-responder baseline: always-Yes {const_yes}/{m}, always-No {const_no}/{m}")

    # direction check: >= 2 should inverse <= 1 on grade, not echo it
    ge, echo, tot = 0, 0, 0
    for e in solid.values():
        a, b = e["ge"].get(2), e["le"].get(1)
        if a is None or b is None:
            continue
        tot += 1
        ge += (a == "Yes") == (e["true"] >= 2)
        echo += a == b
    if tot:
        print(f"  direction check (ECOG >= 2)     correct {ge}/{tot} = {ge/tot:.2f}"
              f"   echoes the <= 1 answer {echo}/{tot} = {echo/tot:.2f}")


def selftest() -> None:
    """Synthesise responders with known behaviour and check the scoring."""
    grades = [0]*6 + [1]*12 + [2]*12 + [3]*5 + [4]*4        # ecog_v0's distribution
    ts = [0, 1, 2, 3, 4]

    def build(answer_fn):
        return [{"id": f"V{i:03d}_LE{t}", "criterion_text": f"ECOG <= {t} required",
                 "expected_grade": g, "ambiguous": False,
                 "eligibility": {"says": answer_fn(g, t)},
                 "grading": {"pred_grade": g}}
                for i, g in enumerate(grades) for t in ts]

    cases = {
        "perfect":     (lambda g, t: "Yes" if g <= t else "No", 39, 39),
        "always-Yes":  (lambda g, t: "Yes",                      6,  6),
        "always-No":   (lambda g, t: "No",                       0,  0),
        # A uniformly lenient model flips one step early — except at grade 0,
        # where the scale floors and an early flip is indistinguishable from a
        # correct one. Those 6 vignettes are free to any lenient responder, which
        # is exactly why the always-Yes baseline is 6/39 and not 0.
        "off-by-one":  (lambda g, t: "Yes" if g <= t + 1 else "No", 6, 6),
    }
    ok = True
    for name, (fn, want_true, want_own) in cases.items():
        by = collect(build(fn))
        hit_t = hit_o = 0
        for e in by.values():
            fp, mono = flip_point(e["le"])
            if mono and fp is not None:
                hit_t += fp == e["true"]
                hit_o += fp == next(iter(e["grades"]))
        good = (hit_t, hit_o) == (want_true, want_own)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<12} "
              f"flip==true {hit_t:>2}/39 (want {want_true})   "
              f"flip==own {hit_o:>2}/39 (want {want_own})")

    # a non-monotone responder must be caught, not scored
    fp, mono = flip_point({0: "Yes", 1: "No", 2: "Yes", 3: "Yes", 4: "Yes"})
    good = (fp, mono) == (0, False)
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  non-monotone flagged (fp={fp}, monotone={mono})")

    fp, mono = flip_point({0: "No", 1: "No", 2: "No", 3: "No", 4: "No"})
    good = (fp, mono) == (None, True)
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  never-Yes -> no flip point (fp={fp})")
    print("\nselftest:", "OK" if ok else "FAILURES")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.paths:
        ap.error("give one or more results JSONs, or --selftest")
    for p in args.paths:
        report(p) if p.exists() else print(f"{p}: missing")
    print()


if __name__ == "__main__":
    main()
