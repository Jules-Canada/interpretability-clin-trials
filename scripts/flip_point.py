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
always-Yes flips at t=0 and so is correct on exactly the grade-0 vignettes,
always-No never flips and scores zero. Grade-0 vignettes are floored — an
early flip there looks correct — so any lenient responder collects them for
free. Read the score against the baseline, never on its own.

The two readings need *different* baselines, because "grade 0" means the true
grade for flip==true but the model's own predicted grade for flip==own. A
lenient grader that calls many vignettes grade 0 raises its own-grade baseline,
and scoring it against the true-grade baseline credits that leniency as
coupling. The degenerate case is a model that grades everything 0 and answers
Yes to everything: it scores a perfect 39/39 on flip==own while computing
nothing at all, and only the own-grade baseline catches it.

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

# Any intermediate's criterion: "ECOG <= 1", "mRS >= 3". The label is captured so
# the report names the scale it actually scored rather than assuming ECOG.
CRIT_RE = re.compile(r"([A-Za-z][A-Za-z ]*?)\s*(<=|>=)\s*(\d+)")


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
        lambda: {"le": {}, "ge": {}, "grades": set(), "true": None,
                 "ambiguous": False, "label": ""})
    for r in rows:
        m = CRIT_RE.search(r["criterion_text"] or "")
        if not m:
            continue
        label, op, val = m.group(1).strip(), m.group(2), int(m.group(3))
        src = r["id"].rsplit("_", 1)[0]
        e = by_src[src]
        e["le" if op == "<=" else "ge"][val] = r["eligibility"]["says"]
        e["grades"].add(r["grading"]["pred_grade"])
        e["true"] = r["expected_grade"]
        e["ambiguous"] = bool(r.get("ambiguous"))
        e["label"] = label
    return dict(by_src)


def own_grade(entry: dict) -> int | None:
    """The model's grade for a vignette, or None if it did not give just one.

    The grading prompt never mentions the criterion, so all copies of a vignette
    must return the same grade. More than one means the run is non-deterministic
    and the vignette carries no usable own-grade.
    """
    return next(iter(entry["grades"])) if len(entry["grades"]) == 1 else None


def score(by_src: dict) -> dict:
    """Flip-point hits and their matched constant-responder baselines.

    Pure — no printing, so the selftest can assert on the baselines themselves
    rather than only on the hit counts.
    """
    solid = {s: e for s, e in by_src.items() if not e["ambiguous"]}
    m = len(solid)

    hit_true = hit_own = monotone = never = 0
    for e in solid.values():
        fp, mono = flip_point(e["le"])
        monotone += mono
        never += fp is None
        if mono and fp is not None:
            own = own_grade(e)
            hit_true += fp == e["true"]
            hit_own += own is not None and fp == own

    # Constant responders, scored exactly the way the model is. always-Yes
    # flips at t=0, so it hits wherever the *target* grade is 0 — and the target
    # is not the same quantity for the two readings, which is why there are two
    # baselines and not one. always-No never flips and so hits nothing, always.
    base_true = sum(e["true"] == 0 for e in solid.values())
    base_own = sum(own_grade(e) == 0 for e in solid.values())
    base_no = 0

    return {"m": m, "monotone": monotone, "never": never,
            "hit_true": hit_true, "hit_own": hit_own,
            "base_true": max(base_true, base_no), "base_own": max(base_own, base_no),
            "const_yes_true": base_true, "const_yes_own": base_own, "const_no": base_no}


def report(path: Path) -> None:
    doc = json.loads(path.read_text())
    by_src = collect(doc["rows"])
    n = len(by_src)
    if not n:
        print(f"{path.name}: no sweep rows (is this a sweep results file?)")
        return

    nondet = {s: e["grades"] for s, e in by_src.items() if len(e["grades"]) > 1}
    s = score(by_src)
    m = s["m"]

    print(f"\n{doc['model']}  —  {path.name}")
    print(f"  {n} vignettes swept ({n - m} ambiguous, excluded from the primary metric)")
    if nondet:
        print(f"  !! {len(nondet)} vignette(s) gave different grades across identical "
              f"grading prompts — the run is not deterministic; stop and investigate")
        for src, g in list(nondet.items())[:5]:
            print(f"       {src}: grades {sorted(g)}")
    print(f"  monotone (answers form a step)  {s['monotone']}/{m} = {s['monotone']/m:.2f}")
    print(f"  never eligible at any threshold  {s['never']}/{m}")

    for label, k, base in (("flip point == true grade", s["hit_true"], s["base_true"]),
                           ("flip point == own grade ", s["hit_own"], s["base_own"])):
        lo, hi = wilson(k, m)
        p = surv(k, m, base / m) if m else 1.0
        verdict = "clears baseline" if p < 0.05 else "NOT distinguishable from baseline"
        print(f"  {label}  {k}/{m} = {k/m:.2f}  95% CI [{lo:.2f}, {hi:.2f}]"
              f"   vs base {base}/{m}   p={p:.4f}  {verdict}")
    print(f"  constant-responder baseline: always-Yes {s['const_yes_true']}/{m} vs true grade,"
          f" {s['const_yes_own']}/{m} vs own grade; always-No {s['const_no']}/{m}")
    if s["const_yes_own"] > s["const_yes_true"]:
        print(f"  note: this model grades 0 more often than the key does"
              f" ({s['const_yes_own']} vs {s['const_yes_true']}) — a lenient grader raises"
              f" its own coupling baseline, so flip==own is held to a higher bar here")

    # Direction check. `>= g` is the exact complement of `<= g-1` on an integer
    # scale, so the pair to compare is derived from whichever reversed criterion
    # the stimulus file actually carries (ECOG >= 2 vs <= 1; mRS >= 3 vs <= 2).
    # A model reading direction answers them differently; one pattern-matching the
    # criterion's shape echoes the `<=` answer. A constant responder echoes 1.00.
    solid = {k: v for k, v in by_src.items() if not v["ambiguous"]}
    label = next((e["label"] for e in solid.values() if e["label"]), "criterion")
    ge = echo = tot = base = 0
    gv = None
    for e in solid.values():
        if not e["ge"]:
            continue
        g = min(e["ge"])
        a, b = e["ge"][g], e["le"].get(g - 1)
        if a is None or b is None:
            continue
        gv = g
        tot += 1
        ge += (a == "Yes") == (e["true"] >= g)
        echo += a == b
        base += e["true"] >= g          # always-Yes is correct exactly here
    if tot:
        p = surv(ge, tot, base / tot) if tot else 1.0
        verdict = "clears baseline" if p < 0.05 else "NOT distinguishable from baseline"
        print(f"  direction check ({label} >= {gv})")
        print(f"      correct {ge}/{tot} = {ge/tot:.2f}   vs always-Yes {base}/{tot}"
              f"   p={p:.4f}  {verdict}")
        print(f"      echoes the <= {gv - 1} answer {echo}/{tot} = {echo/tot:.2f}"
              f"   (a constant responder echoes 1.00)")


def selftest() -> None:
    """Synthesise responders with known behaviour and check the scoring."""
    grades = [0]*6 + [1]*12 + [2]*12 + [3]*5 + [4]*4        # ecog_v0's distribution
    ts = [0, 1, 2, 3, 4]

    def build(answer_fn, grade_fn=lambda g: g):
        return [{"id": f"V{i:03d}_LE{t}", "criterion_text": f"ECOG <= {t} required",
                 "expected_grade": g, "ambiguous": False,
                 "eligibility": {"says": answer_fn(g, t)},
                 "grading": {"pred_grade": grade_fn(g)}}
                for i, g in enumerate(grades) for t in ts]

    # name: (answer_fn, grade_fn, want_true, want_own, want_base_true, want_base_own)
    cases = {
        "perfect":     (lambda g, t: "Yes" if g <= t else "No", lambda g: g, 39, 39, 6, 6),
        "always-Yes":  (lambda g, t: "Yes",                     lambda g: g,  6,  6, 6, 6),
        "always-No":   (lambda g, t: "No",                      lambda g: g,  0,  0, 6, 6),
        # A uniformly lenient model flips one step early — except at grade 0,
        # where the scale floors and an early flip is indistinguishable from a
        # correct one. Those 6 vignettes are free to any lenient responder, which
        # is exactly why the always-Yes baseline is 6/39 and not 0.
        "off-by-one":  (lambda g, t: "Yes" if g <= t + 1 else "No", lambda g: g, 6, 6, 6, 6),
        # The degenerate case the own-grade baseline exists to catch: grade
        # everything 0 and answer Yes to everything. flip==own is a perfect
        # 39/39 while the model computes nothing, and it is only not a discovery
        # because the always-Yes responder scores 39/39 against those grades too.
        "grade-0-collapse": (lambda g, t: "Yes", lambda g: 0, 6, 39, 6, 39),
    }
    ok = True
    for name, (fn, gfn, want_true, want_own, want_bt, want_bo) in cases.items():
        s = score(collect(build(fn, gfn)))
        good = ((s["hit_true"], s["hit_own"]) == (want_true, want_own)
                and (s["base_true"], s["base_own"]) == (want_bt, want_bo))
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<17} "
              f"flip==true {s['hit_true']:>2}/39 (want {want_true})   "
              f"flip==own {s['hit_own']:>2}/39 (want {want_own})   "
              f"base true/own {s['base_true']}/{s['base_own']} "
              f"(want {want_bt}/{want_bo})")

    # The point of the two baselines: the collapse case must NOT look significant.
    s = score(collect(build(lambda g, t: "Yes", lambda g: 0)))
    p_own = surv(s["hit_own"], s["m"], s["base_own"] / s["m"])
    p_wrong = surv(s["hit_own"], s["m"], s["base_true"] / s["m"])
    good = p_own > 0.05 and p_wrong < 0.05
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  grade-0-collapse is not a discovery "
          f"(p vs own base {p_own:.3f} > 0.05; against the true-grade base it would "
          f"have read {p_wrong:.2e})")

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
