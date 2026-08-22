#!/usr/bin/env python3
"""Recompute the derived verdicts in an existing results JSON, offline.

The model outputs — grade distribution, predicted grade, Yes/No mass — are
stored per row, so everything downstream of them (pred_eligible_from_grade,
self_consistent, dissociation, the summary) is a pure function of the file and
can be corrected without pod time.

Written for the 2026-08-21 threshold fix: `pred_eligible_from_grade` used the
intermediate's default rule for every row, so the reversed-threshold distractors
were scored against the criterion they were written to violate. Scoring logic is
imported from run_ecog_stimuli rather than restated here, so the two cannot drift.

    python scripts/rescore_results.py data/ecog_v0_results_*.json
    python scripts/rescore_results.py --check data/*.json     # report, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ecog_stimuli import INTERMEDIATES, eligible_rule, score_rows, summarise

# `intermediate` is stored as the display name ("ECOG performance status"), so
# map back to the config key rather than guessing from the filename.
BY_NAME = {cfg["name"]: key for key, cfg in INTERMEDIATES.items()}


def rescore(path: Path, write: bool) -> int:
    """Rewrite one results file in place. returns: number of rows that changed."""
    doc = json.loads(path.read_text())
    rows = doc["rows"]

    key = BY_NAME.get(doc.get("intermediate"))
    if key is None:
        print(f"  {path.name}: unknown intermediate {doc.get('intermediate')!r} — skipped")
        return 0
    cfg = INTERMEDIATES[key]

    before = [(r["grading"]["pred_eligible_from_grade"], r["self_consistent"],
               r["dissociation"]) for r in rows]

    for r in rows:
        rule, source = eligible_rule(r, cfg)
        r["grading"]["pred_eligible_from_grade"] = "Yes" if rule(r["grading"]["pred_grade"]) else "No"
        r["grading"]["threshold_source"] = source
    score_rows(rows)

    changed = [r for r, b in zip(rows, before)
               if (r["grading"]["pred_eligible_from_grade"], r["self_consistent"],
                   r["dissociation"]) != b]

    old_sum = doc["summary"]
    new_sum = summarise(rows)
    print(f"  {path.name}")
    for r, b in zip(rows, before):
        now = (r["grading"]["pred_eligible_from_grade"], r["self_consistent"],
               r["dissociation"])
        if now != b:
            print(f"    {r['id']}: {r['criterion_text']}")
            print(f"      pred_grade={r['grading']['pred_grade']} says={r['eligibility']['says']}"
                  f"  implied {b[0]} -> {now[0]}   self_consistent {b[1]} -> {now[1]}")
    for k in ("self_consistent", "eligibility_correct", "grade_correct"):
        if old_sum.get(k) != new_sum.get(k):
            print(f"    summary.{k}: {old_sum.get(k)} -> {new_sum.get(k)}")
    if old_sum.get("dissociation") != new_sum.get("dissociation"):
        print(f"    summary.dissociation: {old_sum.get('dissociation')}")
        print(f"                       -> {new_sum.get('dissociation')}")
    if not changed:
        print("    no change")

    if write and changed:
        doc["summary"] = new_sum
        doc.setdefault("rescored", []).append({
            "date": date.today().isoformat(),
            "reason": "per-row criterion threshold (reversed-threshold distractors)",
            "rows_changed": [r["id"] for r in changed],
        })
        path.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"    written ({len(changed)} row(s) changed)")
    return len(changed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    total = 0
    for p in args.paths:
        if not p.exists():
            print(f"  {p}: missing"); continue
        total += rescore(p, write=not args.check)
    print(f"\n{total} row(s) changed across {len(args.paths)} file(s)"
          + ("  (--check: nothing written)" if args.check else ""))


if __name__ == "__main__":
    main()
