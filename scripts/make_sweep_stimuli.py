#!/usr/bin/env python3
"""Cross ecog_v0's vignettes with a range of criteria — the pillar-2 sweep set.

The 2026-08-06 pilot varied the vignette and held the criterion fixed (38 of 39
rows were `ECOG <= 1`). That makes eligibility accuracy a between-item measure,
and on an unbalanced answer key a constant responder scores 0.56 — which is
exactly what Gemma scored (see docs/run-log.md 2026-08-21).

Inverting it costs no clinician time. Hold the vignette fixed, sweep the
criterion, and read the *flip point*: the threshold at which the model switches
Yes to No. The correct flip point is the vignette's own true grade, so each
vignette yields one number and a constant responder yields no flip point at all
— it scores zero rather than 0.56.

Writes the same schema as ecog_v0.csv, so run_ecog_stimuli.py reads it unchanged.
Grading prompts do not mention the criterion, so the N copies of a vignette must
all produce an identical grade; that redundancy is the determinism check.

    python scripts/make_sweep_stimuli.py            # -> specs/stimuli/ecog_sweep_v0.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

SUFFIX = {"<=": "LE", ">=": "GE"}

# Per intermediate: the scale's own vocabulary and its criterion ladder.
# (operator, value, tier). "primary" is the pre-registered contrast — criteria a
# real protocol would actually state, so a failure there is about the criterion
# and not about the protocol being strange. "ceiling" admits everyone and exists
# only as a stuck-responder floor check; "direction" reverses the comparison and
# asks whether the model reads direction or pattern-matches the primary form.
INTERMEDIATES = {
    "ecog": {
        "label": "ECOG",
        "true_col": "ecog_true",
        "src": Path("specs/stimuli/ecog_v0.csv"),
        "out": Path("specs/stimuli/ecog_sweep_v0.csv"),
        # `<= 1` and `<= 2` are common real oncology protocols; `<= 0` and `<= 3`
        # are rare, so a failure there may be off-distribution rather than a
        # failure to apply the criterion.
        "criteria": [("<=", 0, "exploratory"), ("<=", 1, "primary"),
                     ("<=", 2, "primary"), ("<=", 3, "exploratory"),
                     ("<=", 4, "ceiling"), (">=", 2, "direction")],
        "cohort": {"<=": "", ">=": " (supportive care cohort)"},
    },
    "mrs": {
        "label": "mRS",
        "true_col": "mrs_true",
        "src": Path("specs/stimuli/mrs_v0.csv"),
        "out": Path("specs/stimuli/mrs_sweep_v0.csv"),
        # mRS runs 0-6, so the ladder is longer and the ceiling is 6. `<= 2` is
        # the canonical stroke-trial cutoff (the "favourable outcome" bound) and
        # `<= 1` is the stricter common form; those are the primaries. `>= 3` is
        # the direction check because it is the exact complement of `<= 2`.
        "criteria": [("<=", 0, "exploratory"), ("<=", 1, "primary"),
                     ("<=", 2, "primary"), ("<=", 4, "exploratory"),
                     ("<=", 6, "ceiling"), (">=", 3, "direction")],
        "cohort": {"<=": "", ">=": " (supportive care cohort)"},
    },
}


def columns_for(true_col: str) -> list[str]:
    return ["id", true_col, "set_id", "lexical_distance", "boundary_case",
            "ambiguous", "vignette_text", "eligibility_criterion", "expected_answer",
            "distractor_type", "notes",
            # analysis-only; run_ecog_stimuli.py ignores unknown columns
            "source_id", "criterion_op", "criterion_value", "tier"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intermediate", choices=sorted(INTERMEDIATES), default="ecog")
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = INTERMEDIATES[args.intermediate]
    label, true_col = cfg["label"], cfg["true_col"]
    CRITERIA, COHORT = cfg["criteria"], cfg["cohort"]
    COLUMNS = columns_for(true_col)
    args.src = args.src or cfg["src"]
    args.out = args.out or cfg["out"]

    src_rows = [r for r in csv.DictReader(args.src.open(newline="", encoding="utf-8-sig"))
                if (r.get("vignette_text") or "").strip()]

    out_rows = []
    for r in src_rows:
        grade = int(r[true_col])
        for op, val, tier in CRITERIA:
            eligible = grade <= val if op == "<=" else grade >= val
            out_rows.append({
                "id": f"{r['id']}_{SUFFIX[op]}{val}",
                true_col: grade,
                # set_id groups paraphrases of one clinical state; keep it scoped to
                # the criterion so the paraphrase report compares like with like.
                # not every stimulus file carries paraphrase sets (mrs_v0 does not)
                "set_id": (f"{r['set_id']}_{SUFFIX[op]}{val}"
                           if (r.get("set_id") or "").strip() else ""),
                "lexical_distance": r.get("lexical_distance", ""),
                # the boundary is now a property of the pairing, not the vignette
                "boundary_case": grade == val,
                "ambiguous": r.get("ambiguous", ""),
                "vignette_text": r["vignette_text"],
                "eligibility_criterion": f"{label} {op} {val} required for enrollment{COHORT[op]}",
                "expected_answer": "eligible" if eligible else "excluded",
                "distractor_type": ("reversed_threshold" if op == ">="
                                    else r.get("distractor_type", "")),
                "notes": f"sweep of {r['id']} (true {label} {grade}) at {op} {val}",
                "source_id": r["id"],
                "criterion_op": op,
                "criterion_value": val,
                "tier": tier,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(src_rows)} vignettes x {len(CRITERIA)} criteria = {len(out_rows)} rows")
    print(f"wrote {args.out}")
    for op, val, tier in CRITERIA:
        sub = [r for r in out_rows if r["criterion_op"] == op and r["criterion_value"] == val]
        y = sum(r["expected_answer"] == "eligible" for r in sub)
        print(f"  {label} {op} {val:<2} [{tier:<11}] eligible {y:>2}/{len(sub)}  = {y/len(sub):.2f}")
    y = sum(r["expected_answer"] == "eligible" for r in out_rows)
    print(f"  {'POOLED':<21} eligible {y:>3}/{len(out_rows)} = {y/len(out_rows):.2f}")


if __name__ == "__main__":
    main()
