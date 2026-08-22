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

SRC = Path("specs/stimuli/ecog_v0.csv")
OUT = Path("specs/stimuli/ecog_sweep_v0.csv")

# (operator, value, tier). "primary" is the pre-registered contrast: both are
# common real oncology protocols, so a failure there is about the criterion and
# not about the protocol being strange. The rest are exploratory —
# `<= 0` and `<= 3` are rare in practice, `<= 4` admits everyone and exists only
# as a stuck-responder floor check, `>= 2` is the direction check.
CRITERIA = [
    ("<=", 0, "exploratory"),
    ("<=", 1, "primary"),
    ("<=", 2, "primary"),
    ("<=", 3, "exploratory"),
    ("<=", 4, "ceiling"),
    (">=", 2, "direction"),
]

SUFFIX = {"<=": "LE", ">=": "GE"}

COLUMNS = ["id", "ecog_true", "set_id", "lexical_distance", "boundary_case",
           "ambiguous", "vignette_text", "eligibility_criterion", "expected_answer",
           "distractor_type", "notes",
           # analysis-only; run_ecog_stimuli.py ignores unknown columns
           "source_id", "criterion_op", "criterion_value", "tier"]

COHORT = {"<=": "", ">=": " (supportive care cohort)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    src_rows = [r for r in csv.DictReader(args.src.open(newline="", encoding="utf-8-sig"))
                if (r.get("vignette_text") or "").strip()]

    out_rows = []
    for r in src_rows:
        grade = int(r["ecog_true"])
        for op, val, tier in CRITERIA:
            eligible = grade <= val if op == "<=" else grade >= val
            out_rows.append({
                "id": f"{r['id']}_{SUFFIX[op]}{val}",
                "ecog_true": grade,
                # set_id groups paraphrases of one clinical state; keep it scoped to
                # the criterion so the paraphrase report compares like with like.
                "set_id": f"{r['set_id']}_{SUFFIX[op]}{val}" if r["set_id"].strip() else "",
                "lexical_distance": r["lexical_distance"],
                # the boundary is now a property of the pairing, not the vignette
                "boundary_case": grade == val,
                "ambiguous": r["ambiguous"],
                "vignette_text": r["vignette_text"],
                "eligibility_criterion": f"ECOG {op} {val} required for enrollment{COHORT[op]}",
                "expected_answer": "eligible" if eligible else "excluded",
                "distractor_type": "reversed_threshold" if op == ">=" else r["distractor_type"],
                "notes": f"sweep of {r['id']} (true ECOG {grade}) at {op} {val}",
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
        print(f"  ECOG {op} {val:<2} [{tier:<11}] eligible {y:>2}/{len(sub)}  = {y/len(sub):.2f}")
    y = sum(r["expected_answer"] == "eligible" for r in out_rows)
    print(f"  {'POOLED':<21} eligible {y:>3}/{len(out_rows)} = {y/len(out_rows):.2f}")


if __name__ == "__main__":
    main()
