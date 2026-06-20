#!/usr/bin/env python
"""
scripts/profile_criteria.py

Profile the STRUCTURE of real ClinicalTrials.gov eligibility criteria, to ground
the prompt redesign (prompts/eligibility.py) in the real distribution rather than
guesses. Answers: what structural features do real I/E criteria use, and how
off-distribution are our synthetic prompts?

Reads the local protocol corpus (JSONL, one protocol per line, `full_text`
field; see CLAUDE.md Findings). For a sample of protocols it locates the
eligibility section(s) and reports the share containing each structural feature:
numeric+unit thresholds, temporal qualifiers, EXCEPTIONS/carve-outs,
conjunction/aggregation, named instruments, relative (x ULN) thresholds.

Headline result (3,000-doc sample, 2026-06-19): carve-outs appear in ~44% of
real protocols and numeric thresholds in ~76% — both axes our Round-1 prompts
tested at ~0%. Motivated the structure-spanning redesign + the exception tier.

Usage:
    python scripts/profile_criteria.py
    python scripts/profile_criteria.py --n 5000 \
        --corpus ~/Desktop/protocol_corpus/ct_corpus/protocols.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re

DEFAULT_CORPUS = os.path.expanduser(
    "~/Desktop/protocol_corpus/ct_corpus/protocols.jsonl"
)

# Section-level structural features (matched anywhere in the eligibility blob).
FEATURES = {
    "numeric_threshold": re.compile(
        r"[<>≤≥]\s*\d|\d+\s*(mg|g/dl|x 10|/μl|/ul|ml/min|mmhg)", re.I),
    "temporal_qualifier": re.compile(
        r"(prior \d+\s*(days|weeks|months)|within \d+\s*(days|weeks|months))", re.I),
    "exception_carveout": re.compile(
        r"\b(unless|except|are eligible if|is eligible if|with the exception|"
        r"gilbert|provided that)\b", re.I),
    "conjunction": re.compile(r"\bdefined as\b|:\s*$", re.I),
    "named_instrument": re.compile(
        r"\b(RECIST|ECOG|CTCAE|NYHA|Karnofsky|Cockcroft|NCI[- ]CTC)\b"),
    "relative_threshold": re.compile(
        r"(x|×|times)\s*(the\s*)?(upper limit|uln)", re.I),
}

ELIG_MARKER = re.compile(r"(inclusion criteria|exclusion criteria)", re.I)


def eligibility_blob(full_text: str, window: int = 2500) -> str:
    """Concatenate the spans following each Inclusion/Exclusion Criteria marker.

    Crude but robust: we don't trust per-line splitting (it shatters nested
    criteria and biases length stats), so we only report section-level feature
    presence over the blob, not per-criterion length.
    """
    spans = [full_text[m.start():m.start() + window]
             for m in ELIG_MARKER.finditer(full_text)]
    return "\n".join(spans)


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile real eligibility-criteria structure.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--n", type=int, default=3000, help="protocols to scan")
    args = ap.parse_args()

    n_docs = 0
    counts = {k: 0 for k in FEATURES}
    with open(args.corpus) as f:
        for _ in range(args.n):
            line = f.readline()
            if not line:
                break
            blob = eligibility_blob(json.loads(line)["full_text"])
            if not blob:
                continue
            n_docs += 1
            for name, pat in FEATURES.items():
                if pat.search(blob):
                    counts[name] += 1

    print(f"corpus: {args.corpus}")
    print(f"protocols with an eligibility section: {n_docs}\n")
    print("structural feature present in eligibility section:")
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name:20} {c/n_docs:5.0%}   ({c}/{n_docs})")
    print("\nOur Round-1 prompts hit ~0% on every axis except numeric_threshold "
          "(age/ecog only). The redesign spans these — see prompts/eligibility.py.")


if __name__ == "__main__":
    main()
