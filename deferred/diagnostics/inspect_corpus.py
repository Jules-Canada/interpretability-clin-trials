#!/usr/bin/env python3
"""
scripts/inspect_corpus.py

Inspect and optionally clean a clinical trial protocol JSONL corpus.
Reports quality metrics across a sample, then writes a cleaned version.

Usage - inspect only (no output written):
    python scripts/inspect_corpus.py --input /path/to/protocols.jsonl

Usage - inspect + write cleaned file:
    python scripts/inspect_corpus.py \
        --input /path/to/protocols.jsonl \
        --output data/protocols_clean.jsonl \
        --sample 200

Cleaning applied:
  - Strip whitespace-only lines (PDF table column padding artifacts)
  - Collapse 3+ consecutive blank lines to 2
  - Remove page header/footer lines ("Page N of M")
  - Replace private-use Unicode (U+E000-U+F8FF, Wingdings bullets) with standard bullet
  - Remove ASCII control characters (OCR artifacts from corrupt PDFs)
  - Strip trailing whitespace per line
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# Private-use Unicode range U+E000-U+F8FF (Wingdings/Symbol font from PDF)
# Private-use Unicode range U+E000-U+F8FF (Wingdings/Symbol font from PDF)
# Built with chr() to avoid source-file encoding issues with these codepoints.
_PRIVATE_USE = re.compile("[%s-%s]" % (chr(0xE000), chr(0xF8FF)))
_CONTROL_CHARS = re.compile("[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Whitespace-only line
_BLANK_LINE = re.compile(r"^\s+$", re.MULTILINE)
# 3+ consecutive newlines
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
# Page header/footer artifacts from PDF extraction
_PAGE_ARTIFACT = re.compile(
    r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_text(text: str) -> str:
    # Replace private-use Unicode bullets with standard bullet
    text = _PRIVATE_USE.sub("•", text)
    # Remove ASCII control characters (OCR artifacts from corrupt PDF extraction)
    text = _CONTROL_CHARS.sub("", text)
    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Remove lines that are purely whitespace (table padding artifacts)
    text = _BLANK_LINE.sub("", text)
    # Remove page header/footer lines
    text = _PAGE_ARTIFACT.sub("", text)
    # Collapse 3+ newlines to 2
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose(text: str) -> dict:
    lines = text.split("\n")
    blank_lines = sum(1 for l in lines if not l.strip())
    whitespace_only = sum(1 for l in lines if l and not l.strip())
    triple_newlines = len(re.findall(r"\n{3,}", text))
    private_use = len(_PRIVATE_USE.findall(text))
    control_chars = len(_CONTROL_CHARS.findall(text))
    page_artifacts = len(_PAGE_ARTIFACT.findall(text))
    unicode_bullets = text.count("•") + text.count("●") + text.count("·")
    ws_ratio = sum(1 for c in text if c in " \t\n\r") / max(len(text), 1)
    return {
        "chars": len(text),
        "lines": len(lines),
        "blank_lines": blank_lines,
        "whitespace_only_lines": whitespace_only,
        "triple_newlines": triple_newlines,
        "private_use_unicode": private_use,
        "control_chars": control_chars,
        "page_artifacts": page_artifacts,
        "unicode_bullets": unicode_bullets,
        "whitespace_ratio": ws_ratio,
    }


def summarise(metrics_list: list[dict]) -> None:
    n = len(metrics_list)
    keys = list(metrics_list[0].keys())
    print(f"\n{'Metric':<30} {'Mean':>10} {'Max':>10} {'Docs with issue':>16}")
    print("-" * 70)
    for key in keys:
        vals = [m[key] for m in metrics_list]
        mean = sum(vals) / n
        mx = max(vals)
        nonzero = sum(1 for v in vals if v > 0)
        if key == "whitespace_ratio":
            print(f"{key:<30} {mean:>9.1%} {mx:>9.1%} {nonzero:>14d}/{n}")
        elif key in ("chars", "lines"):
            print(f"{key:<30} {mean:>10,.0f} {mx:>10,} {nonzero:>14d}/{n}")
        else:
            print(f"{key:<30} {mean:>10.1f} {mx:>10} {nonzero:>14d}/{n}")
    print()


def show_examples(text: str, pattern: re.Pattern, label: str, context: int = 80) -> None:
    matches = list(pattern.finditer(text))
    if not matches:
        return
    print(f"\n  {label} ({len(matches)} found):")
    for m in matches[:3]:
        start = max(0, m.start() - 20)
        snippet = repr(text[start : start + context])
        print(f"    {snippet}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect and clean a clinical trial JSONL corpus.")
    p.add_argument("--input",      required=True,  help="Input JSONL path")
    p.add_argument("--output",     default=None,   help="Output JSONL path (omit for dry run)")
    p.add_argument("--sample",     type=int, default=100, help="Docs to sample for diagnostics")
    p.add_argument("--text_field", default="full_text",   help="JSON field containing document text")
    p.add_argument("--verbose",    action="store_true",   help="Show per-doc issue examples")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)

    print(f"Corpus: {in_path}")
    print(f"Sample size: {args.sample} docs")

    total = sum(1 for _ in open(in_path))
    print(f"Total docs: {total:,}")

    step = max(1, total // args.sample)
    sample_indices = set(range(0, total, step))

    raw_metrics: list[dict] = []
    clean_metrics: list[dict] = []
    sampled_docs: list[tuple] = []

    with open(in_path) as f:
        for i, line in enumerate(f):
            if i not in sample_indices:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get(args.text_field, "")
            if not text:
                continue
            raw_metrics.append(diagnose(text))
            cleaned = clean_text(text)
            clean_metrics.append(diagnose(cleaned))
            sampled_docs.append((doc, text, cleaned))
            if len(sampled_docs) >= args.sample:
                break

    print(f"\nSampled: {len(sampled_docs)} docs")

    print("\n=== RAW corpus diagnostics ===")
    summarise(raw_metrics)

    print("=== AFTER cleaning ===")
    summarise(clean_metrics)

    raw_chars   = sum(m["chars"] for m in raw_metrics)
    clean_chars = sum(m["chars"] for m in clean_metrics)
    print(f"Char reduction: {raw_chars:,} -> {clean_chars:,} ({(raw_chars - clean_chars) / raw_chars:.1%} removed)\n")

    if args.verbose:
        print("=== Per-doc issue examples (first 3 with issues) ===")
        shown = 0
        for doc, raw_text, _ in sampled_docs:
            if shown >= 3:
                break
            m = diagnose(raw_text)
            if (m["private_use_unicode"] == 0 and m["control_chars"] == 0
                    and m["page_artifacts"] == 0 and m["whitespace_only_lines"] < 5):
                continue
            print(f"\n  {doc.get('nct_id', '?')} ({m['chars']:,} chars):")
            show_examples(raw_text, _PRIVATE_USE,   "Private-use Unicode")
            show_examples(raw_text, _CONTROL_CHARS,  "Control characters")
            show_examples(raw_text, _PAGE_ARTIFACT,  "Page artifacts")
            show_examples(raw_text, _BLANK_LINE,     "Whitespace-only lines")
            shown += 1

    if not args.output:
        print("(No --output specified - dry run only. Add --output to write cleaned file.)")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing cleaned corpus to {out_path} ...")

    written = 0
    skipped = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            text = doc.get(args.text_field, "")
            if not text.strip():
                skipped += 1
                continue
            doc[args.text_field] = clean_text(text)
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. {written:,} docs written, {skipped} skipped.")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
