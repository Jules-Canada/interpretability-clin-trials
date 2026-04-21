#!/usr/bin/env python
"""
scripts/label_features.py

Read feature activation contexts from find_top_activations.py output and
prompt Claude to produce a natural-language label for each feature.

This is step 2 of the feature labeling pipeline:
  1. find_top_activations.py  — find top activating token contexts
  2. label_features.py        ← you are here
  3. graphs/export.py         — uses labels to populate clerp field

Output: data/feature_labels.jsonl
  One JSON object per feature:
    {
      "layer": 4,
      "feature": 211,
      "label": "tokens following conjunctions in medical eligibility lists",
      "raw_response": "..."
    }

Usage:
    python scripts/label_features.py \\
        --activations_path data/feature_activations.jsonl \\
        --output_path data/feature_labels.jsonl

    # Dry-run (print first 3 prompts, don't call API):
    python scripts/label_features.py --dry_run --n_dry 3

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


SYSTEM_PROMPT = """\
You are an expert in mechanistic interpretability of language models. \
Your job is to identify what pattern or concept a neural network feature \
detects, based on the tokens that most activate it.

You will be shown the top activating token contexts for a single feature. \
Each context shows a short token window, with the activating token marked \
with >>> ... <<<. Token strings may include special prefixes like Ġ (space) \
or Ċ (newline).

Write a concise natural-language label (5–12 words) describing:
- What type of token or context pattern activates this feature
- The semantic or syntactic role of the activating token

Be specific. Prefer "tokens following 'and' in list enumerations" over \
"list tokens". If the pattern is unclear from the examples, write \
"unclear pattern" rather than guessing.

Respond with ONLY the label — no explanation, no bullet points, no quotes."""


def format_context(example: dict) -> str:
    """Format a single top activation example for the prompt."""
    tokens = example.get("context_tokens")
    if not tokens:
        # Fall back to token indices if no decoded tokens available
        ids = example.get("context_token_ids", [])
        return f"  activation={example['activation']:.3f}  token_ids={ids}"

    target_pos = example.get("target_token_pos", 0)
    # Mark the activating token
    parts = []
    for i, tok in enumerate(tokens):
        if i == target_pos:
            parts.append(f">>>{tok}<<<")
        else:
            parts.append(tok)
    context_str = "".join(parts)
    return f"  activation={example['activation']:.3f}  context: {context_str}"


def build_user_prompt(layer: int, feature: int, top_examples: list[dict]) -> str:
    lines = [f"Feature L{layer}F{feature} — top activating contexts:\n"]
    for i, ex in enumerate(top_examples[:10], 1):  # cap at 10 examples
        lines.append(f"{i}. {format_context(ex)}")
    return "\n".join(lines)


def label_one(
    client,
    layer: int,
    feature: int,
    top_examples: list[dict],
    model: str,
) -> tuple[str, str]:
    """
    Returns (label, raw_response).
    """
    user_prompt = build_user_prompt(layer, feature, top_examples)
    message = client.messages.create(
        model=model,
        max_tokens=64,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = message.content[0].text.strip()
    # Strip surrounding quotes if Claude added them
    label = raw.strip('"\'')
    return label, raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label CLT features using Claude API.")
    p.add_argument("--activations_path", type=str, default="data/feature_activations.jsonl",
                   help="Output of find_top_activations.py")
    p.add_argument("--output_path",      type=str, default="data/feature_labels.jsonl",
                   help="Where to write labels (appended, not overwritten)")
    p.add_argument("--model",            type=str, default="claude-haiku-4-5-20251001",
                   help="Claude model to use for labeling")
    p.add_argument("--rate_limit_rps",   type=float, default=4.0,
                   help="Requests per second to stay under API rate limits")
    p.add_argument("--dry_run",          action="store_true",
                   help="Print prompts without calling API")
    p.add_argument("--n_dry",            type=int, default=5,
                   help="Number of features to print in dry-run mode")
    p.add_argument("--resume",           action="store_true",
                   help="Skip features already in output_path")
    return p.parse_args()


def load_already_labeled(output_path: Path) -> set[tuple[int, int]]:
    """Return set of (layer, feature) pairs already in the output file."""
    done: set[tuple[int, int]] = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((int(obj["layer"]), int(obj["feature"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    args = parse_args()

    activations_path = Path(args.activations_path)
    if not activations_path.exists():
        print(f"ERROR: activations file not found: {activations_path}")
        print("Run find_top_activations.py first.")
        sys.exit(1)

    # Load all features
    entries: list[dict] = []
    with open(activations_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Filter to only features that have activation examples
    entries = [e for e in entries if e.get("top_examples")]
    print(f"Loaded {len(entries)} features with activations from {activations_path}")

    if args.dry_run:
        print(f"\n--- DRY RUN: printing first {args.n_dry} prompts ---\n")
        for entry in entries[:args.n_dry]:
            layer = entry["layer"]
            feat  = entry["feature"]
            prompt = build_user_prompt(layer, feat, entry["top_examples"])
            print(f"=== L{layer}F{feat} ===")
            print(prompt)
            print()
        return

    # Real labeling
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = None
    try:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    except Exception:
        pass

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[tuple[int, int]] = set()
    if args.resume:
        already_done = load_already_labeled(output_path)
        print(f"Resuming: {len(already_done)} features already labeled, skipping.")

    to_label = [
        e for e in entries
        if (int(e["layer"]), int(e["feature"])) not in already_done
    ]
    print(f"Labeling {len(to_label)} features with {args.model} ...")
    print(f"Estimated cost: ${len(to_label) * 0.0004:.2f}  (~$0.0004/feature at Haiku rates)\n")

    min_interval = 1.0 / args.rate_limit_rps
    n_ok = 0
    n_err = 0

    with open(output_path, "a") as out_f:
        for i, entry in enumerate(to_label):
            layer = int(entry["layer"])
            feat  = int(entry["feature"])
            t0 = time.time()

            try:
                label, raw = label_one(client, layer, feat, entry["top_examples"], args.model)
                record = {
                    "layer":        layer,
                    "feature":      feat,
                    "label":        label,
                    "raw_response": raw,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_ok += 1
                print(f"  [{i+1}/{len(to_label)}] L{layer}F{feat}: {label}")
            except Exception as e:
                n_err += 1
                print(f"  [{i+1}/{len(to_label)}] L{layer}F{feat}: ERROR — {e}")

            # Rate limiting
            elapsed = time.time() - t0
            sleep_for = max(0.0, min_interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    print(f"\nDone: {n_ok} labeled, {n_err} errors → {output_path}")
    print("Next step: python graphs/export.py  (to populate clerp field from labels)")


if __name__ == "__main__":
    main()
