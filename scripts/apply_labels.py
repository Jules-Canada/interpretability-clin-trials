#!/usr/bin/env python
"""
scripts/apply_labels.py

Patch the `clerp` field in existing graph JSONs with natural-language labels
from label_features.py output.

Reads feature_labels.jsonl, iterates over all graph JSONs in graph_data/,
and updates the clerp field for any cross layer transcoder node that has a label.
Nodes without a label keep their existing clerp value.

Usage:
    python scripts/apply_labels.py
    python scripts/apply_labels.py \
        --labels data/feature_labels.jsonl \
        --graph_dir frontend/graph_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch clerp fields in graph JSONs with feature labels.")
    p.add_argument("--labels",    type=str, default="data/feature_labels.jsonl")
    p.add_argument("--graph_dir", type=str, default="frontend/graph_data")
    return p.parse_args()


def load_labels(path: Path) -> dict[tuple[int, int], str]:
    labels: dict[tuple[int, int], str] = {}
    if not path.exists():
        print(f"ERROR: labels file not found: {path}")
        return labels
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                labels[(int(obj["layer"]), int(obj["feature"]))] = obj["label"]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return labels


def patch_graph(data: dict, labels: dict[tuple[int, int], str]) -> tuple[int, int]:
    """Patch clerp fields in-place. Returns (n_patched, n_nodes)."""
    n_patched = 0
    for node in data.get("nodes", []):
        if node.get("feature_type") != "cross layer transcoder":
            continue
        key = (int(node.get("layer", -1)), int(node.get("feature", -1)))
        label = labels.get(key)
        if label:
            node["clerp"] = label
            n_patched += 1
    return n_patched, len(data.get("nodes", []))


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels))
    print(f"Loaded {len(labels)} labels from {args.labels}")

    graph_files = sorted(Path(args.graph_dir).glob("*.json"))
    if not graph_files:
        print(f"No graph JSONs found in {args.graph_dir}")
        return

    print(f"Patching {len(graph_files)} graphs in {args.graph_dir}/\n")

    total_patched = 0
    for path in graph_files:
        data = json.loads(path.read_text())
        n_patched, n_nodes = patch_graph(data, labels)
        path.write_text(json.dumps(data, indent=2))
        print(f"  {path.stem}: {n_patched}/{n_nodes} nodes patched")
        total_patched += n_patched

    print(f"\nDone. {total_patched} clerp fields updated across {len(graph_files)} graphs.")


if __name__ == "__main__":
    main()
