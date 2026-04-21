#!/usr/bin/env python
"""
scripts/collect_graph_features.py

Scan all attribution graph JSONs in frontend/graph_data/ and collect the
union of (layer, feature) pairs that appear across them. Writes the result
to data/graph_features.json for use as --features_file in find_top_activations.py.

This means feature labeling only processes features that actually appear in
your graphs — typically 200-500 features rather than all 2048×24 = 49,152.

Usage:
    python scripts/collect_graph_features.py
    python scripts/collect_graph_features.py --graph_dir frontend/graph_data --output data/graph_features.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect unique (layer, feature) pairs from graph JSONs.")
    p.add_argument("--graph_dir", type=str, default="frontend/graph_data",
                   help="Directory containing graph JSON files")
    p.add_argument("--output",    type=str, default="data/graph_features.json",
                   help="Output JSON file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)

    graph_files = sorted(graph_dir.glob("*.json"))
    if not graph_files:
        print(f"No JSON files found in {graph_dir}")
        sys.exit(1)

    print(f"Scanning {len(graph_files)} graph files in {graph_dir}/\n")

    features: set[tuple[int, int]] = set()
    per_graph: dict[str, int] = {}

    for path in graph_files:
        data = json.loads(path.read_text())
        slug = path.stem
        count_before = len(features)

        for node in data.get("nodes", []):
            if node.get("feature_type") == "cross layer transcoder":
                layer   = node.get("layer")
                feature = node.get("feature")
                if layer is not None and feature is not None:
                    features.add((int(layer), int(feature)))

        count_after = len(features)
        per_graph[slug] = count_after - count_before
        print(f"  {slug}: {per_graph[slug]} new features  (running total: {count_after})")

    feature_list = sorted(features)

    out = {
        "n_features": len(feature_list),
        "source_graphs": [p.stem for p in graph_files],
        "features": feature_list,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nTotal unique features: {len(feature_list)}")
    print(f"Layers represented: {sorted(set(l for l, _ in feature_list))}")
    print(f"Written to: {out_path}")
    print(f"\nNext step:")
    print(f"  python scripts/find_top_activations.py \\")
    print(f"      --checkpoint <ckpt> --activation_path data/activations/pythia-410m.h5 \\")
    print(f"      --n_layers 24 --d_model 1024 --d_mlp 4096 --n_features 2048 \\")
    print(f"      --features_file {out_path}")


if __name__ == "__main__":
    main()
