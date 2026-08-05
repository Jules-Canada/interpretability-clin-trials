#!/usr/bin/env python3
"""
Compare feature layer distribution across CLT variants.

Reads graph JSONs for the 7 Track B prompts × 3 CLT conditions:
  - baseline (L0~91, protocols, 1e-2)
  - pubmed_1e2 (L0~42, PubMed, 1e-2)
  - pubmed_5e2 (L0~16, PubMed, 5e-2)

Reports: layer histogram, % L0 features, deepest feature layer, completeness.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

GRAPH_DIR = Path(__file__).parent.parent / "frontend" / "graph_data"

PROMPTS = [
    "pharm_warfarin_moa",
    "pharm_metformin",
    "path_insulin_organ",
    "path_mi_artery",
    "path_dvt_complication",
    "anat_csf_production",
    "anat_bile_storage",
]

CONDITIONS = {
    "baseline (L0~91)": "",
    "pubmed_1e2 (L0~42)": "_pubmed_1e2",
    "pubmed_5e2 (L0~16)": "_pubmed_5e2",
}


def load_graph(slug: str) -> dict | None:
    path = GRAPH_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def analyze_graph(graph: dict) -> dict:
    feature_nodes = [
        n for n in graph["nodes"]
        if n["feature_type"] == "cross layer transcoder" and n.get("layer") is not None
    ]
    layers = [n["layer"] for n in feature_nodes]
    n_total = len(layers)
    layer_counts = Counter(layers)

    n_l0 = layer_counts.get(0, 0)
    max_layer = max(layers) if layers else -1
    n_deep = sum(c for l, c in layer_counts.items() if l >= 5)

    completeness = graph.get("metadata", {}).get("completeness")
    if completeness is None:
        completeness = graph.get("qParams", {}).get("completeness")

    return {
        "n_features": n_total,
        "n_l0": n_l0,
        "pct_l0": 100 * n_l0 / n_total if n_total else 0,
        "n_deep": n_deep,
        "pct_deep": 100 * n_deep / n_total if n_total else 0,
        "max_layer": max_layer,
        "layer_counts": layer_counts,
        "completeness": completeness,
    }


def main():
    print("=" * 80)
    print("Layer Distribution Comparison: Baseline vs PubMed CLTs")
    print("=" * 80)

    for cond_name, suffix in CONDITIONS.items():
        print(f"\n{'─' * 80}")
        print(f"  {cond_name}")
        print(f"{'─' * 80}")

        all_layers = Counter()
        total_features = 0
        total_l0 = 0
        total_deep = 0
        completeness_vals = []

        for prompt in PROMPTS:
            slug = f"{prompt}{suffix}"
            graph = load_graph(slug)
            if graph is None:
                print(f"  {prompt:<30s}  MISSING")
                continue

            stats = analyze_graph(graph)
            total_features += stats["n_features"]
            total_l0 += stats["n_l0"]
            total_deep += stats["n_deep"]
            all_layers += stats["layer_counts"]
            if stats["completeness"] is not None:
                completeness_vals.append(stats["completeness"])

            print(
                f"  {prompt:<30s}  "
                f"feats={stats['n_features']:3d}  "
                f"L0={stats['n_l0']:2d} ({stats['pct_l0']:4.0f}%)  "
                f"L5+={stats['n_deep']:2d} ({stats['pct_deep']:4.0f}%)  "
                f"max_L={stats['max_layer']:2d}  "
                f"comp={stats['completeness']:.3f}" if stats['completeness'] else ""
            )

        if total_features == 0:
            print("  No graphs found for this condition.")
            continue

        print()
        print(f"  TOTALS:  {total_features} features")
        print(f"    L0:    {total_l0} ({100*total_l0/total_features:.0f}%)")
        print(f"    L5+:   {total_deep} ({100*total_deep/total_features:.0f}%)")
        if completeness_vals:
            print(f"    Completeness: {min(completeness_vals):.3f} – {max(completeness_vals):.3f} "
                  f"(mean {sum(completeness_vals)/len(completeness_vals):.3f})")

        print()
        print("  Layer histogram:")
        max_count = max(all_layers.values()) if all_layers else 1
        for layer in sorted(all_layers):
            count = all_layers[layer]
            bar = "█" * int(40 * count / max_count)
            print(f"    L{layer:<3d} {count:3d} {bar}")

    print()
    print("=" * 80)
    print("Key question: does L5+ % increase with sparser CLT / PubMed corpus?")
    print("=" * 80)


if __name__ == "__main__":
    main()
