#!/usr/bin/env python
"""
scripts/compare_graphs.py

Compare the feature content of two circuit-tracer attribution graphs (frontend
JSON). Core question for the eligibility work: when two prompts produce the SAME
answer, do they run through the SAME dictionary features or DIFFERENT ones?
e.g. pemetrexed->No vs pembrolizumab->No — over-exclusion (shared "prior therapy"
circuit, no discrimination) vs genuine drug-class discrimination.

Feature identity = (layer, feature index), influence summed over token positions.
No labels yet (clerp empty until feature labeling), so we read overlap structure
and the LAYER of unique features (late-layer unique = a discriminating concept).

Usage:
    python scripts/compare_graphs.py elig_priortx_pos elig_priortx_neg
    python scripts/compare_graphs.py path/to/a.json path/to/b.json
"""

import collections
import json
import sys
from pathlib import Path


def load_features(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    agg: dict = collections.defaultdict(float)
    for n in d["nodes"]:
        if n.get("feature_type") == "cross layer transcoder":
            agg[(int(n["layer"]), int(n["feature"]))] += float(n.get("influence", 0.0))
    return agg


def resolve(arg: str) -> str:
    return arg if arg.endswith(".json") else f"frontend/graph_data/{arg}.json"


def main() -> None:
    pa, pb = resolve(sys.argv[1]), resolve(sys.argv[2])
    a, b = load_features(pa), load_features(pb)
    sa, sb = set(a), set(b)
    shared, ua, ub = sa & sb, sa - sb, sb - sa
    ia, ib = sum(a.values()), sum(b.values())
    shared_ia = sum(a[k] for k in shared)
    shared_ib = sum(b[k] for k in shared)

    print(f"A = {Path(pa).stem}:  {len(sa)} features, total influence {ia:.2f}")
    print(f"B = {Path(pb).stem}:  {len(sb)} features, total influence {ib:.2f}")
    print(f"shared (layer,feat): {len(shared)}   jaccard {len(shared)/len(sa | sb):.3f}")
    print(f"shared-feature influence:  A {shared_ia/ia:.1%}   B {shared_ib/ib:.1%}")

    def top(agg, keys, n=12):
        return sorted(keys, key=lambda k: -agg[k])[:n]

    print(f"\nTop A-only features (L=layer, f=feat: influence):")
    for k in top(a, ua):
        print(f"  L{k[0]:>2} f{k[1]:<6} {a[k]:.3f}")
    print(f"\nTop B-only features:")
    for k in top(b, ub):
        print(f"  L{k[0]:>2} f{k[1]:<6} {b[k]:.3f}")
    print(f"\nTop shared features:")
    for k in top(a, shared):
        print(f"  L{k[0]:>2} f{k[1]:<6}  A {a[k]:.3f}  B {b[k]:.3f}")


if __name__ == "__main__":
    main()
