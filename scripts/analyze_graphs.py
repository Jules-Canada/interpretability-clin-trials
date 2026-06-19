#!/usr/bin/env python
"""
scripts/analyze_graphs.py

Per-graph structural summary for circuit-tracer frontend JSONs. No feature labels
yet (clerp empty), so this reports: the answer (top logit), completeness proxy
(feature vs mlp-error influence), feature/error node counts, the layer profile of
feature influence (where the decision computation sits), and the top features.

Usage:
    python scripts/analyze_graphs.py                # all frontend/graph_data/elig_*.json
    python scripts/analyze_graphs.py elig_priortx_pos
"""

import collections
import glob
import json
import sys
from pathlib import Path

N_LAYERS = 34  # gemma-3-4b
BUCKETS = [("early L0-11", 0, 11), ("mid L12-22", 12, 22), ("late L23-33", 23, 33)]


def summarize(path: str) -> None:
    d = json.loads(Path(path).read_text())
    nodes = d["nodes"]
    feats = [n for n in nodes if n.get("feature_type") == "cross layer transcoder"]
    errs = [n for n in nodes if "error" in (n.get("feature_type") or "")]
    logits = [n for n in nodes if n.get("feature_type") == "logit"]

    feat_inf = sum(n.get("influence", 0.0) for n in feats)
    err_inf = sum(n.get("influence", 0.0) for n in errs)
    comp = feat_inf / (feat_inf + err_inf) if (feat_inf + err_inf) else float("nan")

    # answer = highest-prob logit node
    ans = max(logits, key=lambda n: n.get("token_prob", 0.0), default=None)
    ans_s = f"{ans.get('clerp') or ans.get('jsNodeId')!r} p={ans.get('token_prob', 0):.2f}" \
        if ans else "n/a"

    # layer profile of feature influence
    layer_inf = collections.defaultdict(float)
    agg = collections.defaultdict(float)
    for n in feats:
        L = int(n["layer"])
        layer_inf[L] += n.get("influence", 0.0)
        agg[(L, int(n["feature"]))] += n.get("influence", 0.0)
    buckets = {name: sum(layer_inf[L] for L in range(lo, hi + 1)) / feat_inf
               for name, lo, hi in BUCKETS} if feat_inf else {}
    top = sorted(agg.items(), key=lambda x: -x[1])[:5]

    print(f"=== {Path(path).stem} ===")
    print(f"  answer: {ans_s}")
    print(f"  completeness: {comp:.3f}   (feat_inf {feat_inf:.0f} / err_inf {err_inf:.0f})")
    print(f"  nodes: {len(feats)} feature, {len(errs)} error, {len(logits)} logit")
    print(f"  feature influence by depth: " +
          "  ".join(f"{k} {v:.0%}" for k, v in buckets.items()))
    print("  top features: " +
          "; ".join(f"L{L}f{f} {inf:.2f}" for (L, f), inf in top))


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = [a if a.endswith(".json") else f"frontend/graph_data/{a}.json" for a in args]
    else:
        paths = sorted(glob.glob("frontend/graph_data/elig_*.json"))
    for p in paths:
        summarize(p)


if __name__ == "__main__":
    main()
