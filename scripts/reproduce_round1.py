#!/usr/bin/env python
"""
scripts/reproduce_round1.py

Reproduce (and slightly extend) the Round-1 circuit numbers from the graph JSONs
alone — NO GPU, NO model, NO internet. Written to verify that every number in
docs/round1_eligibility_summary.md rests on reproduced output before it goes into
the writeup.

For each model (gemma-3-4b-it in frontend/graph_data/, medgemma-4b-it in
frontend/graph_data/medgemma/) and each of the 5 Round-1 pairs, it reports:
  - per-graph: completeness proxy, feature/error node counts, early/mid/late
    feature-influence split;
  - within-pair overlap: jaccard over (layer,feature) + shared-influence share
    (the over-exclusion signature: knowledge pairs overlap HIGH, controls LOW);
  - cross-model overlap: same prompt, gemma vs medgemma (the ~85%-reorganization
    claim).

Graphs are ~100 MB each, so we load each once, keep only the compact
(layer,feature)->influence aggregate, and drop the raw JSON. Empty/truncated
files (e.g. the 0-byte gemma elig_age_pos.json) are skipped with a warning.

Usage:
    python scripts/reproduce_round1.py
    python scripts/reproduce_round1.py --json data/round1_reproduced.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Round-1 pairs and their type. (age is a control but gemma's pos graph is the
# truncated one — handled gracefully; medgemma's age pair is intact.)
PAIRS = [
    ("age", "control"),
    ("ecog", "control"),
    ("histology", "control"),
    ("priortx", "knowledge"),
    ("stage", "knowledge"),
]
MODELS = {
    "gemma-3-4b-it": "frontend/graph_data",
    "medgemma-4b-it": "frontend/graph_data/medgemma",
}
N_LAYERS = 34
BUCKETS = [("early", 0, 11), ("mid", 12, 22), ("late", 23, 33)]


def load_graph(path: Path):
    """Return (feature_agg, meta) or (None, reason) for a graph JSON.

    feature_agg: {(layer, feature): summed influence}
    meta: completeness proxy, node counts, depth split.
    """
    if not path.exists():
        return None, "missing"
    raw = path.read_text()
    if not raw.strip():
        return None, "empty/truncated"
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"malformed ({e})"

    agg: dict = collections.defaultdict(float)
    feat_inf = err_inf = 0.0
    n_feat = n_err = 0
    layer_inf: dict = collections.defaultdict(float)
    for n in d["nodes"]:
        ft = n.get("feature_type") or ""
        inf = float(n.get("influence") or 0.0)
        if ft == "cross layer transcoder":
            L = int(n["layer"])
            agg[(L, int(n["feature"]))] += inf
            feat_inf += inf
            layer_inf[L] += inf
            n_feat += 1
        elif "error" in ft:
            err_inf += inf
            n_err += 1
    comp = feat_inf / (feat_inf + err_inf) if (feat_inf + err_inf) else float("nan")
    depth = {name: (sum(layer_inf[L] for L in range(lo, hi + 1)) / feat_inf
                    if feat_inf else 0.0)
             for name, lo, hi in BUCKETS}
    meta = {"completeness": comp, "n_feat": n_feat, "n_err": n_err,
            "feat_inf": feat_inf, "err_inf": err_inf, "depth": depth}
    return dict(agg), meta


def overlap(a: dict, b: dict) -> dict:
    """Jaccard + shared-influence share between two feature aggregates."""
    sa, sb = set(a), set(b)
    shared = sa & sb
    union = sa | sb
    ia, ib = sum(a.values()), sum(b.values())
    shared_ia = sum(a[k] for k in shared)
    shared_ib = sum(b[k] for k in shared)
    return {
        "jaccard": len(shared) / len(union) if union else float("nan"),
        "n_shared": len(shared), "n_a": len(sa), "n_b": len(sb),
        "shared_inf_a": shared_ia / ia if ia else float("nan"),
        "shared_inf_b": shared_ib / ib if ib else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce Round-1 circuit numbers offline.")
    ap.add_argument("--json", default=None, help="also write results to this JSON path")
    args = ap.parse_args()

    # load everything once: graphs[model][f"{pair}_{side}"] = (agg, meta)
    graphs: dict = {}
    print("Loading graphs (each ~100 MB, one at a time)...")
    for model, base in MODELS.items():
        graphs[model] = {}
        for pair, _ in PAIRS:
            for side in ("pos", "neg"):
                key = f"{pair}_{side}"
                fp = Path(base) / f"elig_{key}.json"
                mb = fp.stat().st_size / 1e6 if fp.exists() else 0
                print(f"  loading {model}/{key} ({mb:.0f} MB)...", flush=True)
                agg, meta = load_graph(fp)
                graphs[model][key] = (agg, meta)
                if agg is None:
                    print(f"  ! {model}/{key}: {meta}", flush=True)

    out: dict = {"per_graph": {}, "within_pair": {}, "cross_model": {}}

    # ---- per-graph completeness + depth ---------------------------------- #
    print("\n" + "=" * 72)
    print("PER-GRAPH  (completeness, node counts, feature-influence depth)")
    print("=" * 72)
    for model in MODELS:
        print(f"\n[{model}]")
        print(f"  {'graph':16} {'compl':>6} {'feat':>6} {'err':>5}  "
              f"{'early':>6} {'mid':>6} {'late':>6}")
        for pair, _ in PAIRS:
            for side in ("pos", "neg"):
                key = f"{pair}_{side}"
                agg, meta = graphs[model][key]
                if agg is None:
                    print(f"  {key:16} {'--':>6}  ({meta})")
                    continue
                dp = meta["depth"]
                print(f"  {key:16} {meta['completeness']:6.3f} "
                      f"{meta['n_feat']:6d} {meta['n_err']:5d}  "
                      f"{dp['early']:6.0%} {dp['mid']:6.0%} {dp['late']:6.0%}")
                out["per_graph"][f"{model}/{key}"] = {
                    "completeness": meta["completeness"],
                    "n_feat": meta["n_feat"], "n_err": meta["n_err"],
                    "depth": meta["depth"]}

    # ---- within-pair overlap (the over-exclusion signature) -------------- #
    print("\n" + "=" * 72)
    print("WITHIN-PAIR OVERLAP  (pos vs neg; controls LOW, knowledge HIGH)")
    print("=" * 72)
    for model in MODELS:
        print(f"\n[{model}]")
        print(f"  {'pair':12} {'type':10} {'jaccard':>8} {'shared_inf':>12}")
        for pair, kind in PAIRS:
            a, _ = graphs[model][f"{pair}_pos"]
            b, _ = graphs[model][f"{pair}_neg"]
            if a is None or b is None:
                print(f"  {pair:12} {kind:10} {'--':>8}  (missing side)")
                continue
            o = overlap(a, b)
            sinf = (o["shared_inf_a"] + o["shared_inf_b"]) / 2
            print(f"  {pair:12} {kind:10} {o['jaccard']:8.3f} {sinf:11.1%}")
            out["within_pair"][f"{model}/{pair}"] = {"type": kind, **o}

    # ---- cross-model overlap (same prompt, gemma vs medgemma) ------------ #
    print("\n" + "=" * 72)
    print("CROSS-MODEL OVERLAP  (same prompt: gemma vs medgemma; low = reorganized)")
    print("=" * 72)
    ga, ma = graphs["gemma-3-4b-it"], graphs["medgemma-4b-it"]
    print(f"  {'graph':16} {'jaccard':>8} {'shared_inf':>12}")
    for pair, _ in PAIRS:
        for side in ("pos", "neg"):
            key = f"{pair}_{side}"
            a, _ = ga[key]
            b, _ = ma[key]
            if a is None or b is None:
                print(f"  {key:16} {'--':>8}  (missing)")
                continue
            o = overlap(a, b)
            sinf = (o["shared_inf_a"] + o["shared_inf_b"]) / 2
            print(f"  {key:16} {o['jaccard']:8.3f} {sinf:11.1%}")
            out["cross_model"][key] = o

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
