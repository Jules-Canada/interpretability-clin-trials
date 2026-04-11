"""
graphs/export.py — Serialize an AttributionGraph to the frontend JSON schema.

The frontend (anthropics/attribution-graphs-frontend) loads graph data from
`/graph_data/<slug>.json` and expects this structure:

{
  "metadata": {
    "prompt_tokens": ["token1", " token2", ...],  // string tokens (len = seq_len)
    "scan": "model_name",                          // e.g. "pythia-410m"
  },
  "nodes": [
    {
      "node_id":            "feat_l0_p3_f12",     // unique string, matched by links
      "jsNodeId":           "feat_l0_p3_f12",     // same as node_id
      "layer":              0,                     // int layer, or "E" for embedding
      "feature":            12,                    // int feature index
      "feature_type":       "cross layer transcoder",
        // one of: "cross layer transcoder" | "embedding"
        //         | "mlp reconstruction error" | "logit"
      "ctx_idx":            3,                     // sequence position (0-indexed)
      "probe_location_idx": 3,                     // same as ctx_idx
      "clerp":              "L0F12@3",             // human-readable label / description
      "vis_link":           "",                    // URL to feature vis (empty if unknown)
      "isLogit":            false,                 // true only for the logit node
    },
    ...
  ],
  "links": [
    {
      "source": "feat_l0_p3_f12",   // node_id of source
      "target": "feat_l1_p3_f7",    // node_id of target
      "weight": 0.034               // float edge weight
    },
    ...
  ]
}

Logit node clerp format: '"token" k(p=0.95)' so the frontend can extract the
probability display value (the `(p=...)` part after `k`).

See frontend/attribution_graph/util-cg.js → formatData() for the full field usage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from graphs.build import AttributionGraph


# ---------------------------------------------------------------------------
# Internal type mapping
# ---------------------------------------------------------------------------

_FEATURE_TYPE_MAP = {
    "feature":   "cross layer transcoder",
    "embedding": "embedding",
    "error":     "mlp reconstruction error",
    "logit":     "logit",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_frontend_json(
    graph: AttributionGraph,
    model_name: str = "pythia-410m",
    logit_probability: float | None = None,
) -> dict:
    """
    Convert an AttributionGraph to the frontend JSON dict.

    Args:
        graph:             Pruned (or full) AttributionGraph from build/prune.
        model_name:        Model identifier written to metadata.scan.
        logit_probability: Softmax probability of the target token (optional).
                           If provided, included in the logit node's clerp string.

    Returns:
        dict ready to be written as JSON.
    """
    nodes_out = []

    for node in graph.nodes:
        node_id = node["id"]
        node_type = node["type"]
        feature_type = _FEATURE_TYPE_MAP.get(node_type, "cross layer transcoder")

        # Layer field: embedding nodes use "E"; all others use int layer
        if node_type == "embedding":
            layer_field = "E"
        else:
            layer_field = node["layer"]  # int or None (logit) — frontend handles None

        # Feature index: embedding/logit nodes get 0 as a placeholder
        feature_field = node["feature"] if node["feature"] is not None else 0

        # Sequence position (ctx_idx)
        ctx_idx = node["position"] if node["position"] is not None else 0

        # Clerp (human label) — logit node gets special "(p=...)" format
        if node_type == "logit":
            if logit_probability is not None and math.isfinite(logit_probability):
                clerp = f'"{graph.target_token}" k(p={logit_probability:.4f})'
            else:
                clerp = f'"{graph.target_token}" k(p=?)'
        else:
            clerp = node.get("label", "")

        frontend_node = {
            "node_id":            node_id,
            "jsNodeId":           node_id,
            "layer":              layer_field,
            "feature":            feature_field,
            "feature_type":       feature_type,
            "ctx_idx":            ctx_idx,
            "probe_location_idx": ctx_idx,
            "clerp":              clerp,
            "vis_link":           "",    # no feature visualiser yet
            "isLogit":            node_type == "logit",
        }
        nodes_out.append(frontend_node)

    links_out = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "weight": edge["weight"],
        }
        for edge in graph.edges
    ]

    data = {
        "metadata": {
            "prompt_tokens": graph.tokens,
            "scan": model_name,
        },
        "nodes": nodes_out,
        "links": links_out,
    }

    return data


def save_graph(
    graph: AttributionGraph,
    output_path: str | Path,
    model_name: str = "pythia-410m",
    logit_probability: float | None = None,
) -> Path:
    """
    Write an AttributionGraph to a JSON file readable by the frontend.

    The frontend expects files under `graph_data/<slug>.json` relative to its
    root.  This function writes to output_path directly — place the file in
    the right directory for your frontend serving setup.

    Args:
        graph:             AttributionGraph to export.
        output_path:       Destination file path (e.g. "frontend/graph_data/trial_001.json").
        model_name:        Written to metadata.scan.
        logit_probability: Softmax probability of target token (optional).

    Returns:
        Resolved Path of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = to_frontend_json(graph, model_name=model_name, logit_probability=logit_probability)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out.resolve()
