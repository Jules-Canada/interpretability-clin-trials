"""
viz/graphs.py — Attribution graph visualizations

Functions for summarizing and plotting attribution graphs in a way
readable by non-technical audiences.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def plot_node_contributions(
    nodes: list[dict[str, Any]],
    target_token: str,
    topk: int = 15,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart showing which features contribute most to the target token logit.

    Args:
        nodes:        list of node dicts from the attribution graph. Expected keys:
                        'label' (str), 'contribution' (float), 'type' (str)
        target_token: the token being traced (e.g. "eligible", "yes")
        topk:         number of top contributors to show
        title:        optional figure title override

    Node types are color-coded: CLT features (blue), token embeddings (green),
    reconstruction errors (orange).
    """
    type_colors = {
        "feature": "#4C72B0",
        "embedding": "#55A868",
        "error": "#C44E52",
    }

    sorted_nodes = sorted(nodes, key=lambda n: abs(n["contribution"]), reverse=True)[:topk]
    labels = [n["label"] for n in sorted_nodes]
    values = [n["contribution"] for n in sorted_nodes]
    colors = [type_colors.get(n.get("type", "feature"), "#888888") for n in sorted_nodes]

    fig, ax = plt.subplots(figsize=(9, max(4, topk * 0.4)))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f'Contribution to logit of "{target_token}"')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.capitalize()) for t, c in type_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.suptitle(
        title or f'What drives the model toward "{target_token}"?',
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def summarize_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    topk_nodes: int = 10,
    topk_edges: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two DataFrames (top nodes, top edges) suitable for display in a notebook
    or export to a slide.

    Args:
        nodes:      list of node dicts with keys: 'label', 'contribution', 'type', 'layer'
        edges:      list of edge dicts with keys: 'source', 'target', 'weight'
        topk_nodes: number of top nodes to include by |contribution|
        topk_edges: number of top edges to include by |weight|

    Returns:
        node_df: DataFrame with columns [Rank, Feature, Type, Layer, Contribution]
        edge_df: DataFrame with columns [Rank, From, To, Weight]
    """
    sorted_nodes = sorted(nodes, key=lambda n: abs(n["contribution"]), reverse=True)[:topk_nodes]
    node_df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Feature": n["label"],
            "Type": n.get("type", "feature").capitalize(),
            "Layer": n.get("layer", "—"),
            "Contribution": f"{n['contribution']:+.4f}",
        }
        for i, n in enumerate(sorted_nodes)
    ])

    sorted_edges = sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)[:topk_edges]
    edge_df = pd.DataFrame([
        {
            "Rank": i + 1,
            "From": e["source"],
            "To": e["target"],
            "Weight": f"{e['weight']:+.4f}",
        }
        for i, e in enumerate(sorted_edges)
    ])

    return node_df, edge_df


def plot_layer_flow(
    nodes: list[dict[str, Any]],
    n_layers: int,
    target_token: str,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart showing total absolute contribution to the target token, grouped by CLT layer.

    Gives a non-technical audience a sense of *which layers* matter most for a decision.

    Args:
        nodes:       list of node dicts with keys 'layer' (int), 'contribution' (float)
        n_layers:    total number of CLT layers
        target_token: token being traced
    """
    layer_totals = {l: 0.0 for l in range(n_layers)}
    for n in nodes:
        layer = n.get("layer")
        if isinstance(layer, int):
            layer_totals[layer] = layer_totals.get(layer, 0.0) + abs(n["contribution"])

    layers = list(range(n_layers))
    totals = [layer_totals[l] for l in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"Layer {l}" for l in layers], totals, color="#4C72B0")
    ax.set_ylabel("Total feature contribution (absolute)")
    ax.set_xlabel("CLT layer")
    fig.suptitle(
        title or f'Which layers drive the prediction of "{target_token}"?',
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
