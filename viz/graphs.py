"""
viz/graphs.py — Attribution graph visualizations

Functions for summarizing and plotting attribution graphs in a way
readable by non-technical audiences.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
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


def plot_transfer_norms(
    transfer: "dict[tuple[int, int], Any]",
    n_features: int,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Bar charts of per-feature transfer vector norms for each (source, target) layer pair.

    One subplot per (l_s, l_t) pair. Shows which features have the strongest cross-layer reach.

    Args:
        transfer:    dict mapping (l_source, l_target) → Tensor of shape (n_features, d_model)
                     as returned by graphs.build._compute_transfer_matrices()
        n_features:  number of CLT features (x-axis range)
        title:       optional suptitle override
    """
    n_pairs = len(transfer)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 3.5))
    if n_pairs == 1:
        axes = [axes]

    for ax, (key, T) in zip(axes, sorted(transfer.items())):
        l_s, l_t = key
        feature_norms = T.cpu().float().norm(dim=1).numpy()
        ax.bar(range(n_features), feature_norms, color="#4C72B0", alpha=0.8, width=1.0)
        ax.set_title(f"T[{l_s}→{l_t}]  per-feature norm", fontsize=9)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("‖transfer vector‖")

    fig.suptitle(
        title or "Transfer matrix norms — cross-layer feature reach",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_influence_scores(
    scores: "list[tuple[Any, float]]",
    top_k_nodes: int,
    node_colors: Optional[dict[str, str]] = None,
    node_type_map: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel figure: histogram of all indirect influence scores + bar chart of top-K nodes.

    Args:
        scores:        list of (node_id, score) pairs from graphs.prune.node_influence_scores()
        top_k_nodes:   cutoff line and top-K selection
        node_colors:   dict mapping node type → color (e.g. {'feature': '#4C72B0', ...})
        node_type_map: dict mapping node_id → type string (used for color coding)
        title:         optional suptitle override
    """
    node_colors = node_colors or {
        "feature":   "#4C72B0",
        "embedding": "#55A868",
        "error":     "#C44E52",
        "logit":     "#8172B2",
    }
    node_type_map = node_type_map or {}

    finite_scores = [(nid, s) for nid, s in scores if s < float("inf")]
    all_score_vals = [s for _, s in finite_scores]
    cutoff_val = sorted(all_score_vals, reverse=True)[
        min(top_k_nodes - 1, len(all_score_vals) - 1)
    ] if all_score_vals else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: histogram
    ax = axes[0]
    ax.hist(all_score_vals, bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(cutoff_val, color="red", linestyle="--",
               label=f"top-{top_k_nodes} cutoff")
    ax.set_xlabel("Indirect influence score")
    ax.set_ylabel("Node count")
    ax.set_title("Influence score distribution", fontsize=10)
    ax.legend(fontsize=8)

    # Right: top-K bar chart
    top_nodes = sorted(finite_scores, key=lambda x: x[1], reverse=True)[:top_k_nodes]
    labels = [nid for nid, _ in top_nodes]
    vals   = [s   for _,   s in top_nodes]
    colors = [node_colors.get(node_type_map.get(nid, "feature"), "#888888") for nid in labels]

    ax = axes[1]
    ax.barh(range(len(labels)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Influence score")
    ax.set_title(f"Top {top_k_nodes} nodes by influence", fontsize=10)

    legend_patches = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in node_colors.items()
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.suptitle(title or "Node influence scores", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_completeness_waterfall(
    contribution_by_type: dict[str, float],
    logit_value: float,
    target_token: str,
    node_colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of contributions to the target token logit by node type.

    Shows how much each type (features, embeddings, errors) explains the logit,
    and draws a vertical line at the actual logit value for a completeness read.

    Args:
        contribution_by_type: dict mapping type string → sum of contributions
        logit_value:          actual logit value for the target token
        target_token:         token being traced (for title)
        node_colors:          optional color mapping by type
        title:                optional suptitle override
    """
    node_colors = node_colors or {
        "feature":   "#4C72B0",
        "embedding": "#55A868",
        "error":     "#C44E52",
    }

    types_sorted = sorted(contribution_by_type.items(), key=lambda x: x[1])
    labels = [t for t, _ in types_sorted] + ["Total"]
    values = [v for _, v in types_sorted] + [sum(contribution_by_type.values())]
    colors = [
        "#555555" if l == "Total" else node_colors.get(l, "#999999")
        for l in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(logit_value, color="red", linestyle="--", linewidth=1.2,
               label=f"Actual logit ({logit_value:.3f})")
    ax.set_xlabel(f'Contribution to logit of "{target_token}"')
    ax.legend(fontsize=9)
    fig.suptitle(
        title or f'Completeness waterfall — "{target_token}"',
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
