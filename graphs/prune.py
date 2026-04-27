"""
graphs/prune.py — Attribution graph pruning via indirect influence matrix.

Implements the pruning algorithm from Appendix: Graph Pruning in:
  "Circuit Tracing: Revealing Computational Graphs in Language Models"
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Algorithm summary:
  1. Build N×N adjacency matrix A where A[i,j] = edge weight from node i to node j.
  2. Compute indirect influence matrix B_ℓ = Σ_{k=1}^{ℓ} A^k  (truncated power series).
     This approximates (I-A)^{-1} - I = A + A² + A³ + ...
     Each additional term adds one more hop of indirect influence.
  3. Score each node by |B_ℓ[node, logit]| — its total (direct + indirect) effect on
     the target logit through all paths of length ≤ ℓ.
  4. Keep top-K nodes by score (logit node always kept).
  5. Among surviving nodes, keep top-K edges by |weight|.

Edge normalization note:
  A is built from raw edge weights, so B_ℓ[:, logit_idx] gives the sum-of-all-paths
  contribution to the logit.  This is the correct quantity for ranking nodes by
  causal importance to the output.
"""

from __future__ import annotations

import torch
from torch import Tensor

from clt.config import AttributionConfig
from graphs.build import AttributionGraph


# ---------------------------------------------------------------------------
# Main pruning function
# ---------------------------------------------------------------------------

def prune_graph(
    graph: AttributionGraph,
    cfg: AttributionConfig | None = None,
) -> AttributionGraph:
    """
    Prune an AttributionGraph to its top-K nodes and top-K edges.

    Args:
        graph: Full attribution graph from build_attribution_graph().
        cfg:   AttributionConfig controlling top_k_nodes, top_k_edges,
               and max_path_length. Uses defaults if None.

    Returns:
        A new AttributionGraph containing only the top-K nodes (by indirect
        influence on the logit) and top-K edges (by absolute weight) among
        the surviving nodes.
    """
    if cfg is None:
        cfg = AttributionConfig()

    if not graph.nodes:
        return graph

    # -----------------------------------------------------------------------
    # Step 1: Index nodes
    # -----------------------------------------------------------------------
    node_ids = [n["id"] for n in graph.nodes]
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    # Find logit node index (there should be exactly one)
    logit_indices = [i for i, n in enumerate(graph.nodes) if n["type"] == "logit"]
    if not logit_indices:
        # No logit node — return graph as-is
        return graph
    logit_idx = logit_indices[0]

    # -----------------------------------------------------------------------
    # Step 2: Build adjacency matrix A (N × N, float64 for numerical stability)
    # -----------------------------------------------------------------------
    # A[i, j] = edge weight from node i to node j
    # (float64: indirect paths multiply edge weights — use higher precision)
    A = torch.zeros(N, N, dtype=torch.float64)

    # Filter to valid edges (both endpoints in node_index)
    valid_edges = [
        e for e in graph.edges
        if e["source"] in node_index and e["target"] in node_index
    ]

    for edge in valid_edges:
        i = node_index[edge["source"]]
        j = node_index[edge["target"]]
        A[i, j] += edge["weight"]  # sum if multiple edges between same pair

    # -----------------------------------------------------------------------
    # Step 3: Compute B_ℓ = Σ_{k=1}^{ℓ} A^k  (indirect influence matrix)
    # -----------------------------------------------------------------------
    # B_ℓ[:, logit_idx] gives the total influence of each node on the logit
    # through all paths of length 1, 2, ..., ℓ.
    #
    # Iterative accumulation: B = A + A² + A³ + ...
    #   power = A^k (starts at A^1 = A)
    #   B     = Σ power
    #
    # This avoids computing the full matrix inverse and is exact up to ℓ hops.

    # (N, N) — accumulates A^1 + A^2 + ... + A^ℓ
    B = torch.zeros(N, N, dtype=torch.float64)
    # (N, N) — current power of A
    power = A.clone()

    for _ in range(cfg.max_path_length):
        B = B + power
        power = power @ A  # next power: A^{k+1} = A^k @ A

    # -----------------------------------------------------------------------
    # Step 4: Score and prune nodes
    # -----------------------------------------------------------------------
    # Score of node i = |B[i, logit_idx]| — total indirect influence on target
    # (N,)
    node_scores = B[:, logit_idx].abs()

    # Always keep the logit node itself; score it maximally so it's never pruned
    node_scores[logit_idx] = float("inf")

    # If logit-influence scores are degenerate (near-zero completeness), fall back
    # to activation magnitude for feature nodes. This happens when attention paths
    # dominate the logit and the T-matrix captures <1% of the signal.
    if node_scores.max().item() < 1e-6:
        for i, n in enumerate(graph.nodes):
            if n.get("type") == "feature":
                node_scores[i] = abs(n.get("activation", 0.0))
        node_scores[logit_idx] = float("inf")

    # top_k_nodes includes the logit node
    k_nodes = min(cfg.top_k_nodes, N)
    # (k_nodes,) — indices of top-scoring nodes
    top_node_indices = node_scores.topk(k_nodes).indices.tolist()
    surviving_node_ids = {node_ids[i] for i in top_node_indices}

    pruned_nodes = [n for n in graph.nodes if n["id"] in surviving_node_ids]

    # -----------------------------------------------------------------------
    # Step 5: Filter and prune edges
    # -----------------------------------------------------------------------
    # Keep only edges where both endpoints survived node pruning
    surviving_edges = [
        e for e in valid_edges
        if e["source"] in surviving_node_ids and e["target"] in surviving_node_ids
    ]

    # Among surviving edges, keep top-K by absolute weight
    k_edges = min(cfg.top_k_edges, len(surviving_edges))
    if surviving_edges:
        surviving_edges.sort(key=lambda e: abs(e["weight"]), reverse=True)
        pruned_edges = surviving_edges[:k_edges]
    else:
        pruned_edges = []

    # -----------------------------------------------------------------------
    # Step 6: Build pruned graph
    # -----------------------------------------------------------------------
    pruned = AttributionGraph(
        nodes=pruned_nodes,
        edges=pruned_edges,
        tokens=graph.tokens,
        target_token=graph.target_token,
        target_position=graph.target_position,
        logit_value=graph.logit_value,
        completeness=graph.completeness,
    )

    return pruned


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def node_influence_scores(
    graph: AttributionGraph,
    cfg: AttributionConfig | None = None,
) -> list[tuple[str, float]]:
    """
    Return a sorted list of (node_id, indirect_influence_score) for all nodes.

    Useful for inspecting how much each node contributes to the logit before pruning.

    Args:
        graph: Full attribution graph.
        cfg:   AttributionConfig controlling max_path_length.

    Returns:
        List of (node_id, score) tuples sorted by score descending.
    """
    if cfg is None:
        cfg = AttributionConfig()

    if not graph.nodes:
        return []

    node_ids = [n["id"] for n in graph.nodes]
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    logit_indices = [i for i, n in enumerate(graph.nodes) if n["type"] == "logit"]
    if not logit_indices:
        return []
    logit_idx = logit_indices[0]

    A = torch.zeros(N, N, dtype=torch.float64)
    for edge in graph.edges:
        if edge["source"] in node_index and edge["target"] in node_index:
            i = node_index[edge["source"]]
            j = node_index[edge["target"]]
            A[i, j] += edge["weight"]

    B = torch.zeros(N, N, dtype=torch.float64)
    power = A.clone()
    for _ in range(cfg.max_path_length):
        B = B + power
        power = power @ A

    scores = B[:, logit_idx].abs().tolist()
    ranked = sorted(zip(node_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked
