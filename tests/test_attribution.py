"""
tests/test_attribution.py — Unit tests for attribution graph construction and pruning.

Uses the same 2-layer toy transformer from test_clt_toy.py.
Tests verify:
  - transfer matrix shapes and self-consistency
  - readout vector shape and gradient correctness
  - full graph structure (node types, edge types, completeness)
  - pruning: node/edge count, logit node always survives
  - indirect influence matrix B is non-negative for pure positive graphs
"""

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from clt.config import AttributionConfig, CLTConfig
from clt.model import CrossLayerTranscoder
from graphs.build import (
    AttributionGraph,
    _compute_readout_vector,
    _compute_transfer_matrices,
    build_attribution_graph,
)
from graphs.prune import node_influence_scores, prune_graph


# ---------------------------------------------------------------------------
# Toy model constants
# ---------------------------------------------------------------------------

N_LAYERS = 2
D_MODEL = 64
D_MLP = 256
N_FEATURES = 32     # small for fast tests
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
D_VOCAB = 100
SEQ = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_transformer() -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        n_ctx=SEQ,
        act_fn="gelu",
        normalization_type="LN",
        d_vocab=D_VOCAB,
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def toy_clt(toy_transformer: HookedTransformer) -> CrossLayerTranscoder:
    clt_cfg = CLTConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_mlp=D_MLP,
        n_features=N_FEATURES,
    )
    clt = CrossLayerTranscoder(clt_cfg)
    # Use random (non-zero) decoder weights so edges are non-trivial
    for source_decoders in clt.decoders:
        for decoder in source_decoders:
            torch.nn.init.normal_(decoder.weight, std=0.01)
    # CLT must live on the same device as the model it's paired with
    device = next(toy_transformer.parameters()).device
    clt = clt.to(device)
    clt.eval()
    return clt


@pytest.fixture(scope="module")
def toy_tokens() -> torch.Tensor:
    """(1, SEQ) integer token tensor."""
    return torch.randint(0, D_VOCAB, (1, SEQ))


@pytest.fixture(scope="module")
def toy_cache(toy_transformer, toy_tokens):
    """Full activation cache from one forward pass."""
    with torch.no_grad():
        _, cache = toy_transformer.run_with_cache(toy_tokens)
    return cache


# ---------------------------------------------------------------------------
# Transfer matrix tests
# ---------------------------------------------------------------------------


def test_transfer_matrix_shapes(toy_clt, toy_transformer):
    L = N_LAYERS
    transfer = _compute_transfer_matrices(toy_clt, toy_transformer, L)

    # Should have entries for all (l_s, l_t) where l_t > l_s
    # and l_t = 1, ..., L
    for l_s in range(L):
        for l_t in range(l_s + 1, L + 1):
            key = (l_s, l_t)
            assert key in transfer, f"Missing transfer key {key}"
            T = transfer[key]
            # (n_features, d_model)
            assert T.shape == (N_FEATURES, D_MODEL), (
                f"Transfer {key} shape {T.shape} != ({N_FEATURES}, {D_MODEL})"
            )


def test_transfer_matrix_accumulates(toy_clt, toy_transformer):
    """
    T[l_s → l_t+1] should equal T[l_s → l_t] + W_dec[l_s→l_t].T @ W_out_l_t.
    Verifies the cumulative structure of the loop in _compute_transfer_matrices.
    """
    L = N_LAYERS
    transfer = _compute_transfer_matrices(toy_clt, toy_transformer, L)

    for l_s in range(L - 1):
        T_early = transfer[(l_s, l_s + 1)]   # (F, d_model)
        T_late  = transfer[(l_s, l_s + 2)]   # (F, d_model)

        # The difference should be W_dec[l_s → l_s+1].T @ W_out_{l_s+1}
        decoder = toy_clt.decoders[l_s][1]   # offset 1 → l_intermediate = l_s+1
        W_dec_T = decoder.weight.T            # (F, d_mlp)
        W_out = toy_transformer.blocks[l_s + 1].mlp.W_out  # (d_mlp, d_model)
        expected_increment = W_dec_T @ W_out  # (F, d_model)

        diff = T_late - T_early
        assert torch.allclose(diff, expected_increment, atol=1e-5), (
            f"Transfer matrix accumulation wrong for l_s={l_s}"
        )


# ---------------------------------------------------------------------------
# Readout vector tests
# ---------------------------------------------------------------------------


def test_readout_vector_shape(toy_transformer, toy_cache, toy_tokens):
    target_token_idx = 42
    v = _compute_readout_vector(
        toy_transformer, toy_cache, target_position=-1,
        target_token_idx=target_token_idx, L=N_LAYERS,
    )
    assert v.shape == (D_MODEL,), f"Readout vector shape {v.shape} != ({D_MODEL},)"


def test_readout_vector_nonzero(toy_transformer, toy_cache, toy_tokens):
    """Gradient of logit w.r.t. residual stream should be non-zero."""
    target_token_idx = 42
    v = _compute_readout_vector(
        toy_transformer, toy_cache, target_position=-1,
        target_token_idx=target_token_idx, L=N_LAYERS,
    )
    assert v.abs().max().item() > 1e-8, "Readout vector is all zeros"


def test_readout_vector_varies_by_token(toy_transformer, toy_cache):
    """Different target tokens → different readout vectors."""
    v1 = _compute_readout_vector(toy_transformer, toy_cache, -1, 10, N_LAYERS)
    v2 = _compute_readout_vector(toy_transformer, toy_cache, -1, 20, N_LAYERS)
    assert not torch.allclose(v1, v2, atol=1e-6), (
        "Readout vectors for different tokens should differ"
    )


# ---------------------------------------------------------------------------
# Full graph construction tests
# ---------------------------------------------------------------------------


def test_graph_has_logit_node(toy_transformer, toy_clt, toy_tokens):
    cfg = AttributionConfig(min_activation=0.0)  # include all features
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, cfg)

    logit_nodes = [n for n in graph.nodes if n["type"] == "logit"]
    assert len(logit_nodes) == 1, f"Expected 1 logit node, got {len(logit_nodes)}"


def test_graph_has_embedding_nodes(toy_transformer, toy_clt, toy_tokens):
    cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, cfg)

    embed_nodes = [n for n in graph.nodes if n["type"] == "embedding"]
    # One embedding node per sequence position
    assert len(embed_nodes) == SEQ, (
        f"Expected {SEQ} embedding nodes, got {len(embed_nodes)}"
    )


def test_graph_has_error_nodes(toy_transformer, toy_clt, toy_tokens):
    cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, cfg)

    error_nodes = [n for n in graph.nodes if n["type"] == "error"]
    assert len(error_nodes) == N_LAYERS, (
        f"Expected {N_LAYERS} error nodes, got {len(error_nodes)}"
    )


def test_graph_all_edges_have_valid_endpoints(toy_transformer, toy_clt, toy_tokens):
    cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, cfg)

    node_ids = {n["id"] for n in graph.nodes}
    for edge in graph.edges:
        assert edge["source"] in node_ids, f"Edge source {edge['source']} not in nodes"
        assert edge["target"] in node_ids, f"Edge target {edge['target']} not in nodes"


def test_graph_logit_value_matches_model(toy_transformer, toy_clt, toy_tokens):
    """The stored logit_value should match the model's actual output."""
    cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, cfg)

    with torch.no_grad():
        logits = toy_transformer(toy_tokens)
    expected = logits[0, -1, 42].item()
    assert abs(graph.logit_value - expected) < 1e-4, (
        f"Graph logit_value {graph.logit_value} != model logit {expected}"
    )


def test_graph_completeness_finite(toy_transformer, toy_clt, toy_tokens):
    """Completeness should be a finite number (not nan/inf)."""
    cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, cfg)
    import math
    assert math.isfinite(graph.completeness), (
        f"Graph completeness is not finite: {graph.completeness}"
    )


def test_graph_unique_node_ids(toy_transformer, toy_clt, toy_tokens):
    cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, cfg)

    ids = [n["id"] for n in graph.nodes]
    assert len(ids) == len(set(ids)), "Duplicate node IDs in graph"


# ---------------------------------------------------------------------------
# Pruning tests
# ---------------------------------------------------------------------------


def test_prune_respects_top_k_nodes(toy_transformer, toy_clt, toy_tokens):
    full_cfg = AttributionConfig(min_activation=0.0, top_k_nodes=5, top_k_edges=100)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, full_cfg)

    pruned = prune_graph(graph, full_cfg)
    assert len(pruned.nodes) <= 5, (
        f"Pruned graph has {len(pruned.nodes)} nodes, expected ≤ 5"
    )


def test_prune_respects_top_k_edges(toy_transformer, toy_clt, toy_tokens):
    full_cfg = AttributionConfig(min_activation=0.0, top_k_nodes=100, top_k_edges=3)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, full_cfg)

    pruned = prune_graph(graph, full_cfg)
    assert len(pruned.edges) <= 3, (
        f"Pruned graph has {len(pruned.edges)} edges, expected ≤ 3"
    )


def test_prune_keeps_logit_node(toy_transformer, toy_clt, toy_tokens):
    """Logit node must survive pruning regardless of top_k_nodes."""
    full_cfg = AttributionConfig(min_activation=0.0, top_k_nodes=1, top_k_edges=100)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, full_cfg)

    pruned = prune_graph(graph, full_cfg)
    logit_nodes = [n for n in pruned.nodes if n["type"] == "logit"]
    assert len(logit_nodes) == 1, "Logit node was removed by pruning"


def test_prune_edges_only_between_survivors(toy_transformer, toy_clt, toy_tokens):
    """All edges in pruned graph must connect surviving nodes."""
    full_cfg = AttributionConfig(min_activation=0.0, top_k_nodes=5, top_k_edges=20)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, full_cfg)

    pruned = prune_graph(graph, full_cfg)
    surviving_ids = {n["id"] for n in pruned.nodes}
    for edge in pruned.edges:
        assert edge["source"] in surviving_ids, (
            f"Pruned edge source {edge['source']} not in surviving nodes"
        )
        assert edge["target"] in surviving_ids, (
            f"Pruned edge target {edge['target']} not in surviving nodes"
        )


def test_prune_preserves_metadata(toy_transformer, toy_clt, toy_tokens):
    """Pruned graph should keep tokens, target_token, logit_value, completeness."""
    full_cfg = AttributionConfig(min_activation=0.0, top_k_nodes=5, top_k_edges=10)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, full_cfg)
    pruned = prune_graph(graph, full_cfg)

    assert pruned.tokens == graph.tokens
    assert pruned.target_token == graph.target_token
    assert pruned.logit_value == graph.logit_value
    assert pruned.completeness == graph.completeness


def test_node_influence_scores_length(toy_transformer, toy_clt, toy_tokens):
    """node_influence_scores returns one entry per node."""
    full_cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, full_cfg)
    scores = node_influence_scores(graph, full_cfg)
    assert len(scores) == len(graph.nodes), (
        f"Expected {len(graph.nodes)} scores, got {len(scores)}"
    )


def test_node_influence_scores_sorted(toy_transformer, toy_clt, toy_tokens):
    """node_influence_scores should be sorted descending."""
    full_cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(toy_transformer, toy_clt, toy_tokens, 42, full_cfg)
    scores = node_influence_scores(graph, full_cfg)
    values = [s for _, s in scores]
    assert values == sorted(values, reverse=True), "Scores not sorted descending"
