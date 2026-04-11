"""
tests/test_export.py — Frontend JSON schema validation for graphs/export.py.

Verifies that to_frontend_json() produces output that matches the schema
expected by anthropics/attribution-graphs-frontend (util-cg.js → formatData).

Schema requirements checked:
  - Top-level keys: metadata, nodes, links
  - metadata: prompt_tokens (list of str), scan (str)
  - Each node: node_id, jsNodeId, layer, feature, feature_type, ctx_idx,
               probe_location_idx, clerp, vis_link, isLogit
  - Each link: source, target, weight (float)
  - feature_type values in allowed set
  - node_id and jsNodeId are equal (frontend uses both lookup paths)
  - source/target in links match node_ids
  - Logit node: isLogit=True, clerp contains '(p='
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from clt.config import AttributionConfig, CLTConfig
from clt.model import CrossLayerTranscoder
from graphs.build import build_attribution_graph
from graphs.export import save_graph, to_frontend_json
from graphs.prune import prune_graph


# ---------------------------------------------------------------------------
# Constants + fixtures
# ---------------------------------------------------------------------------

N_LAYERS = 2
D_MODEL = 64
D_MLP = 256
N_FEATURES = 32
N_HEADS = 4
D_VOCAB = 100
SEQ = 6

ALLOWED_FEATURE_TYPES = {
    "cross layer transcoder",
    "embedding",
    "mlp reconstruction error",
    "logit",
}


@pytest.fixture(scope="module")
def toy_transformer():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_MODEL // N_HEADS,
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
def toy_clt(toy_transformer):
    clt_cfg = CLTConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, d_mlp=D_MLP, n_features=N_FEATURES,
    )
    clt = CrossLayerTranscoder(clt_cfg)
    for source_decoders in clt.decoders:
        for decoder in source_decoders:
            torch.nn.init.normal_(decoder.weight, std=0.01)
    # CLT must live on the same device as the model it's paired with
    device = next(toy_transformer.parameters()).device
    clt = clt.to(device)
    clt.eval()
    return clt


@pytest.fixture(scope="module")
def sample_graph(toy_transformer, toy_clt):
    tokens = torch.randint(0, D_VOCAB, (1, SEQ))
    cfg = AttributionConfig(min_activation=0.0, top_k_nodes=10, top_k_edges=20)
    graph = build_attribution_graph(toy_transformer, toy_clt, tokens, 42, cfg)
    return prune_graph(graph, cfg)


@pytest.fixture(scope="module")
def frontend_data(sample_graph):
    return to_frontend_json(sample_graph, model_name="test-model", logit_probability=0.123)


# ---------------------------------------------------------------------------
# Top-level structure tests
# ---------------------------------------------------------------------------


def test_top_level_keys(frontend_data):
    assert "metadata" in frontend_data
    assert "nodes" in frontend_data
    assert "links" in frontend_data


def test_metadata_prompt_tokens(frontend_data, sample_graph):
    tokens = frontend_data["metadata"]["prompt_tokens"]
    assert isinstance(tokens, list)
    assert len(tokens) == SEQ
    assert all(isinstance(t, str) for t in tokens)


def test_metadata_scan(frontend_data):
    assert frontend_data["metadata"]["scan"] == "test-model"


# ---------------------------------------------------------------------------
# Node schema tests
# ---------------------------------------------------------------------------


def test_nodes_is_list(frontend_data):
    assert isinstance(frontend_data["nodes"], list)
    assert len(frontend_data["nodes"]) > 0


def test_node_required_fields(frontend_data):
    required = {
        "node_id", "jsNodeId", "layer", "feature", "feature_type",
        "ctx_idx", "probe_location_idx", "clerp", "vis_link", "isLogit",
    }
    for i, node in enumerate(frontend_data["nodes"]):
        missing = required - set(node.keys())
        assert not missing, f"Node {i} missing fields: {missing}"


def test_node_id_equals_js_node_id(frontend_data):
    for node in frontend_data["nodes"]:
        assert node["node_id"] == node["jsNodeId"], (
            f"node_id {node['node_id']!r} != jsNodeId {node['jsNodeId']!r}"
        )


def test_node_feature_type_valid(frontend_data):
    for node in frontend_data["nodes"]:
        assert node["feature_type"] in ALLOWED_FEATURE_TYPES, (
            f"Invalid feature_type: {node['feature_type']!r}"
        )


def test_node_feature_is_int(frontend_data):
    for node in frontend_data["nodes"]:
        assert isinstance(node["feature"], int), (
            f"Node feature should be int, got {type(node['feature'])}"
        )


def test_node_ctx_idx_is_int(frontend_data):
    for node in frontend_data["nodes"]:
        assert isinstance(node["ctx_idx"], int), (
            f"ctx_idx should be int, got {type(node['ctx_idx'])}"
        )
        assert 0 <= node["ctx_idx"] < SEQ, (
            f"ctx_idx {node['ctx_idx']} out of range [0, {SEQ})"
        )


def test_node_probe_location_idx_equals_ctx_idx(frontend_data):
    for node in frontend_data["nodes"]:
        assert node["probe_location_idx"] == node["ctx_idx"], (
            "probe_location_idx should equal ctx_idx"
        )


def test_logit_node_is_logit_true(frontend_data):
    logit_nodes = [n for n in frontend_data["nodes"] if n["feature_type"] == "logit"]
    assert len(logit_nodes) == 1, f"Expected 1 logit node, got {len(logit_nodes)}"
    assert logit_nodes[0]["isLogit"] is True


def test_non_logit_nodes_is_logit_false(frontend_data):
    for node in frontend_data["nodes"]:
        if node["feature_type"] != "logit":
            assert node["isLogit"] is False, (
                f"Non-logit node {node['node_id']} has isLogit=True"
            )


def test_logit_clerp_contains_probability(frontend_data):
    logit_node = next(n for n in frontend_data["nodes"] if n["feature_type"] == "logit")
    assert "(p=" in logit_node["clerp"], (
        f"Logit clerp missing '(p=...)': {logit_node['clerp']!r}"
    )


def test_embedding_nodes_layer_is_E(frontend_data):
    for node in frontend_data["nodes"]:
        if node["feature_type"] == "embedding":
            assert node["layer"] == "E", (
                f"Embedding node layer should be 'E', got {node['layer']!r}"
            )


def test_unique_node_ids(frontend_data):
    ids = [n["node_id"] for n in frontend_data["nodes"]]
    assert len(ids) == len(set(ids)), "Duplicate node_ids in frontend output"


# ---------------------------------------------------------------------------
# Link schema tests
# ---------------------------------------------------------------------------


def test_links_is_list(frontend_data):
    assert isinstance(frontend_data["links"], list)


def test_link_required_fields(frontend_data):
    for i, link in enumerate(frontend_data["links"]):
        assert "source" in link, f"Link {i} missing 'source'"
        assert "target" in link, f"Link {i} missing 'target'"
        assert "weight" in link, f"Link {i} missing 'weight'"


def test_link_weight_is_float(frontend_data):
    for link in frontend_data["links"]:
        assert isinstance(link["weight"], float), (
            f"Link weight should be float, got {type(link['weight'])}"
        )
        assert math.isfinite(link["weight"]), f"Link weight is not finite: {link['weight']}"


def test_link_endpoints_are_valid_node_ids(frontend_data):
    node_ids = {n["node_id"] for n in frontend_data["nodes"]}
    for link in frontend_data["links"]:
        assert link["source"] in node_ids, (
            f"Link source {link['source']!r} not in node_ids"
        )
        assert link["target"] in node_ids, (
            f"Link target {link['target']!r} not in node_ids"
        )


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------


def test_save_graph_writes_valid_json(sample_graph):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "graph_data" / "test.json"
        result_path = save_graph(sample_graph, out, model_name="test-model")

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "nodes" in data
        assert "links" in data


def test_save_graph_creates_parent_dirs(sample_graph):
    with tempfile.TemporaryDirectory() as tmpdir:
        deep_path = Path(tmpdir) / "a" / "b" / "c" / "graph.json"
        save_graph(sample_graph, deep_path)
        assert deep_path.exists()


def test_save_graph_roundtrip(sample_graph):
    """to_frontend_json → save → load should produce identical data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "graph.json"
        save_graph(sample_graph, out, model_name="round-trip-test", logit_probability=0.5)
        with open(out) as f:
            loaded = json.load(f)

    original = to_frontend_json(sample_graph, model_name="round-trip-test", logit_probability=0.5)
    assert loaded == original
