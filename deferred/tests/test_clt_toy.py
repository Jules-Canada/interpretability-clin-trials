"""
tests/test_clt_toy.py — End-to-end pipeline test on a 2-layer toy model.

Must pass before running on Pythia-410m. See CLAUDE.md dev rule 1.

Toy config: n_layers=2, d_model=64, d_mlp=256, n_features=128.
n_features < d_mlp deliberately — exercises the reconstruction zero-init path.
"""

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from clt.config import CLTConfig
from clt.model import CrossLayerTranscoder

# ---------------------------------------------------------------------------
# Toy model constants
# ---------------------------------------------------------------------------

N_LAYERS = 2
D_MODEL = 64
D_MLP = 256       # intentionally larger than N_FEATURES
N_FEATURES = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
D_VOCAB = 100
BATCH = 2
SEQ = 16


# ---------------------------------------------------------------------------
# Fixtures (module-scoped so the transformer is only instantiated once)
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
def clt_cfg() -> CLTConfig:
    return CLTConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_mlp=D_MLP,
        n_features=N_FEATURES,
    )


@pytest.fixture(scope="module")
def activations(toy_transformer) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract residual streams and MLP outputs from the toy transformer."""
    tokens = torch.randint(0, D_VOCAB, (BATCH, SEQ))
    resid_streams: list[torch.Tensor] = []
    mlp_outputs: list[torch.Tensor] = []

    with torch.no_grad():
        _, cache = toy_transformer.run_with_cache(tokens)
        for l in range(N_LAYERS):
            # (BATCH, SEQ, D_MODEL)
            resid_streams.append(cache[f"blocks.{l}.hook_resid_pre"])
            # (BATCH, SEQ, D_MLP) — post-activation hidden state, pre-W_out
            mlp_outputs.append(cache[f"blocks.{l}.mlp.hook_post"])

    return resid_streams, mlp_outputs


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def clt(clt_cfg) -> CrossLayerTranscoder:
    return CrossLayerTranscoder(clt_cfg).to(_default_device())


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_encode_shapes(clt, activations):
    resid_streams, _ = activations
    feature_acts = clt.encode(resid_streams)

    assert len(feature_acts) == N_LAYERS
    for l, a in enumerate(feature_acts):
        assert a.shape == (BATCH, SEQ, N_FEATURES), \
            f"Layer {l} feature_acts shape {a.shape} != {(BATCH, SEQ, N_FEATURES)}"


def test_decode_shapes(clt, activations):
    resid_streams, _ = activations
    feature_acts = clt.encode(resid_streams)
    mlp_recons = clt.decode(feature_acts)

    assert len(mlp_recons) == N_LAYERS
    for l, r in enumerate(mlp_recons):
        assert r.shape == (BATCH, SEQ, D_MLP), \
            f"Layer {l} reconstruction shape {r.shape} != {(BATCH, SEQ, D_MLP)}"


def test_forward_shapes(clt, activations):
    resid_streams, _ = activations
    feature_acts, mlp_recons = clt(resid_streams)

    for l in range(N_LAYERS):
        assert feature_acts[l].shape == (BATCH, SEQ, N_FEATURES)
        assert mlp_recons[l].shape == (BATCH, SEQ, D_MLP)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

def test_loss_keys_and_finite(clt, activations):
    resid_streams, mlp_targets = activations
    losses = clt.loss(resid_streams, mlp_targets)

    assert set(losses.keys()) == {"total", "reconstruction", "sparsity"}
    for key, val in losses.items():
        assert val.isfinite(), f"Loss '{key}' is not finite: {val}"


def test_loss_total_equals_sum(clt, activations):
    resid_streams, mlp_targets = activations
    losses = clt.loss(resid_streams, mlp_targets)

    expected = losses["reconstruction"] + losses["sparsity"]
    assert torch.allclose(losses["total"], expected), \
        f"total={losses['total']:.6f} != recon+sparsity={expected:.6f}"


def test_loss_decreases_over_steps(clt_cfg, activations):
    """Ten gradient steps must reduce the total loss."""
    resid_streams, mlp_targets = activations

    model = CrossLayerTranscoder(clt_cfg).to(_default_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history: list[float] = []
    for _ in range(10):
        optimizer.zero_grad()
        losses = model.loss(resid_streams, mlp_targets)
        losses["total"].backward()
        optimizer.step()
        history.append(losses["total"].item())

    assert history[-1] < history[0], (
        f"Loss did not decrease over 10 steps: {history[0]:.4f} → {history[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# JumpReLU / sparsity tests
# ---------------------------------------------------------------------------

def test_jumprelu_produces_zeros(clt, activations):
    """JumpReLU must zero out some activations (features must be sparse)."""
    resid_streams, _ = activations
    feature_acts, _ = clt(resid_streams)

    for l, a in enumerate(feature_acts):
        frac_zero = (a.abs() < 1e-6).float().mean().item()
        assert frac_zero > 0.0, \
            f"Layer {l}: no zero activations — JumpReLU may not be thresholding"


def test_l0_per_layer_nonnegative(clt, activations):
    resid_streams, _ = activations
    feature_acts, _ = clt(resid_streams)

    l0 = clt.l0_per_layer(feature_acts)
    assert len(l0) == N_LAYERS
    for l, val in enumerate(l0):
        assert val >= 0.0, f"Negative L0 at layer {l}: {val}"
        assert val <= N_FEATURES, f"L0 exceeds n_features at layer {l}: {val}"


# ---------------------------------------------------------------------------
# Cross-layer transcoding test
# ---------------------------------------------------------------------------

def test_cross_layer_contribution(clt_cfg, activations):
    """
    Zeroing layer-0 features must change reconstructions at ALL layers (l=0 and l=1),
    confirming cross-layer decoding paths are wired correctly.

    Uses a fresh model with randomly-initialized (non-zero) decoders so that
    zeroing features has a measurable effect regardless of training state.
    """
    import torch.nn as nn
    resid_streams, _ = activations

    # Fresh model; replace zero-init decoders with small random weights
    model = CrossLayerTranscoder(clt_cfg).to(_default_device())
    for l_source in range(N_LAYERS):
        for decoder in model.decoders[l_source]:
            nn.init.normal_(decoder.weight, std=0.01)

    feature_acts, _ = model(resid_streams)

    zeroed = [a.clone() for a in feature_acts]
    # (BATCH, SEQ, N_FEATURES) — zero out layer-0 contributions
    zeroed[0] = torch.zeros_like(zeroed[0])

    recons_full = model.decode(feature_acts)
    recons_zeroed = model.decode(zeroed)

    for l in range(N_LAYERS):
        assert not torch.allclose(recons_full[l], recons_zeroed[l]), \
            f"Layer {l} reconstruction unchanged after zeroing layer-0 features — " \
            f"cross-layer decoder W_dec[0→{l}] may be disconnected"


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def test_active_features_structure(clt, activations):
    resid_streams, _ = activations
    feature_acts, _ = clt(resid_streams)

    active = clt.active_features(feature_acts)
    assert isinstance(active, list)
    if active:
        assert set(active[0].keys()) == {"layer", "batch", "pos", "feature", "activation"}
        for entry in active:
            assert 0 <= entry["layer"] < N_LAYERS
            assert entry["activation"] > 1e-6
