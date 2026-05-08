"""
tests/test_attribution_completeness.py

Regression tests for graphs.build.build_attribution_graph completeness.
Phase 2 of the autograd rewrite (docs/autograd_plan.md).

Three tests at three model scales:

  1. Toy 2-layer transformer            — fast, runs on every pytest invocation
  2. Pythia-70m on the Phase 0 prompt   — fast (~2s), runs on every invocation
  3. MedGemma-4B on an eligibility prompt
                                        — @pytest.mark.slow, skipped unless
                                          checkpoint + saved scales are present

Each test asserts a completeness threshold. If a future change to the attribution
code (or to TransformerLens) silently degrades the graph, one of these tests will
catch it before the pipeline graphs go stale.

To run only the fast tests:
    .venv/bin/python -m pytest tests/test_attribution_completeness.py -v

To include the slow MedGemma test (requires checkpoint with saved scales — see
scripts/compute_clt_scales.py):
    .venv/bin/python -m pytest tests/test_attribution_completeness.py -v -m slow

To include both fast and slow:
    .venv/bin/python -m pytest tests/test_attribution_completeness.py -v -m "slow or not slow"
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from clt.config import AttributionConfig, CLTConfig
from clt.model import CrossLayerTranscoder
from graphs.build import build_attribution_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _random_clt(clt_cfg: CLTConfig, decoder_std: float = 0.01) -> CrossLayerTranscoder:
    """Build a CLT with non-zero random decoders so edges are non-trivial."""
    clt = CrossLayerTranscoder(clt_cfg)
    for source_decoders in clt.decoders:
        for decoder in source_decoders:
            torch.nn.init.normal_(decoder.weight, std=decoder_std)
    clt.eval()
    return clt


# ---------------------------------------------------------------------------
# Test 1 — Toy 2-layer model
# ---------------------------------------------------------------------------
#
# Fixed seed + fixed config + fixed token sequence + fixed target token gives a
# fully deterministic completeness number.  Recorded at Phase 2 authoring time
# (2026-05-03) under torch 2.11.0 on MPS with the autograd build.

TOY_SEED = 0
TOY_N_LAYERS = 2
TOY_D_MODEL = 64
TOY_D_MLP = 256
TOY_N_FEATURES = 32
TOY_N_HEADS = 4
TOY_D_HEAD = TOY_D_MODEL // TOY_N_HEADS
TOY_D_VOCAB = 100
TOY_SEQ_LEN = 8
TOY_TARGET_TOKEN_IDX = 42

TOY_RECORDED_COMPLETENESS = 1.000000
TOY_TOLERANCE = 0.02


def test_toy_completeness_within_tolerance():
    """Toy 2-layer transformer + random CLT must give the recorded completeness."""
    torch.manual_seed(TOY_SEED)
    device = _device()

    cfg = HookedTransformerConfig(
        n_layers=TOY_N_LAYERS,
        d_model=TOY_D_MODEL,
        n_heads=TOY_N_HEADS,
        d_head=TOY_D_HEAD,
        d_mlp=TOY_D_MLP,
        n_ctx=TOY_SEQ_LEN,
        act_fn="gelu",
        normalization_type="LN",
        d_vocab=TOY_D_VOCAB,
    )
    model = HookedTransformer(cfg).to(device)
    model.eval()

    clt_cfg = CLTConfig(
        n_layers=TOY_N_LAYERS,
        d_model=TOY_D_MODEL,
        d_mlp=TOY_D_MLP,
        n_features=TOY_N_FEATURES,
    )
    clt = _random_clt(clt_cfg).to(device)

    tokens = torch.randint(0, TOY_D_VOCAB, (1, TOY_SEQ_LEN), device=device)

    attr_cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(model, clt, tokens, TOY_TARGET_TOKEN_IDX, attr_cfg)

    assert math.isfinite(graph.completeness), \
        f"Toy completeness is not finite: {graph.completeness}"
    assert abs(graph.completeness - TOY_RECORDED_COMPLETENESS) < TOY_TOLERANCE, (
        f"Toy completeness {graph.completeness:.6f} drifted from "
        f"recorded {TOY_RECORDED_COMPLETENESS:.6f} by more than ±{TOY_TOLERANCE}. "
        f"If the autograd build changed deliberately, re-run "
        f"scripts/phase0_baseline.py-style probe and update TOY_RECORDED_COMPLETENESS."
    )


# ---------------------------------------------------------------------------
# Test 2 — Pythia-70m on the Phase 0 prompt
# ---------------------------------------------------------------------------
#
# Same configuration recorded in docs/phase0_baseline.json (Phase 0 manual baseline)
# and docs/phase1_autograd.json (Phase 1 autograd validation).  Phase 1 reached
# completeness 0.8996; the 0.85 threshold leaves ~0.05 headroom for normal
# floating-point variation across hardware.

PYTHIA_MODEL = "EleutherAI/pythia-70m"
PYTHIA_SEED = 0
PYTHIA_N_FEATURES = 128
PYTHIA_PROMPT = "The capital of France is"
PYTHIA_TARGET_TOKEN = " Paris"
PYTHIA_MIN_COMPLETENESS = 0.85


def test_pythia70m_completeness_above_threshold():
    """Pythia-70m + random CLT must reach the documented completeness floor."""
    torch.manual_seed(PYTHIA_SEED)
    device = _device()

    model = HookedTransformer.from_pretrained(PYTHIA_MODEL)
    model.eval()
    model = model.to(device)

    clt_cfg = CLTConfig(
        n_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        d_mlp=model.cfg.d_mlp,
        n_features=PYTHIA_N_FEATURES,
    )
    clt = _random_clt(clt_cfg).to(device)

    tokens = model.to_tokens(PYTHIA_PROMPT).to(device)
    target_token_idx = model.to_single_token(PYTHIA_TARGET_TOKEN)

    attr_cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(model, clt, tokens, target_token_idx, attr_cfg)

    assert math.isfinite(graph.completeness), \
        f"Pythia-70m completeness is not finite: {graph.completeness}"
    assert graph.completeness >= PYTHIA_MIN_COMPLETENESS, (
        f"Pythia-70m completeness {graph.completeness:.6f} fell below the "
        f"threshold {PYTHIA_MIN_COMPLETENESS}. The autograd build may have "
        f"regressed; check docs/phase1_autograd.json for the previously-recorded "
        f"value (0.8996 at authoring time)."
    )


# ---------------------------------------------------------------------------
# Test 3 — MedGemma-4B on an eligibility prompt (slow, opt-in)
# ---------------------------------------------------------------------------
#
# This is the test that actually matters most for the Phase 2 milestone: it
# proves the autograd rewrite fixes the MedGemma completeness inflation that
# motivated the rewrite (manual code: 4–8; target: ≥ 0.5).
#
# Skip conditions:
#   - CLT checkpoint not present locally
#   - Checkpoint missing saved RMS scales (per-prompt RMS fallback is known to
#     give garbage on Gemma 3 — see CLAUDE.md "RMS scale persistence")
#
# Run with the saved-scales checkpoint produced after Phase 4's
# scripts/compute_clt_scales.py pass.  See docs/autograd_plan.md Phase 4.

MEDGEMMA_CHECKPOINT = Path("checkpoints/medgemma-4b-1024/clt_inference.pt")
MEDGEMMA_MODEL = "google/medgemma-4b-pt"
MEDGEMMA_N_FEATURES = 1024
MEDGEMMA_PROMPT = "The patient meets all inclusion criteria and is therefore"
MEDGEMMA_TARGET_TOKEN = " eligible"
MEDGEMMA_MIN_COMPLETENESS = 0.5


def _medgemma_skip_reason() -> str | None:
    """Return None if the test can run; a string explaining why otherwise."""
    if not MEDGEMMA_CHECKPOINT.exists():
        return f"checkpoint not found at {MEDGEMMA_CHECKPOINT}"
    ckpt = torch.load(MEDGEMMA_CHECKPOINT, map_location="cpu", weights_only=True)
    if "resid_scales" not in ckpt or "mlp_scales" not in ckpt:
        return ("checkpoint missing saved RMS scales — run "
                "scripts/compute_clt_scales.py first (needs HDF5 access)")
    return None


@pytest.mark.slow
def test_medgemma_completeness_above_threshold():
    """MedGemma-4B + trained CLT must reach completeness ≥ 0.5 with the autograd build."""
    skip = _medgemma_skip_reason()
    if skip is not None:
        pytest.skip(skip)

    device = _device()

    model = HookedTransformer.from_pretrained(MEDGEMMA_MODEL)
    model.eval()
    model = model.to(device)

    clt_cfg = CLTConfig(
        n_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        d_mlp=model.cfg.d_mlp,
        n_features=MEDGEMMA_N_FEATURES,
    )
    clt = CrossLayerTranscoder(clt_cfg)

    ckpt = torch.load(MEDGEMMA_CHECKPOINT, map_location=device, weights_only=True)
    clt.load_state_dict(ckpt["model_state_dict"])
    clt = clt.to(device)
    clt.eval()
    scales_loaded = clt.load_scales_from_checkpoint(ckpt)
    assert scales_loaded, "expected saved scales after the skip-reason check passed"

    tokens = model.to_tokens(MEDGEMMA_PROMPT).to(device)
    target_token_idx = model.to_single_token(MEDGEMMA_TARGET_TOKEN)

    attr_cfg = AttributionConfig(min_activation=0.0)
    graph = build_attribution_graph(model, clt, tokens, target_token_idx, attr_cfg)

    assert math.isfinite(graph.completeness), \
        f"MedGemma completeness is not finite: {graph.completeness}"
    assert graph.completeness >= MEDGEMMA_MIN_COMPLETENESS, (
        f"MedGemma completeness {graph.completeness:.6f} fell below the target "
        f"{MEDGEMMA_MIN_COMPLETENESS}. The autograd rewrite was supposed to fix "
        f"the post-norm-induced inflation; if it didn't, see docs/diagnostics_log.md "
        f"and consider Phase 5 (CLT retrain) of docs/autograd_plan.md."
    )
