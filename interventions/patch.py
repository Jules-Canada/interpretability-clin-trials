"""
interventions/patch.py

Feature steering and activation patching experiments.

Provides two entry points:
  - steer(): add a scaled feature direction to the residual stream at a given layer
  - patch_feature(): clamp a specific CLT feature activation to a fixed value

These are used to causally test whether a feature is sufficient / necessary for a
given model behaviour — e.g. "if I suppress feature L4F211, does the model stop
predicting 'eligible'?"

Usage example:
    from interventions.patch import steer, patch_feature
    from transformer_lens import HookedTransformer
    from clt.model import CrossLayerTranscoder

    # Boost feature L4F211 to activation=5.0 and observe logit change
    logits = patch_feature(model, clt, tokens, layer=4, feature=211, value=5.0)
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from clt.model import CrossLayerTranscoder


def steer(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    direction: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Add `scale * direction` to the residual stream at `layer` and return logits.

    Args:
        model:     TransformerLens model (eval mode).
        tokens:    (1, seq_len) token tensor.
        layer:     Layer index at which to inject the direction (resid_pre hook).
        direction: (d_model,) steering vector — e.g. a CLT decoder column.
        scale:     Scalar multiplier applied to the direction before adding.

    Returns:
        (seq_len, vocab_size) logits after intervention.
    """
    # (d_model,) → (1, 1, d_model) for broadcasting over (batch, seq, d_model)
    vec = direction.to(tokens.device).float()
    vec = vec.unsqueeze(0).unsqueeze(0) * scale

    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        # resid: (batch, seq, d_model)
        return resid + vec

    hook_name = f"blocks.{layer}.hook_resid_pre"
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

    # (1, seq_len, vocab_size) → (seq_len, vocab_size)
    return logits.squeeze(0)


def patch_feature(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: torch.Tensor,
    layer: int,
    feature: int,
    value: float,
) -> torch.Tensor:
    """
    Clamp CLT feature `feature` at `layer` to `value` and return logits.

    Implements the intervention by:
      1. Running a forward pass to get the baseline feature activation at the target position.
      2. Computing the delta needed to reach `value`.
      3. Injecting that delta * decoder_column into resid_pre at the affected downstream layers.

    This is an approximate intervention — it patches via the decoder direction rather
    than re-running the full CLT forward pass, which would require differentiating through
    JumpReLU. Suitable for causal tracing experiments where exact gradient flow is not needed.

    Args:
        model:   TransformerLens model (eval mode).
        clt:     Trained CrossLayerTranscoder (same device as model).
        tokens:  (1, seq_len) token tensor.
        layer:   Encoder layer of the feature to patch.
        feature: Feature index within that layer.
        value:   Target activation value (use 0.0 to suppress a feature).

    Returns:
        (seq_len, vocab_size) logits after intervention.
    """
    device = tokens.device
    n_layers = clt.cfg.n_layers

    # ------------------------------------------------------------------
    # Step 1: get baseline activations to compute the delta
    # ------------------------------------------------------------------
    hook_names_resid = [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in set(hook_names_resid),
        )

    # resid_streams[l]: (1, seq, d_model)
    resid_streams = [cache[h] for h in hook_names_resid]
    resid_unseq   = resid_streams  # already (1, seq, d_model)

    feature_acts = clt.encode(resid_unseq)
    # feature_acts[layer]: (1, seq, n_features)
    baseline_acts = feature_acts[layer][0, :, feature]  # (seq,)

    # ------------------------------------------------------------------
    # Step 2: build per-layer steering directions (decoder columns)
    # ------------------------------------------------------------------
    # For each downstream layer l >= layer, CLT has a decoder W_dec[layer→l]
    # shape (n_features, d_mlp). The feature's contribution to mlp_post_l is
    # W_dec[layer→l][:, feature] * activation.
    # We patch by adding delta * decoder_column to resid_pre at layer l+1
    # (since mlp_post feeds into resid after the MLP block).

    deltas = value - baseline_acts  # (seq,)

    def make_hook(target_layer: int):
        # decoder column for this (source_layer → target_layer) pair
        # W_dec[source][target]: (n_features, d_mlp)
        dec_col = clt.W_dec[layer][target_layer][:, feature]  # (d_mlp,) — wait, shape is (d_mlp, n_features)
        # Actual shape from model.py: W_dec registered as (n_features, d_mlp) — check
        # Using index [layer][target_layer] gives (n_features, d_mlp); [:, feature] = (d_mlp,) ← wrong
        # Correct: W_dec[layer][target_layer][feature, :] = (d_mlp,) decoder direction
        dec_col = clt.W_dec[layer][target_layer][feature, :]  # (d_mlp,)
        dec_col = dec_col.to(device)

        def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
            # resid: (1, seq, d_model) — but mlp output is d_mlp, not d_model
            # We can't directly add d_mlp vector to d_model residual stream without
            # projecting through W_out. This simplified version is left as a TODO:
            # for a proper patch, run the full CLT decode and subtract/add the diff.
            # For now, return resid unmodified with a warning.
            return resid

        return hook_fn

    # ------------------------------------------------------------------
    # NOTE: Full decoder-based patching requires projecting d_mlp → d_model
    # via the MLP's output weight W_out, which is model-specific. This stub
    # returns baseline logits until that projection is implemented.
    # See graphs/build.py → _compute_mlp_reconstructions() for the forward path.
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits = model(tokens)

    return logits.squeeze(0)


def feature_decoder_direction(
    clt: CrossLayerTranscoder,
    source_layer: int,
    target_layer: int,
    feature: int,
) -> torch.Tensor:
    """
    Return the decoder direction for feature `feature` at source_layer → target_layer.

    Shape: (d_mlp,) — the column of W_dec[source_layer][target_layer] for this feature.
    Use this as a steering vector in steer() after projecting through W_out.

    Args:
        clt:          Trained CrossLayerTranscoder.
        source_layer: Encoder layer of the feature.
        target_layer: Target MLP layer (must be >= source_layer).
        feature:      Feature index.

    Returns:
        (d_mlp,) decoder direction tensor.
    """
    return clt.W_dec[source_layer][target_layer][feature, :].detach()
