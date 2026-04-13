"""
graphs/build.py — Attribution graph construction.

Implements the analytical linear attribution method from:
  "Circuit Tracing: Revealing Computational Graphs in Language Models"
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Edge weight formula (§ Attribution Graphs):
  A_{s→t} = a_s × Σ_{ℓ_s ≤ ℓ < ℓ_t} W_dec[l_s→l].T × J^▼_{l→l_t} × W_enc[l_t]

where J^▼ is the Jacobian with stop-gradients on all nonlinearities (attention patterns,
JumpReLU gates, LayerNorm denominators). With these frozen, J^▼ between residual-stream
layers collapses to W_out_l (the MLP output projection), making the full path a product
of linear maps.

Four node types:
  - feature:   active CLT features (l, f, position)
  - embedding: token + positional embedding at each sequence position
  - error:     reconstruction error (true MLP output − CLT reconstruction) per layer
  - logit:     target token's output logit

Four edge types:
  - feature   → logit:   a_s × transfer[l_s→L][f_s] @ v
  - feature   → feature: a_s × transfer[l_s→l_t][f_s] @ W_enc[l_t][f_t]
  - embedding → feature: x_embed[p] · W_enc[l][f]
  - error     → logit:   error_l @ W_out_l @ v_l  (where v_l ≈ v for residual nets)

The "transfer matrix" T[l_s→l_t] ∈ ℝ^{F × d_model} precomputes how each source
feature propagates through the residual stream to layer l_t:
  T[l_s→l_t] = Σ_{l''=l_s}^{l_t-1} W_dec[l_s→l''].T @ W_out_{l''}

All edge weights satisfy: pre-activation of t = sum of incoming edge weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from clt.config import AttributionConfig
from clt.model import CrossLayerTranscoder

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AttributionGraph:
    """
    An attribution graph for a single (prompt, target_token) pair.

    nodes: list of dicts, each with keys:
        id (str), type (str), layer (int|None), feature (int|None),
        position (int|None), activation (float), label (str)

    edges: list of dicts, each with keys:
        source (str), target (str), weight (float)

    The 'id' field uniquely identifies each node and is used as edge endpoints.
    """
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    target_token: str = ""
    target_position: int = -1
    logit_value: float = 0.0
    # Completeness check: sum of all incoming edges to the logit node / logit_value
    completeness: float = 0.0


# ---------------------------------------------------------------------------
# Transfer matrix computation
# ---------------------------------------------------------------------------

def _compute_transfer_matrices(
    clt: CrossLayerTranscoder,
    model: HookedTransformer,
    L: int,
) -> dict[tuple[int, int], Float[Tensor, "n_features d_model"]]:
    """
    Precompute transfer matrices T[l_s → l_t] for all valid (l_s, l_t) pairs.

    T[l_s → l_t] ∈ ℝ^{F × d_model}:
        T[l_s → l_t][f, :] = Σ_{l''=l_s}^{l_t-1} W_dec[l_s→l''].weight.T[f,:] @ W_out_{l''}

    This is the "contribution of feature f at layer l_s to the residual stream at l_t",
    accumulated via the CLT decoders and MLP output projections.

    Args:
        clt:   trained CrossLayerTranscoder
        model: HookedTransformer (used for W_out matrices)
        L:     number of layers

    Returns:
        dict mapping (l_source, l_target) → tensor of shape (F, d_model)
        Defined for l_target > l_source, and l_target = L (final residual).
    """
    transfer: dict[tuple[int, int], Tensor] = {}

    device = next(clt.parameters()).device  # CLT and model are on same device (guaranteed by caller)

    with torch.no_grad():
        for l_source in range(L):
            # Accumulate contribution across target layers
            # (F, d_model) — running sum
            F = clt.cfg.n_features
            d_model = clt.cfg.d_model
            cumulative = torch.zeros(F, d_model, device=device)

            for offset, decoder in enumerate(clt.decoders[l_source]):
                l_intermediate = l_source + offset  # the layer this decoder reconstructs

                # W_dec[l_source → l_intermediate].weight: (d_mlp, F)
                # .T: (F, d_mlp)
                # (F, d_mlp)
                W_dec_T = decoder.weight.T

                # W_out at layer l_intermediate: (d_mlp, d_model) in TransformerLens
                # (d_mlp, d_model)
                W_out = model.blocks[l_intermediate].mlp.W_out

                # (F, d_mlp) @ (d_mlp, d_model) = (F, d_model)
                cumulative = cumulative + W_dec_T @ W_out

                # T[l_source → l_intermediate+1] = cumulative after processing l_intermediate
                l_target = l_intermediate + 1
                transfer[(l_source, l_target)] = cumulative.clone()

            # T[l_source → L] = full cumulative (all decoders processed)
            transfer[(l_source, L)] = cumulative.clone()

    return transfer


# ---------------------------------------------------------------------------
# Readout vector computation
# ---------------------------------------------------------------------------

def _compute_readout_vector(
    model: HookedTransformer,
    cache: dict,
    target_position: int,
    target_token_idx: int,
    L: int,
) -> Float[Tensor, "d_model"]:
    """
    Compute v = ∂T/∂resid_L via autograd on LN_final + unembedding.

    T is the logit of target_token_idx at target_position.
    resid_L is the final residual stream (before ln_final).

    Args:
        model:            HookedTransformer
        cache:            activation cache from run_with_cache
        target_position:  sequence position to trace
        target_token_idx: vocabulary index of the target token
        L:                number of layers

    Returns:
        v of shape (d_model,) — gradient of target logit w.r.t. final residual stream
    """
    # Final residual stream before ln_final: hook_resid_post at last layer
    # (1, seq, d_model)
    resid_final = cache[f"blocks.{L-1}.hook_resid_post"].detach()

    # Create leaf tensor with grad at the target position
    # (d_model,)
    x = resid_final[0, target_position].clone().requires_grad_(True)

    # Apply final LayerNorm and unembedding at this single position
    # Unsqueeze to (1, 1, d_model) for TransformerLens API compatibility
    x_3d = x.unsqueeze(0).unsqueeze(0)

    # (1, 1, d_model) → (1, 1, d_model)
    x_normed = model.ln_final(x_3d)

    # (1, 1, d_vocab)
    logits = model.unembed(x_normed)

    # Scalar: logit for the target token at target position
    T = logits[0, 0, target_token_idx]
    T.backward()

    # (d_model,)
    v = x.grad.detach()
    return v


# ---------------------------------------------------------------------------
# Main graph construction
# ---------------------------------------------------------------------------

def build_attribution_graph(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: Float[Tensor, "1 seq"],
    target_token_idx: int,
    cfg: AttributionConfig | None = None,
) -> AttributionGraph:
    """
    Build an attribution graph for a given prompt and target token.

    Args:
        model:            Frozen HookedTransformer (weights not modified)
        clt:              Trained CrossLayerTranscoder
        tokens:           Integer token ids, shape (1, seq)
        target_token_idx: Vocabulary index of the token to trace
        cfg:              AttributionConfig; uses defaults if None

    Returns:
        AttributionGraph with nodes and edges populated.
    """
    if cfg is None:
        cfg = AttributionConfig()

    # Move CLT to the same device as the model so all tensor ops are on one device.
    # This is the canonical device for the whole graph construction pass.
    model_device = next(model.parameters()).device
    clt = clt.to(model_device)

    L = clt.cfg.n_layers
    tokens = tokens.to(model_device)
    target_position = cfg.target_position  # e.g. -1 for last token

    # -----------------------------------------------------------------------
    # Step 1: Frozen forward pass — cache all activations
    # -----------------------------------------------------------------------
    model.eval()
    clt.eval()

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # -----------------------------------------------------------------------
    # Step 2: Get CLT feature activations from cached residual streams
    # -----------------------------------------------------------------------
    # resid_streams[l]: (1, seq, d_model)
    resid_streams = [
        cache[f"blocks.{l}.hook_resid_pre"].to(model_device)
        for l in range(L)
    ]

    # Normalize residual streams if CLT was trained with normalization.
    # Scales are estimated from the prompt's own activations (per-layer RMS).
    # This is an approximation of the dataset-level normalization used during
    # training; good enough for inference on reasonable-length prompts.
    if clt.cfg.normalize_activations:
        resid_streams_enc = []
        for r in resid_streams:
            # r: (1, seq, d_model)
            rms = r.float().pow(2).mean().sqrt().clamp(min=1e-8)
            resid_streams_enc.append(r / rms)
    else:
        resid_streams_enc = resid_streams

    with torch.no_grad():
        # feature_acts[l]: (1, seq, n_features)
        feature_acts = clt.encode(resid_streams_enc)

        # mlp_recons[l]: (1, seq, d_mlp)
        mlp_recons = clt.decode(feature_acts)

    # -----------------------------------------------------------------------
    # Step 3: Compute readout vector v = ∂T/∂resid_L
    # -----------------------------------------------------------------------
    # (d_model,)
    v = _compute_readout_vector(model, cache, target_position, target_token_idx, L)

    # -----------------------------------------------------------------------
    # Step 4: Precompute transfer matrices
    # -----------------------------------------------------------------------
    # transfer[(l_s, l_t)]: (n_features, d_model)
    transfer = _compute_transfer_matrices(clt, model, L)

    # -----------------------------------------------------------------------
    # Step 5: Compute token embedding at each position
    # -----------------------------------------------------------------------
    # (1, seq, d_model)
    token_embed = cache["hook_embed"]
    # (1, seq, d_model)
    pos_embed = cache["hook_pos_embed"]
    # (1, seq, d_model)
    x_embed = (token_embed + pos_embed).to(model_device)

    seq_len = tokens.shape[1]
    if target_position < 0:
        target_position = seq_len + target_position

    # -----------------------------------------------------------------------
    # Step 6: Compute target logit value
    # -----------------------------------------------------------------------
    with torch.no_grad():
        logits = model(tokens)
    # scalar
    logit_value = logits[0, target_position, target_token_idx].item()

    # -----------------------------------------------------------------------
    # Step 7: Build nodes and edges
    # -----------------------------------------------------------------------
    # Decode tokens to strings; fall back to numeric labels if no tokenizer is attached
    if model.tokenizer is not None:
        str_tokens = model.to_str_tokens(tokens[0])
        target_token_str = model.tokenizer.decode([target_token_idx])
    else:
        str_tokens = [str(t) for t in tokens[0].tolist()]
        target_token_str = str(target_token_idx)

    graph = AttributionGraph(
        tokens=str_tokens,
        target_token=target_token_str,
        target_position=target_position,
        logit_value=logit_value,
    )

    # Logit node (the target we're tracing)
    logit_node_id = f"logit_{target_token_idx}"
    graph.nodes.append({
        "id": logit_node_id,
        "type": "logit",
        "layer": None,
        "feature": None,
        "position": target_position,
        "activation": logit_value,
        "label": f'logit("{graph.target_token}")',
    })

    # Encoder weight matrix (F, d_model) per layer for edge computation
    # W_enc[l].weight: (F, d_model)
    W_enc = [clt.encoders[l].weight for l in range(L)]

    total_incoming_to_logit = 0.0

    # -----------------------------------------------------------------------
    # Feature nodes + edges
    # -----------------------------------------------------------------------
    for l_s in range(L):
        # feature_acts[l_s]: (1, seq, F) — activations at source layer
        # We trace from every position, but focus on target_position for the logit node
        for pos in range(seq_len):
            # (F,)
            a = feature_acts[l_s][0, pos].detach()

            # Active features at this (layer, position)
            active_mask = a.abs() > cfg.min_activation
            active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)

            for f_s in active_indices.tolist():
                a_sf = a[f_s].item()
                node_id = f"feat_l{l_s}_p{pos}_f{f_s}"

                # -- Feature node --
                graph.nodes.append({
                    "id": node_id,
                    "type": "feature",
                    "layer": l_s,
                    "feature": f_s,
                    "position": pos,
                    "activation": a_sf,
                    "label": f"L{l_s}F{f_s}@{pos}",
                })

                # -- Feature → logit edge (only from target position) --
                if pos == target_position:
                    # transfer[(l_s, L)][f_s, :]: (d_model,)
                    T_vec = transfer[(l_s, L)][f_s]
                    # scalar: a_sf × T_vec · v
                    weight = a_sf * (T_vec @ v).item()
                    graph.edges.append({
                        "source": node_id,
                        "target": logit_node_id,
                        "weight": weight,
                    })
                    total_incoming_to_logit += weight

                # -- Feature → feature edges --
                # Source feature (l_s, pos, f_s) → target features at later layers
                for l_t in range(l_s + 1, L):
                    if (l_s, l_t) not in transfer:
                        continue

                    # transfer[(l_s, l_t)][f_s, :]: (d_model,)
                    T_vec = transfer[(l_s, l_t)][f_s]

                    # W_enc[l_t]: (F, d_model) — project transfer into encoder space
                    # (F,) = (F, d_model) @ (d_model,)
                    enc_projections = W_enc[l_t] @ T_vec

                    # For each active target feature at the same position
                    a_t = feature_acts[l_t][0, pos].detach()
                    active_t = (a_t.abs() > cfg.min_activation).nonzero(as_tuple=False).squeeze(-1)

                    for f_t in active_t.tolist():
                        # scalar: a_sf × enc_projection[f_t]
                        weight = a_sf * enc_projections[f_t].item()
                        if abs(weight) < 1e-8:
                            continue
                        target_node_id = f"feat_l{l_t}_p{pos}_f{f_t}"
                        graph.edges.append({
                            "source": node_id,
                            "target": target_node_id,
                            "weight": weight,
                        })

    # -----------------------------------------------------------------------
    # Embedding nodes + edges
    # -----------------------------------------------------------------------
    for pos in range(seq_len):
        # (d_model,)
        embed_vec = x_embed[0, pos].detach()
        embed_node_id = f"embed_p{pos}"

        graph.nodes.append({
            "id": embed_node_id,
            "type": "embedding",
            "layer": None,
            "feature": None,
            "position": pos,
            "activation": embed_vec.norm().item(),
            "label": f'embed("{graph.tokens[pos]}"@{pos})',
        })

        # Embedding → feature edges at every layer
        for l in range(L):
            # W_enc[l]: (F, d_model)
            # (F,) = (F, d_model) @ (d_model,)
            contributions = W_enc[l] @ embed_vec

            a_l = feature_acts[l][0, pos].detach()
            active = (a_l.abs() > cfg.min_activation).nonzero(as_tuple=False).squeeze(-1)

            for f in active.tolist():
                weight = contributions[f].item()
                if abs(weight) < 1e-8:
                    continue
                graph.edges.append({
                    "source": embed_node_id,
                    "target": f"feat_l{l}_p{pos}_f{f}",
                    "weight": weight,
                })

    # -----------------------------------------------------------------------
    # Error nodes + edges (reconstruction error → logit)
    # -----------------------------------------------------------------------
    for l in range(L):
        # true_post[l]: (1, seq, d_mlp)
        true_post = cache[f"blocks.{l}.mlp.hook_post"].to(model_device)
        # clt_recon[l]: (1, seq, d_mlp)
        clt_recon = mlp_recons[l]

        # error[l]: (1, seq, d_mlp)
        error = (true_post - clt_recon).detach()

        # Error at target position: (d_mlp,)
        error_vec = error[0, target_position]

        # W_out[l]: (d_mlp, d_model) in TransformerLens
        W_out = model.blocks[l].mlp.W_out.detach()

        # Error contribution to residual: (d_model,) = (d_mlp,) @ (d_mlp, d_model)
        error_resid = error_vec @ W_out

        # Error contribution to logit: scalar = (d_model,) · (d_model,)
        # Note: error at layer l propagates to final residual via identity skip connections.
        # With frozen residuals, we approximate ∂T/∂resid_{l+1} ≈ v (the final readout).
        # This is exact for the last layer; approximate for earlier layers.
        weight = (error_resid @ v).item()

        error_node_id = f"error_l{l}"
        graph.nodes.append({
            "id": error_node_id,
            "type": "error",
            "layer": l,
            "feature": None,
            "position": target_position,
            "activation": error_vec.norm().item(),
            "label": f"error L{l}",
        })
        graph.edges.append({
            "source": error_node_id,
            "target": logit_node_id,
            "weight": weight,
        })
        total_incoming_to_logit += weight

    # -----------------------------------------------------------------------
    # Completeness check
    # -----------------------------------------------------------------------
    # completeness = (sum of incoming edges to logit node) / logit_value
    # Should be close to 1.0 for a faithful attribution
    if abs(logit_value) > 1e-8:
        graph.completeness = total_incoming_to_logit / logit_value
    else:
        graph.completeness = float("nan")

    return graph
