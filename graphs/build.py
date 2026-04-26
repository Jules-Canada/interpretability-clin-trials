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

import time
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
    mlp_rms_per_layer: dict[int, float] | None = None,
) -> dict[tuple[int, int], Float[Tensor, "n_features d_model"]]:
    """
    Precompute transfer matrices T[l_s → l_t] for all valid (l_s, l_t) pairs.

    T[l_s → l_t] ∈ ℝ^{F × d_model}:
        T[l_s → l_t][f, :] = Σ_{l''=l_s}^{l_t-1} mlp_rms[l''] * W_dec[l_s→l''].weight.T[f,:] @ W_out_{l''}

    The mlp_rms factor corrects for activation normalization: the CLT decoder was trained to
    reconstruct mlp_post / mlp_rms, so its output is in normalized space. Multiplying by
    mlp_rms converts back to the raw-space contribution to the residual stream.

    Args:
        clt:               trained CrossLayerTranscoder
        model:             HookedTransformer (used for W_out matrices)
        L:                 number of layers
        mlp_rms_per_layer: per-layer RMS of raw mlp_post activations (from prompt cache).
                           Required when clt.cfg.normalize_activations is True.

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

                # Scale by mlp_rms to convert decoder output (normalized space) back to
                # the raw-space MLP contribution to the residual stream.
                rms = mlp_rms_per_layer[l_intermediate] if mlp_rms_per_layer else 1.0

                # (F, d_mlp) @ (d_mlp, d_model) = (F, d_model)
                cumulative = cumulative + rms * (W_dec_T @ W_out)

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

    t0 = time.time()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    print(f"  [t] forward pass:       {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Step 2: Get CLT feature activations from cached residual streams
    # -----------------------------------------------------------------------
    t0 = time.time()
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
    print(f"  [t] CLT encode/decode:  {time.time()-t0:.2f}s", flush=True)

    # Per-layer RMS of raw mlp_post — used to un-normalize the CLT decoder outputs.
    # The CLT was trained to reconstruct mlp_post / rms, so decoder output is in
    # normalized space; multiplying by rms converts back to raw space for error and
    # transfer matrix computations.
    mlp_rms_per_layer: dict[int, float] | None = None
    if clt.cfg.normalize_activations:
        mlp_rms_per_layer = {}
        with torch.no_grad():
            for l in range(L):
                raw = cache[f"blocks.{l}.mlp.hook_post"]
                mlp_rms_per_layer[l] = raw.float().pow(2).mean().sqrt().clamp(min=1e-8).item()
        print(f"  [debug] mlp_rms L0={mlp_rms_per_layer[0]:.3f}  L16={mlp_rms_per_layer[16]:.3f}  L33={mlp_rms_per_layer[33]:.3f}", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Compute readout vector v = ∂T/∂resid_L
    # -----------------------------------------------------------------------
    t0 = time.time()
    # (d_model,)
    v = _compute_readout_vector(model, cache, target_position, target_token_idx, L)
    print(f"  [t] readout vector:     {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Precompute transfer matrices
    # -----------------------------------------------------------------------
    t0 = time.time()
    # transfer[(l_s, l_t)]: (n_features, d_model)
    transfer = _compute_transfer_matrices(clt, model, L, mlp_rms_per_layer)
    print(f"  [t] transfer matrices:  {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Step 5: Compute token embedding at each position
    # -----------------------------------------------------------------------
    # (1, seq, d_model)
    token_embed = cache["hook_embed"]
    # Pythia uses RoPE (no absolute position embedding in the residual stream).
    # For models with learned position embeddings, add hook_pos_embed.
    if "hook_pos_embed" in cache.cache_dict:
        pos_embed = cache["hook_pos_embed"]
        x_embed = (token_embed + pos_embed).to(model_device)
    else:
        x_embed = token_embed.to(model_device)

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

    n_active_at_target = sum(
        (feature_acts[l][0, target_position].abs() > cfg.min_activation).sum().item()
        for l in range(L)
    )
    print(f"  [debug] seq_len={seq_len}  n_active_at_target={n_active_at_target}  logit={logit_value:.4f}", flush=True)

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
    _debug_feat_logit_sum = 0.0
    _debug_error_logit_sum = 0.0

    # -----------------------------------------------------------------------
    # Feature nodes + edges
    # Vectorized: for each (l_s, pos, l_t), compute all active→active edge
    # weights in one batch matmul rather than looping over feature pairs.
    # Math is identical to the paper formula; this is a performance optimization.
    # -----------------------------------------------------------------------
    # Pre-move tensors to CPU for vectorized ops (avoids repeated MPS↔CPU transfers)
    t0 = time.time()
    transfer_cpu = {k: v.cpu() for k, v in transfer.items()}
    W_enc_cpu = [w.cpu() for w in W_enc]
    v_cpu = v.cpu()
    print(f"  [t] GPU→CPU transfer:   {time.time()-t0:.2f}s  ({len(transfer_cpu)} matrices)", flush=True)
    t0 = time.time()

    for l_s in range(L):
        for pos in range(seq_len):
            # (F,)
            a = feature_acts[l_s][0, pos].detach().cpu()
            active_mask = a.abs() > cfg.min_activation
            active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            active_idx_list = active_indices.tolist()
            # (n_active_s,)
            a_vals = a[active_indices]

            # -- Feature nodes --
            for f_s in active_idx_list:
                a_sf = a[f_s].item()
                node_id = f"feat_l{l_s}_p{pos}_f{f_s}"
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
                    T_vec = transfer_cpu[(l_s, L)][f_s]
                    weight = a_sf * (T_vec @ v_cpu).item()
                    graph.edges.append({
                        "source": node_id,
                        "target": logit_node_id,
                        "weight": weight,
                    })
                    total_incoming_to_logit += weight
                    _debug_feat_logit_sum += weight

            # -- Feature → feature edges (vectorized over source × target features) --
            # Only compute at target_position: features at other positions have no
            # path to the logit node (cross-position paths go through frozen attention,
            # not through the linear MLP→residual→MLP chain we trace here).
            if pos != target_position:
                continue
            for l_t in range(l_s + 1, L):
                if (l_s, l_t) not in transfer_cpu:
                    continue

                a_t = feature_acts[l_t][0, pos].detach().cpu()
                active_t_mask = a_t.abs() > cfg.min_activation
                active_t_indices = active_t_mask.nonzero(as_tuple=False).squeeze(-1)
                if active_t_indices.numel() == 0:
                    continue

                # (n_active_s, d_model)
                T_rows = transfer_cpu[(l_s, l_t)][active_indices]
                # (n_active_t, d_model)
                W_rows = W_enc_cpu[l_t][active_t_indices]

                # weights[i,j] = a_vals[i] * (T_rows[i] · W_rows[j])
                # (n_active_s, n_active_t)
                enc_proj = T_rows @ W_rows.T
                weights_mat = a_vals.unsqueeze(1) * enc_proj

                # Use list comprehensions (10-20x faster than explicit for-loop over pairs).
                # Threshold at 1e-4 rather than 1e-8 to prune near-zero edges that
                # contribute nothing to completeness but dominate memory and runtime.
                nonzero_mask = weights_mat.abs() > 1e-4
                if not nonzero_mask.any():
                    continue
                ii_arr, jj_arr = nonzero_mask.nonzero(as_tuple=True)
                w_vals = weights_mat[nonzero_mask].tolist()
                ii_list = ii_arr.tolist()
                jj_list = jj_arr.tolist()
                t_idx_list = active_t_indices.tolist()
                graph.edges.extend([
                    {
                        "source": f"feat_l{l_s}_p{pos}_f{active_idx_list[ii]}",
                        "target": f"feat_l{l_t}_p{pos}_f{t_idx_list[jj]}",
                        "weight": w,
                    }
                    for ii, jj, w in zip(ii_list, jj_list, w_vals)
                ])

    print(f"  [t] feature nodes+edges:{time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Embedding nodes + edges
    # -----------------------------------------------------------------------
    t0 = time.time()
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

        # Embedding → feature edges at every layer (vectorized)
        embed_vec_cpu = embed_vec.cpu()
        for l in range(L):
            # (F,) = (F, d_model) @ (d_model,)
            contributions = W_enc_cpu[l] @ embed_vec_cpu

            a_l = feature_acts[l][0, pos].detach().cpu()
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

    print(f"  [t] embedding edges:    {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Error nodes + edges (reconstruction error → logit)
    # -----------------------------------------------------------------------
    t0 = time.time()
    for l in range(L):
        # true_post[l]: (1, seq, d_mlp)
        true_post = cache[f"blocks.{l}.mlp.hook_post"].to(model_device)
        # clt_recon[l]: (1, seq, d_mlp) — in normalized space if normalize_activations=True
        clt_recon = mlp_recons[l]

        # Un-normalize: clt_recon is mlp_post/rms so multiply by rms to get raw space,
        # then subtract from raw true_post to get the true reconstruction error.
        if mlp_rms_per_layer is not None:
            error = (true_post - clt_recon * mlp_rms_per_layer[l]).detach()
        else:
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
        _debug_error_logit_sum += weight

    print(f"  [t] error edges:        {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Completeness check
    # -----------------------------------------------------------------------
    print(f"  [debug] logit_value={logit_value:.4f}  feat_sum={_debug_feat_logit_sum:.4f}  error_sum={_debug_error_logit_sum:.4f}  total={total_incoming_to_logit:.4f}", flush=True)

    # completeness = (sum of incoming edges to logit node) / logit_value
    # Should be close to 1.0 for a faithful attribution
    if abs(logit_value) > 1e-8:
        graph.completeness = total_incoming_to_logit / logit_value
    else:
        graph.completeness = float("nan")

    return graph
