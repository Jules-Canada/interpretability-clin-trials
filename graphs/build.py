"""
graphs/build.py — Attribution graph construction.

Implements the analytical linear attribution method from:
  "Circuit Tracing: Revealing Computational Graphs in Language Models"
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Edge weight formula (§ Attribution Graphs):
  A_{s→t} = a_s × Σ_{ℓ_s ≤ ℓ < ℓ_t} W_dec[l_s→l].T × J^▼_{l→l_t} × W_enc[l_t]

where J^▼ is the Jacobian with stop-gradients on all nonlinearities (attention patterns,
JumpReLU gates, LayerNorm denominators).

Five node types:
  - feature:   active CLT features (l, f, position)
  - embedding: token + positional embedding at each sequence position
  - error:     reconstruction error (true MLP output − CLT reconstruction) per layer
  - attention: cross-position attention head output (l, h) — the part not in self-loop
  - logit:     target token's output logit

Five edge types:
  - feature   → logit:   a_s × corrected_logit_transfer[l_s][f_s]
  - feature   → feature: a_s × transfer[l_s→l_t][f_s] @ W_enc[l_t][f_t]
  - embedding → feature: x_embed[p] · W_enc[l][f]
  - error     → logit:   error_vec @ effective_readout[l]
  - attention → logit:   v_{l+1} · (hook_z[h]@W_O[h] − self_loop[h])

Attention Jacobians (required for completeness ≥ 0.5):
  With frozen attention pattern A_h, each layer propagates residual deltas through
  both the skip connection and attention. The effective readout vector at layer l is
  backpropagated from v = ∂logit/∂resid_L:

    v_L = v
    v_l = (I + J_l)^T @ v_{l+1}
    where J_l = Σ_h A_h[target,target] * W_V^h @ W_O^h   (d_model × d_model)

  The effective readout in MLP output space at layer l:
    effective_readout[l] = W_out[l] @ v_{l+1}            (d_mlp,)

  Feature → logit transfer (scalar per feature, includes attention):
    corrected_logit_transfer[l_s][f] =
        Σ_{l_t ≥ l_s} rms[l_t] * W_dec[l_s→l_t][:, f] · effective_readout[l_t]

  Feature → feature transfer uses the MLP-only T matrix (attention paths between
  intermediate layers are not yet included — a known approximation).
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
# Attention Jacobian propagation
# ---------------------------------------------------------------------------

def _compute_attention_propagated_v(
    model: HookedTransformer,
    cache: dict,
    v: Float[Tensor, "d_model"],
    target_position: int,
    L: int,
) -> list[Float[Tensor, "d_model"]]:
    """
    Backpropagate v through frozen attention Jacobians to get the effective
    readout vector at each layer.

    v_at_layer[L] = v  (gradient of logit w.r.t. final residual)
    v_at_layer[l] = (I + J_l)^T @ v_at_layer[l+1]

    where J_l = Σ_h A_h[target, target] * W_V^h @ W_O^h   captures how a
    residual delta at the target position at the start of layer l propagates
    forward to the start of layer l+1 via the same-position attention self-loop.

    Args:
        model:           HookedTransformer (W_V, W_O accessed per layer)
        cache:           activation cache containing frozen attention patterns
        v:               readout vector (d_model,) from _compute_readout_vector
        target_position: sequence position being traced
        L:               number of layers

    Returns:
        list of length L+1; v_at_layer[l] has shape (d_model,), float32
    """
    v_at_layer: list[Tensor | None] = [None] * (L + 1)
    # float64 on CPU: early-layer v can grow (1+σ)^L ≈ 8000× — feat_sum and error_sum
    # accumulate large terms that nearly cancel. float32 loses precision there; float64
    # handles it. MPS doesn't support float64, so we always run this on CPU.
    v_at_layer[L] = v.cpu().double()

    with torch.no_grad():
        for l in range(L - 1, -1, -1):
            # Frozen attention pattern — (n_heads, seq, seq)
            # Self-attention weight at target position: (n_heads,)
            a_tt = cache[f"blocks.{l}.attn.hook_pattern"][0, :, target_position, target_position].cpu().double()

            # W_V: (n_kv_heads, d_model, d_head)  — may be < n_heads under GQA
            # W_O: (n_heads,    d_head,  d_model)
            W_V = model.blocks[l].attn.W_V.cpu().double()
            W_O = model.blocks[l].attn.W_O.cpu().double()

            # GQA: expand W_V from n_kv_heads to n_heads so einsums align.
            n_heads_l   = W_O.shape[0]
            n_kv_heads_l = W_V.shape[0]
            if n_kv_heads_l != n_heads_l:
                W_V = W_V.repeat_interleave(n_heads_l // n_kv_heads_l, dim=0)

            v_curr = v_at_layer[l + 1]

            # Post-attention norm (Gemma 3 only): applied after attn_out, before adding
            # to residual.  J_ln1_post^T = diag(γ / scale_post_attn).  The skip
            # connection bypasses this, so we only apply it on the attention path.
            post_attn_scale_key = f"blocks.{l}.ln1_post.hook_scale"
            if post_attn_scale_key in cache.cache_dict:
                scale_post_attn = cache[post_attn_scale_key][0, target_position, 0].cpu().double()
                gamma_post_attn = model.blocks[l].ln1_post.w.cpu().double()
                v_for_attn = v_curr * gamma_post_attn / scale_post_attn
            else:
                v_for_attn = v_curr

            # Pre-attention LayerNorm Jacobian (frozen denominator).
            ln_scale = cache[f"blocks.{l}.ln1.hook_scale"][0, target_position, 0].cpu().double()
            if hasattr(model.blocks[l].ln1, 'w'):
                # RMSNorm (Gemma): no centering, learnable weight γ
                ln_weight = model.blocks[l].ln1.w.cpu().double()  # (d_model,)
                v_curr_ln = v_for_attn * ln_weight / ln_scale
            else:
                # LayerNormPre (Pythia): no learnable weight, centering
                v_curr_ln = (v_for_attn - v_for_attn.mean()) / ln_scale

            # J_l^T @ v_curr = Σ_h a_tt[h] * (W_V[h] @ W_O[h])^T @ J_LN^T @ v_curr
            # (n_heads, d_head): W_V[h]^T @ (J_LN^T @ v_curr)
            v_val = torch.einsum("hmd,m->hd", W_V, v_curr_ln)
            # (n_heads, d_model): v_val[h] @ W_O[h] for each head
            v_out = torch.einsum("hdm,hd->hm", W_O, v_val)
            # (d_model,): Σ_h a_tt[h] * v_out[h]
            J_T_v = torch.einsum("h,hm->m", a_tt, v_out)

            # P_l^T @ v_curr = v_curr + J_l^T @ v_curr  (skip connection + attention)
            v_at_layer[l] = v_curr + J_T_v

    return v_at_layer  # type: ignore[return-value]


def _compute_corrected_logit_transfer(
    clt: CrossLayerTranscoder,
    model: HookedTransformer,
    v_at_layer: list[Float[Tensor, "d_model"]],
    L: int,
    mlp_rms_per_layer: dict[int, float] | None = None,
    cache: dict | None = None,
    target_position: int = -1,
) -> dict[int, Float[Tensor, "n_features"]]:
    """
    Compute per-feature scalar contributions to the logit, including attention paths.

    For feature f at source layer l_s:
        corrected_T[l_s][f] = Σ_{l_t=l_s}^{L-1}
            rms[l_t] * W_dec[l_s→l_t][:, f] · (W_out[l_t] @ v_at_layer[l_t+1])

    where v_at_layer[l_t+1] is the effective readout vector at layer l_t+1,
    backpropagated through all subsequent attention layers.

    Compare to the MLP-only version which uses v (= v_at_layer[L]) for all l_t.

    Args:
        clt:              trained CrossLayerTranscoder
        model:            HookedTransformer (W_out accessed per layer)
        v_at_layer:       output of _compute_attention_propagated_v
        L:                number of layers
        mlp_rms_per_layer: per-layer normalization scale (same as in transfer matrix)

    Returns:
        dict mapping l_source → (n_features,) float32 tensor
    """
    device = next(clt.parameters()).device

    # Precompute effective readout in MLP output space at each layer:
    # effective_readout[l] = W_out[l] @ v_at_layer[l+1]   (d_mlp,)
    # This is the direction in MLP-output space that contributes to the logit,
    # accounting for all attention layers after l.
    # float64 on CPU throughout: effective_readouts inherit the large magnitude of early-layer
    # v_at_layer vectors; accumulating in float64 prevents precision loss when feat_sum
    # and error_sum are large but nearly equal and opposite.
    effective_readouts: list[Tensor] = []
    with torch.no_grad():
        for l in range(L):
            # (d_mlp, d_model) on CPU float64
            W_out = model.blocks[l].mlp.W_out.cpu().double()
            v_eff = v_at_layer[l + 1]  # (d_model,) float64

            # Post-MLP norm (Gemma 3 only): applied after mlp_out, before adding to
            # residual.  Frozen Jacobian = diag(γ_ln2_post / scale_ln2_post).
            # ∂logit/∂mlp_out = J_ln2_post^T @ v_at_layer[l+1] = γ/scale * v
            if cache is not None:
                post_mlp_scale_key = f"blocks.{l}.ln2_post.hook_scale"
                if post_mlp_scale_key in cache.cache_dict:
                    scale_post_mlp = cache[post_mlp_scale_key][0, target_position, 0].cpu().double()
                    gamma_post_mlp = model.blocks[l].ln2_post.w.cpu().double()
                    v_eff = v_eff * gamma_post_mlp / scale_post_mlp

            # (d_mlp,) = (d_mlp, d_model) @ (d_model,)
            eff = W_out @ v_eff
            effective_readouts.append(eff)

    corrected: dict[int, Tensor] = {}
    with torch.no_grad():
        for l_source in range(L):
            # (n_features,) float64 on CPU — accumulates scalar logit contribution per feature
            acc = torch.zeros(clt.cfg.n_features, dtype=torch.float64)
            for offset, decoder in enumerate(clt.decoders[l_source]):
                l_t = l_source + offset
                # W_dec[l_source→l_t].weight: (d_mlp, n_features) → .T: (n_features, d_mlp)
                W_dec_T = decoder.weight.T.cpu().double()
                rms = mlp_rms_per_layer[l_t] if mlp_rms_per_layer else 1.0
                # (n_features,) = (n_features, d_mlp) @ (d_mlp,)
                acc = acc + rms * (W_dec_T @ effective_readouts[l_t])
            corrected[l_source] = acc

    return corrected, effective_readouts


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
    Compute v = ∂logit/∂resid_L using a frozen-denominator LayerNorm gradient.

    Autograd through LN/RMSNorm gives v_auto · r_L = 0 because LN is degree-0
    homogeneous (Euler's theorem: x · ∇f(x) = 0 for f(cx)=f(x)).  We must freeze
    the denominator (hook_scale) and treat it as a constant:

      LayerNormPre (Pythia): v = (W_U[:, tok] − mean(W_U[:, tok])) / hook_scale
      RMSNorm      (Gemma):  v = W_U[:, tok] * ln_final.w / hook_scale

    Then  v · r_L = logit − b_U[tok]  (exactly, under frozen hook_scale).

    hook_scale is the cached std (LayerNorm) or rms (RMSNorm) of the final residual
    at this target position — the same value that was used in the forward pass.

    Args:
        model:            HookedTransformer
        cache:            activation cache from run_with_cache (must include ln_final hooks)
        target_position:  sequence position to trace
        target_token_idx: vocabulary index of the target token
        L:                number of layers (unused; kept for API consistency)

    Returns:
        v of shape (d_model,) float32 — frozen-denominator readout vector
    """
    with torch.no_grad():
        # hook_scale: (1, seq, 1) — std (LayerNormPre) or rms (RMSNorm) at target pos
        hook_scale = cache["ln_final.hook_scale"][0, target_position, 0].cpu().double()

        # W_U: (d_model, d_vocab) in TransformerLens — column for target token
        W_U_col = model.W_U[:, target_token_idx].cpu().double()  # (d_model,)

        if hasattr(model.ln_final, 'w'):
            # RMSNorm (Gemma): frozen gradient = W_U * γ / rms
            ln_w = model.ln_final.w.cpu().double()  # (d_model,)
            v = W_U_col * ln_w / hook_scale
        else:
            # LayerNormPre (Pythia): frozen gradient = (W_U − mean(W_U)) / std
            v = (W_U_col - W_U_col.mean()) / hook_scale

    # Return float32; _compute_attention_propagated_v will convert to float64 for
    # the high-precision backprop through all L attention layers.
    return v.float()


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
    # Prefer the dataset-level scales bundled with the checkpoint (correct, stable);
    # fall back to per-prompt RMS only if the checkpoint predates the scale-saving
    # change. Per-prompt RMS drifts wildly across prompts and inflates feature
    # contributions by 5-10× — see CLAUDE.md "RMS scale persistence".
    use_saved_scales = clt.cfg.normalize_activations and clt.has_scales()
    if clt.cfg.normalize_activations:
        if use_saved_scales:
            resid_streams_enc = [
                r / clt.resid_scales[l].to(r.device).clamp(min=1e-8)
                for l, r in enumerate(resid_streams)
            ]
        else:
            print("  [warn] CLT has no saved scales; falling back to per-prompt RMS "
                  "(graphs may be untrustworthy — run scripts/compute_clt_scales.py)", flush=True)
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
    # transfer matrix computations. Prefer saved dataset-level scales as above.
    mlp_rms_per_layer: dict[int, float] | None = None
    if clt.cfg.normalize_activations:
        if use_saved_scales:
            mlp_rms_per_layer = {
                l: clt.mlp_scales[l].clamp(min=1e-8).item() for l in range(L)
            }
            src = "saved"
        else:
            mlp_rms_per_layer = {}
            with torch.no_grad():
                for l in range(L):
                    raw = cache[f"blocks.{l}.mlp.hook_post"]
                    mlp_rms_per_layer[l] = raw.float().pow(2).mean().sqrt().clamp(min=1e-8).item()
            src = "per-prompt"
        mid = L // 2
        print(f"  [debug] mlp_rms ({src}) L0={mlp_rms_per_layer[0]:.3f}  L{mid}={mlp_rms_per_layer[mid]:.3f}  L{L-1}={mlp_rms_per_layer[L-1]:.3f}", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Compute readout vector v = ∂T/∂resid_L
    # -----------------------------------------------------------------------
    t0 = time.time()
    # (d_model,)
    v = _compute_readout_vector(model, cache, target_position, target_token_idx, L)
    print(f"  [t] readout vector:     {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Step 3b: Backpropagate v through frozen attention Jacobians
    # v_at_layer[l] = effective readout vector at the start of layer l,
    # accounting for how attention at layers l, l+1, ..., L-1 propagates
    # a residual delta forward to the final logit.
    # -----------------------------------------------------------------------
    t0 = time.time()
    # list[L+1] of (d_model,) tensors; v_at_layer[L] == v
    v_at_layer = _compute_attention_propagated_v(model, cache, v, target_position, L)
    print(f"  [t] attn propagation:   {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Precompute transfer matrices
    # -----------------------------------------------------------------------
    t0 = time.time()
    # transfer[(l_s, l_t)]: (n_features, d_model) — MLP-only, used for feat→feat edges
    transfer = _compute_transfer_matrices(clt, model, L, mlp_rms_per_layer)
    # corrected_logit_transfer[l_s]: (n_features,) — includes attention, used for feat→logit
    # effective_readouts[l]: (d_mlp,) — W_out[l] @ v_at_layer[l+1], used for error→logit
    corrected_logit_transfer, effective_readouts = _compute_corrected_logit_transfer(
        clt, model, v_at_layer, L, mlp_rms_per_layer,
        cache=cache, target_position=target_position,
    )
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

    # b_U[tok]: unembed bias term — logit = v · r_L + b_U, so v · r_L = logit - b_U.
    # completeness denominator must be (logit - b_U), not logit.
    # Gemma/RMSNorm models have no unembed bias (b_U = 0).
    b_U_val = 0.0
    if hasattr(model, 'b_U') and model.b_U is not None:
        b_U_val = model.b_U[target_token_idx].item()
    decomposable_logit = logit_value - b_U_val

    # Verify v: v · r_L must equal decomposable_logit (exact under frozen LN).
    # If this is near-zero, _compute_readout_vector returned wrong v.
    with torch.no_grad():
        r_L = cache[f"blocks.{L-1}.hook_resid_post"][0, target_position].cpu().double()
        v_dot_rL = (v.cpu().double() * r_L).sum().item()
    print(f"  [debug] v.norm={v.cpu().double().norm():.4f}  v·r_L={v_dot_rL:.4f}  decomposable={decomposable_logit:.4f}", flush=True)

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
    _debug_embed_logit_sum = 0.0

    # -----------------------------------------------------------------------
    # Feature nodes + edges
    # Vectorized: for each (l_s, pos, l_t), compute all active→active edge
    # weights in one batch matmul rather than looping over feature pairs.
    # Math is identical to the paper formula; this is a performance optimization.
    # -----------------------------------------------------------------------
    # Pre-move tensors to CPU for vectorized ops (avoids repeated MPS↔CPU transfers)
    t0 = time.time()
    # Cast to float32 on CPU: bfloat16/float16 dot products are not reliably
    # supported by PyTorch on CPU, and we need float32 precision for edge weights.
    transfer_cpu = {k: v.cpu().float() for k, v in transfer.items()}
    W_enc_cpu = [w.cpu().float() for w in W_enc]
    v_cpu = v.cpu().float()
    # Keep float64 on CPU: edge weights are computed by dot products against these
    # tensors; converting to float32 here would lose the precision gained upstream.
    corrected_logit_transfer_cpu = {l_s: t.cpu() for l_s, t in corrected_logit_transfer.items()}
    # effective_readouts_cpu[l]: (d_mlp,) float64 — for error→logit weights
    effective_readouts_cpu = [e.cpu() for e in effective_readouts]
    print(f"  [t] GPU→CPU transfer:   {time.time()-t0:.2f}s  ({len(transfer_cpu)} matrices)", flush=True)
    t0 = time.time()

    for l_s in range(L):
        for pos in range(seq_len):
            # Only build nodes/edges at the target position — features at other
            # positions have no linear path to the logit node (cross-position
            # paths go through frozen attention which is not in our T matrix).
            if pos != target_position:
                continue
            # (F,) — cast to float32; bfloat16 CPU matmuls are not reliably supported
            a = feature_acts[l_s][0, pos].detach().cpu().float()
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
                    # Use attention-corrected scalar transfer (includes attention paths)
                    weight = a_sf * corrected_logit_transfer_cpu[l_s][f_s].item()
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

                a_t = feature_acts[l_t][0, pos].detach().cpu().float()
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

        # Embedding → feature edges: only at target_position, since feature nodes
        # only exist there (non-target positions were skipped in the feature loop).
        if pos != target_position:
            continue

        # Direct embedding → logit edge: v_0 · r_0[target]
        # This is the skip-connection path from the initial token embedding through
        # all subsequent attention self-loops to the logit (baked into v_at_layer[0]).
        # Non-target positions contribute through cross-position attention (attn_sum).
        embed_logit_weight = (v_at_layer[0] @ embed_vec.cpu().double()).item()
        graph.edges.append({
            "source": embed_node_id,
            "target": logit_node_id,
            "weight": embed_logit_weight,
        })
        total_incoming_to_logit += embed_logit_weight
        _debug_embed_logit_sum += embed_logit_weight

        embed_vec_cpu = embed_vec.cpu().float()
        for l in range(L):
            # (F,) = (F, d_model) @ (d_model,)
            contributions = W_enc_cpu[l] @ embed_vec_cpu

            a_l = feature_acts[l][0, pos].detach().cpu().float()
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

        # Error at target position: (d_mlp,) — cpu float64 to match effective_readout
        error_vec = error[0, target_position].cpu().double()

        # Error contribution to logit, including attention propagation:
        # error_vec @ W_out[l] @ v_at_layer[l+1] = error_vec @ effective_readout[l]
        # This is exact: error enters residual at l+1 and propagates through attention
        # at layers l+1, ..., L-1 before reaching the logit.
        weight = (error_vec @ effective_readouts_cpu[l]).item()

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
    # Attention head nodes + edges (cross-position attention → logit)
    # -----------------------------------------------------------------------
    # The backpropagated v_at_layer already bakes in each layer's attention
    # self-loop (same-position → same-position) via J_l = Σ_h A_h[tgt,tgt]*W_V@W_O.
    # What's still missing from completeness: cross-position attention, i.e.
    # Σ_j≠tgt A_h[tgt,j] * (W_V@resid_j)@W_O.  These nodes capture that term.
    #
    # Logit contribution of head h at layer l:
    #   v_{l+1} · (total_attn_out[h] − self_loop_attn[h])
    # where  total_attn_out[h] = hook_z[h] @ W_O[h]   (d_model,)
    #        self_loop_attn[h]  = A_h[tgt,tgt] * (resid_pre[tgt] @ W_V[h]) @ W_O[h]
    t0 = time.time()
    _debug_attn_logit_sum = 0.0

    for l in range(L):
        # (d_model,) float64 CPU — residual at target position before layer l
        resid_target = resid_streams[l][0, target_position].cpu().double()
        # (d_model,) float64 CPU — effective readout at layer l+1
        v_lp1 = v_at_layer[l + 1]

        # (n_heads,) float64 — self-attention weight at target position
        a_tt = cache[f"blocks.{l}.attn.hook_pattern"][0, :, target_position, target_position].cpu().double()

        # (n_heads, d_head) float64 — weighted value sum for each head at target pos
        z_all = cache[f"blocks.{l}.attn.hook_z"][0, target_position].cpu().double()

        # (n_heads, d_head, d_model)
        W_O = model.blocks[l].attn.W_O.cpu().double()

        # Total per-head attn output: einsum over d_head → (n_heads, d_model)
        total_attn = torch.einsum("hd,hdm->hm", z_all, W_O)

        # Self-loop: use hook_v (post-LN value vectors) — includes LayerNorm correctly.
        # hook_v[tgt, h] = LN(resid[tgt]) @ W_V[h], so a_tt[h] * hook_v[tgt, h] is the
        # exact self-loop contribution to z_h[tgt] without recomputing LN manually.
        # GQA: hook_v has shape (n_kv_heads, d_head); expand to n_heads for einsum with W_O.
        v_h_all = cache[f"blocks.{l}.attn.hook_v"][0, target_position].cpu().double()  # (n_kv_heads, d_head)
        n_kv = v_h_all.shape[0]
        n_q  = W_O.shape[0]
        if n_kv != n_q:
            v_h_all = v_h_all.repeat_interleave(n_q // n_kv, dim=0)  # (n_heads, d_head)
        self_loop = a_tt.unsqueeze(1) * torch.einsum("hd,hdm->hm", v_h_all, W_O)  # (n_heads, d_model)

        # Cross-position attn output (the missing term): (n_heads, d_model).
        # cross_attn lives in pre-post-attn-norm space (raw Σ_h hook_z @ W_O).
        # On Gemma 3, ln1_post is applied to attn_out before the residual add, so we
        # must pull v_{l+1} back through that frozen RMSNorm Jacobian (γ/scale)
        # before dotting with cross_attn — same correction as in
        # _compute_attention_propagated_v above. Pre-norm models (Pythia) skip this.
        cross_attn = total_attn - self_loop

        post_attn_scale_key = f"blocks.{l}.ln1_post.hook_scale"
        if post_attn_scale_key in cache.cache_dict:
            scale_post_attn = cache[post_attn_scale_key][0, target_position, 0].cpu().double()
            gamma_post_attn = model.blocks[l].ln1_post.w.cpu().double()
            v_for_cross = v_lp1 * gamma_post_attn / scale_post_attn
        else:
            v_for_cross = v_lp1

        # Logit contribution per head: v_for_cross · cross_attn[h] → (n_heads,)
        logit_contribs = torch.einsum("m,hm->h", v_for_cross, cross_attn)

        for h in range(n_q):
            contrib = logit_contribs[h].item()
            if abs(contrib) < 1e-8:
                continue
            node_id = f"attn_l{l}_h{h}"
            graph.nodes.append({
                "id": node_id,
                "type": "attention",
                "layer": l,
                "feature": h,
                "position": target_position,
                "activation": abs(contrib),
                "label": f"L{l}H{h}",
            })
            graph.edges.append({
                "source": node_id,
                "target": logit_node_id,
                "weight": contrib,
            })
            total_incoming_to_logit += contrib
            _debug_attn_logit_sum += contrib

    print(f"  [t] attention nodes:    {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Completeness check
    # -----------------------------------------------------------------------
    print(f"  [debug] logit_value={logit_value:.4f}  b_U={b_U_val:.4f}  decomposable={decomposable_logit:.4f}", flush=True)
    print(f"  [debug] feat_sum={_debug_feat_logit_sum:.4f}  error_sum={_debug_error_logit_sum:.4f}  attn_sum={_debug_attn_logit_sum:.4f}  embed_sum={_debug_embed_logit_sum:.4f}  total={total_incoming_to_logit:.4f}", flush=True)

    # completeness = total_incoming / (logit - b_U)
    # b_U is the unembed bias, which is not decomposed into the graph edges.
    # With a correct frozen-denominator v, total_incoming ≈ logit - b_U → completeness ≈ 1.0.
    if abs(decomposable_logit) > 1e-8:
        graph.completeness = total_incoming_to_logit / decomposable_logit
    else:
        graph.completeness = float("nan")

    return graph
