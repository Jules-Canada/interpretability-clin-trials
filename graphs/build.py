"""
graphs/build.py — Attribution graph construction.

Implements the analytical linear attribution method from:
  "Circuit Tracing: Revealing Computational Graphs in Language Models"
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Edge weight formula (§ Attribution Graphs):
  A_{s→t} = a_s × W_dec[l_s].T × J^▼_{l_s→l_t} × W_enc[l_t]

where J^▼ is the Jacobian with stop-gradients on all nonlinearities (attention
patterns, JumpReLU gates, LayerNorm/RMSNorm denominators).  J^▼ is computed via
ONE torch.autograd backward pass through a frozen-nonlinearity forward — the
same chain rule the paper specifies, no manual per-architecture derivation.

Frozen via TransformerLens hooks (cached values substituted, .detach()'d):
  - attn.hook_pattern at every layer
  - hook_scale on every LayerNorm/RMSNorm: ln1, ln2 on every architecture;
    ln1_post, ln2_post on Gemma 3; ln_final.  Anything new (e.g. ln3 on a
    future architecture) flows through automatically — every *.hook_scale
    that exists in the cache is frozen.

Detached:
  - hook_mlp_out at every layer.  Severs the autograd path back through the
    MLP so ∂logit/∂hook_resid_pre[l] only flows along skip + attention,
    matching the paper's linearised replacement model where MLPs are replaced
    by CLT decoder outputs (and the actual MLP only contributes via the
    constant error term).

Five node and edge types (same as the paper):
  - feature   → logit:    a_s × Σ_{l_t ≥ l_s} mlp_rms[l_t] · W_dec[l_s→l_t][:, f] · grad_at_mlp_post[l_t]
  - feature   → feature:  a_s × T[l_s→l_t][f_s] · W_enc[l_t][f_t]
                          where T[l_s→l_t] = Σ_{l''} mlp_rms[l''] · W_dec[l_s→l''].T @ W_out[l'']
  - embedding → feature:  embed[pos] · W_enc[l][f]
  - embedding → logit:    embed[pos] · grad_at_resid_pre[0]
  - error     → logit:    error_vec · grad_at_mlp_post[l]
  - attention → logit:    cross_attn[h] · grad_at_attn_out[l]
                          (cross_attn = total_attn[h] − a_tt[h] · self_loop[h]
                          captures cross-position attention not already in the
                          same-position self-loop baked into grad_at_resid_pre.)

For the design rationale and validation history, see docs/autograd_plan.md and
docs/diagnostics_log.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from clt.config import AttributionConfig
from clt.model import CrossLayerTranscoder


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
    # sum of all incoming edges to the logit node / (logit - b_U)
    completeness: float = 0.0


# ---------------------------------------------------------------------------
# Transfer matrices (feature → feature edges)
# ---------------------------------------------------------------------------

def _compute_transfer_matrices(
    clt: CrossLayerTranscoder,
    model: HookedTransformer,
    L: int,
    mlp_rms_per_layer: dict[int, float] | None = None,
) -> dict[tuple[int, int], Float[Tensor, "n_features d_model"]]:
    """
    T[l_s → l_t][f, :] = Σ_{l''=l_s}^{l_t-1} mlp_rms[l''] · W_dec[l_s→l''].T[f] @ W_out_{l''}

    Used only for feature → feature edges (the d_model-space projection of a
    feature's contribution to subsequent residual streams, before encoding).

    Note: this matrix does NOT include any post-MLP-norm Jacobian.  feat → feat
    edges therefore have a small approximation gap on Gemma 3; tightening it
    is open question 4 in docs/autograd_plan.md.
    """
    transfer: dict[tuple[int, int], Tensor] = {}
    device = next(clt.parameters()).device
    F = clt.cfg.n_features
    d_model = clt.cfg.d_model

    with torch.no_grad():
        for l_source in range(L):
            cumulative = torch.zeros(F, d_model, device=device)
            for offset, decoder in enumerate(clt.decoders[l_source]):
                l_intermediate = l_source + offset
                W_dec_T = decoder.weight.T                                # (F, d_mlp)
                W_out = model.blocks[l_intermediate].mlp.W_out            # (d_mlp, d_model)
                rms = mlp_rms_per_layer[l_intermediate] if mlp_rms_per_layer else 1.0
                cumulative = cumulative + rms * (W_dec_T @ W_out)
                transfer[(l_source, l_intermediate + 1)] = cumulative.clone()
            transfer[(l_source, L)] = cumulative.clone()

    return transfer


# ---------------------------------------------------------------------------
# Attribution gradients (autograd backward through frozen-nonlinearity forward)
# ---------------------------------------------------------------------------

def _compute_attribution_gradients(
    model: HookedTransformer,
    cache,
    tokens: Float[Tensor, "1 seq"],
    target_position: int,
    target_token_idx: int,
    L: int,
) -> dict[str, list[Tensor]]:
    """
    Return the per-layer gradients of target_logit through the linearised model:

        grad['resid_pre'][l]  : ∂target_logit / ∂hook_resid_pre[l]
                                for l in 0..L  (l=L is the final hook_resid_post)
                                shape (d_model,)

        grad['attn_out'][l]   : ∂target_logit / ∂hook_attn_out[l]
                                for l in 0..L-1
                                shape (d_model,)
                                (post-attn-norm Jacobian baked in by autograd
                                if Gemma 3, since ln1_post is in the graph
                                with frozen scale.)

        grad['mlp_post'][l]   : ∂target_logit / ∂hook_mlp_post[l]
                                for l in 0..L-1
                                shape (d_mlp,)
                                (Derived after autograd, since the mlp_out
                                detach severs the autograd path to mlp_post.)

    All tensors are returned as CPU float64 — matches downstream edge-weight
    accumulation precision in build_attribution_graph.
    """
    # ---- Build the frozen-nonlinearity hook list -----------------------

    def freeze_to_cached(name: str):
        """Hook that replaces the live value at `name` with its cached .detach()."""
        cached = cache[name].detach()

        def hook(value, *, hook, _c=cached):
            return _c

        return name, hook

    def detach_in_place(value, *, hook):
        return value.detach()

    fwd_hooks: list[tuple[str, callable]] = []

    # 1. Frozen attention pattern at every layer
    for l in range(L):
        fwd_hooks.append(freeze_to_cached(f"blocks.{l}.attn.hook_pattern"))

    # 2. Frozen LN/RMSNorm scale (denominator) at every position it appears.
    #    ln1, ln2, ln_final exist on every architecture; ln1_post and ln2_post
    #    are Gemma-3-only and are skipped silently if absent from cache.
    for l in range(L):
        for sub in ("ln1", "ln2", "ln1_post", "ln2_post"):
            key = f"blocks.{l}.{sub}.hook_scale"
            if key in cache.cache_dict:
                fwd_hooks.append(freeze_to_cached(key))
    if "ln_final.hook_scale" in cache.cache_dict:
        fwd_hooks.append(freeze_to_cached("ln_final.hook_scale"))

    # 3. Detach hook_mlp_out — severs autograd's MLP→residual link so that the
    #    backward pass at hook_resid_pre matches the paper's linearised model
    #    (skip + attention only; MLP is replaced by CLT, which lives outside
    #    this autograd graph).
    for l in range(L):
        fwd_hooks.append((f"blocks.{l}.hook_mlp_out", detach_in_place))

    # ---- Capture intermediates we want gradients for -------------------

    captured: dict[tuple[str, int], Tensor] = {}

    def make_capture(key: tuple[str, int]):
        def hook(value, *, hook):
            captured[key] = value
            return value

        return hook

    for l in range(L):
        fwd_hooks.append((f"blocks.{l}.hook_resid_pre", make_capture(("resid_pre", l))))
        fwd_hooks.append((f"blocks.{l}.hook_attn_out",  make_capture(("attn_out",  l))))
    # Use the final hook_resid_post as resid_pre[L] — it's the input to ln_final.
    fwd_hooks.append((f"blocks.{L-1}.hook_resid_post", make_capture(("resid_pre", L))))

    # ---- Frozen forward + autograd backward in one shot ----------------

    model.eval()
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)

    # Resolve negative target_position to a positive index for indexing the logit.
    seq_len = tokens.shape[1]
    pos_abs = target_position if target_position >= 0 else seq_len + target_position
    target_logit = logits[0, pos_abs, target_token_idx]

    # torch.autograd.grad computes only the requested gradients and does not
    # accumulate to .grad — leaves the model state untouched.
    capture_keys = list(captured.keys())
    capture_tensors = [captured[k] for k in capture_keys]
    grads_tuple = torch.autograd.grad(target_logit, capture_tensors)
    grad_by_key = dict(zip(capture_keys, grads_tuple))

    # ---- Slice to target position, transfer to CPU float64 -------------

    def to_target_cpu64(t: Tensor) -> Tensor:
        # t is (1, seq, ...); take target position and promote.
        return t[0, pos_abs].detach().cpu().double()

    grad_at_resid_pre = [to_target_cpu64(grad_by_key[("resid_pre", l)]) for l in range(L + 1)]
    grad_at_attn_out  = [to_target_cpu64(grad_by_key[("attn_out",  l)]) for l in range(L)]

    # ---- Derive grad_at_mlp_post manually ------------------------------
    # Forward at layer l:  resid contribution = post_mlp_norm(W_out @ mlp_post)
    # With post-norm scale frozen, post_mlp_norm is a diagonal multiply by γ/scale,
    # so the chain rule gives:
    #   ∂logit/∂mlp_post = W_out.T @ (γ/scale).detach() * ∂logit/∂resid_pre[l+1]

    grad_at_mlp_post: list[Tensor] = []
    for l in range(L):
        v_eff = grad_at_resid_pre[l + 1]                                  # (d_model,)

        post_mlp_key = f"blocks.{l}.ln2_post.hook_scale"
        if post_mlp_key in cache.cache_dict:
            scale = cache[post_mlp_key][0, pos_abs, 0].detach().cpu().double()
            gamma = model.blocks[l].ln2_post.w.detach().cpu().double()    # (d_model,)
            v_eff = v_eff * gamma / scale

        W_out = model.blocks[l].mlp.W_out.detach().cpu().double()          # (d_mlp, d_model)
        grad_at_mlp_post.append(W_out @ v_eff)                             # (d_mlp,)

    return {
        "resid_pre": grad_at_resid_pre,
        "attn_out":  grad_at_attn_out,
        "mlp_post":  grad_at_mlp_post,
    }


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

    model_device = next(model.parameters()).device
    clt = clt.to(model_device)

    L = clt.cfg.n_layers
    tokens = tokens.to(model_device)
    target_position = cfg.target_position

    # -----------------------------------------------------------------------
    # Step 1: cache forward (frozen — no grad needed for caching)
    # -----------------------------------------------------------------------
    model.eval()
    clt.eval()

    t0 = time.time()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    print(f"  [t] forward pass:       {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Step 2: CLT encode/decode + per-layer RMS scales
    # -----------------------------------------------------------------------
    t0 = time.time()
    resid_streams = [
        cache[f"blocks.{l}.hook_resid_pre"].to(model_device) for l in range(L)
    ]

    use_saved_scales = clt.cfg.normalize_activations and clt.has_scales()
    if clt.cfg.normalize_activations:
        if use_saved_scales:
            resid_streams_enc = [
                r / clt.resid_scales[l].to(r.device).clamp(min=1e-8)
                for l, r in enumerate(resid_streams)
            ]
        else:
            print("  [warn] CLT has no saved scales; falling back to per-prompt RMS "
                  "(graphs may be untrustworthy — run scripts/compute_clt_scales.py)",
                  flush=True)
            resid_streams_enc = []
            for r in resid_streams:
                rms = r.float().pow(2).mean().sqrt().clamp(min=1e-8)
                resid_streams_enc.append(r / rms)
    else:
        resid_streams_enc = resid_streams

    with torch.no_grad():
        feature_acts = clt.encode(resid_streams_enc)
        mlp_recons = clt.decode(feature_acts)
    print(f"  [t] CLT encode/decode:  {time.time()-t0:.2f}s", flush=True)

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
        print(f"  [debug] mlp_rms ({src}) "
              f"L0={mlp_rms_per_layer[0]:.3f}  L{mid}={mlp_rms_per_layer[mid]:.3f}  L{L-1}={mlp_rms_per_layer[L-1]:.3f}",
              flush=True)

    # -----------------------------------------------------------------------
    # Step 3: attribution gradients via one autograd backward pass
    # -----------------------------------------------------------------------
    t0 = time.time()
    grads = _compute_attribution_gradients(
        model, cache, tokens, target_position, target_token_idx, L,
    )
    print(f"  [t] autograd gradients: {time.time()-t0:.2f}s", flush=True)

    # Per-feature scalar contributions to the logit (feat → logit edge weights):
    #   corrected[l_s][f] = Σ_{l_t ≥ l_s} mlp_rms[l_t] · W_dec[l_s→l_t].T @ grad_at_mlp_post[l_t]
    t0 = time.time()
    corrected_logit_transfer: dict[int, Tensor] = {}
    with torch.no_grad():
        for l_source in range(L):
            acc = torch.zeros(clt.cfg.n_features, dtype=torch.float64)
            for offset, decoder in enumerate(clt.decoders[l_source]):
                l_t = l_source + offset
                W_dec_T = decoder.weight.T.detach().cpu().double()        # (F, d_mlp)
                rms = mlp_rms_per_layer[l_t] if mlp_rms_per_layer else 1.0
                acc = acc + rms * (W_dec_T @ grads["mlp_post"][l_t])
            corrected_logit_transfer[l_source] = acc

    transfer = _compute_transfer_matrices(clt, model, L, mlp_rms_per_layer)
    print(f"  [t] transfer matrices:  {time.time()-t0:.2f}s", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: token + position embedding (for embedding nodes)
    # -----------------------------------------------------------------------
    token_embed = cache["hook_embed"]
    if "hook_pos_embed" in cache.cache_dict:
        x_embed = (token_embed + cache["hook_pos_embed"]).to(model_device)
    else:
        x_embed = token_embed.to(model_device)

    seq_len = tokens.shape[1]
    if target_position < 0:
        target_position = seq_len + target_position

    # -----------------------------------------------------------------------
    # Step 5: target logit value + sanity check on the autograd readout
    # -----------------------------------------------------------------------
    with torch.no_grad():
        logits = model(tokens)
    logit_value = logits[0, target_position, target_token_idx].item()

    b_U_val = 0.0
    if hasattr(model, "b_U") and model.b_U is not None:
        b_U_val = model.b_U[target_token_idx].item()
    decomposable_logit = logit_value - b_U_val

    # autograd's grad at resid_post[L-1] is the readout vector v.  Verify
    # v · r_L == decomposable_logit (exact under frozen ln_final scale).
    v_autograd = grads["resid_pre"][L]                                     # cpu float64
    with torch.no_grad():
        r_L = cache[f"blocks.{L-1}.hook_resid_post"][0, target_position].cpu().double()
    v_dot_rL = (v_autograd * r_L).sum().item()
    print(f"  [debug] v.norm={v_autograd.norm():.4f}  "
          f"v·r_L={v_dot_rL:.4f}  decomposable={decomposable_logit:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # Step 6: build nodes and edges
    # -----------------------------------------------------------------------
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
    print(f"  [debug] seq_len={seq_len}  n_active_at_target={n_active_at_target}  "
          f"logit={logit_value:.4f}", flush=True)

    graph = AttributionGraph(
        tokens=str_tokens,
        target_token=target_token_str,
        target_position=target_position,
        logit_value=logit_value,
    )

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

    W_enc = [clt.encoders[l].weight for l in range(L)]

    total_incoming_to_logit = 0.0
    _debug_feat_logit_sum = 0.0
    _debug_error_logit_sum = 0.0
    _debug_embed_logit_sum = 0.0
    _debug_attn_logit_sum = 0.0

    # ---- Feature nodes + feature→logit + feature→feature edges ---------
    t0 = time.time()
    transfer_cpu = {k: v.cpu().float() for k, v in transfer.items()}
    W_enc_cpu = [w.cpu().float() for w in W_enc]

    for l_s in range(L):
        for pos in range(seq_len):
            if pos != target_position:
                continue
            a = feature_acts[l_s][0, pos].detach().cpu().float()
            active_mask = a.abs() > cfg.min_activation
            active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            active_idx_list = active_indices.tolist()
            a_vals = a[active_indices]

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

                weight = a_sf * corrected_logit_transfer[l_s][f_s].item()
                graph.edges.append({
                    "source": node_id,
                    "target": logit_node_id,
                    "weight": weight,
                })
                total_incoming_to_logit += weight
                _debug_feat_logit_sum += weight

            # Vectorised feature → feature edges
            for l_t in range(l_s + 1, L):
                if (l_s, l_t) not in transfer_cpu:
                    continue

                a_t = feature_acts[l_t][0, pos].detach().cpu().float()
                active_t_mask = a_t.abs() > cfg.min_activation
                active_t_indices = active_t_mask.nonzero(as_tuple=False).squeeze(-1)
                if active_t_indices.numel() == 0:
                    continue

                T_rows = transfer_cpu[(l_s, l_t)][active_indices]
                W_rows = W_enc_cpu[l_t][active_t_indices]
                weights_mat = a_vals.unsqueeze(1) * (T_rows @ W_rows.T)

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

    # ---- Embedding nodes + embedding→logit + embedding→feature ---------
    t0 = time.time()
    grad_at_resid_pre0 = grads["resid_pre"][0]                             # cpu float64

    for pos in range(seq_len):
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

        if pos != target_position:
            continue

        # embedding → logit: skip-connection path from resid_pre[0] through the
        # linearised attention chain to the logit.
        embed_logit_weight = (grad_at_resid_pre0 @ embed_vec.cpu().double()).item()
        graph.edges.append({
            "source": embed_node_id,
            "target": logit_node_id,
            "weight": embed_logit_weight,
        })
        total_incoming_to_logit += embed_logit_weight
        _debug_embed_logit_sum += embed_logit_weight

        embed_vec_cpu = embed_vec.cpu().float()
        for l in range(L):
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

    # ---- Error nodes + error→logit edges ------------------------------
    t0 = time.time()
    for l in range(L):
        true_post = cache[f"blocks.{l}.mlp.hook_post"].to(model_device)
        clt_recon = mlp_recons[l]

        if mlp_rms_per_layer is not None:
            error = (true_post - clt_recon * mlp_rms_per_layer[l]).detach()
        else:
            error = (true_post - clt_recon).detach()

        error_vec = error[0, target_position].cpu().double()              # (d_mlp,)
        weight = (error_vec @ grads["mlp_post"][l]).item()

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

    # ---- Cross-position attention head nodes + edges ------------------
    # grad_at_resid_pre[l+1] (autograd) already includes each layer's
    # attention self-loop (same-position → same-position) via the frozen
    # attention pattern.  What's still missing is cross-position attention,
    # i.e. Σ_{j≠tgt} A_h[tgt, j] · (W_V @ resid_j) @ W_O.  These nodes capture
    # that term, with the post-attn-norm Jacobian baked into grad_at_attn_out
    # automatically (autograd handles ln1_post on Gemma 3 via the frozen scale).
    t0 = time.time()
    for l in range(L):
        a_tt = cache[f"blocks.{l}.attn.hook_pattern"][0, :, target_position, target_position].cpu().double()
        z_all = cache[f"blocks.{l}.attn.hook_z"][0, target_position].cpu().double()      # (n_kv_heads, d_head)
        W_O = model.blocks[l].attn.W_O.cpu().double()                                     # (n_heads, d_head, d_model)

        # Per-head total attention output at the target position.
        total_attn = torch.einsum("hd,hdm->hm", z_all, W_O)                               # (n_heads, d_model)

        # Self-loop: a_tt[h] · LN(resid[tgt]) @ W_V[h] @ W_O[h].  Use hook_v
        # (post-LN value vectors) to avoid recomputing LN; expand for GQA.
        v_h_all = cache[f"blocks.{l}.attn.hook_v"][0, target_position].cpu().double()    # (n_kv_heads, d_head)
        n_kv = v_h_all.shape[0]
        n_q  = W_O.shape[0]
        if n_kv != n_q:
            v_h_all = v_h_all.repeat_interleave(n_q // n_kv, dim=0)                       # (n_heads, d_head)
        self_loop = a_tt.unsqueeze(1) * torch.einsum("hd,hdm->hm", v_h_all, W_O)          # (n_heads, d_model)

        cross_attn = total_attn - self_loop                                               # (n_heads, d_model)

        # grad_at_attn_out[l] = ∂logit/∂hook_attn_out[l].  On Gemma 3 the
        # post-attn-norm sits between hook_attn_out and the residual addition;
        # autograd has already pulled the gradient back through it (via the
        # frozen ln1_post scale) so no manual γ/scale is needed here.
        v_for_cross = grads["attn_out"][l]                                                # (d_model,) cpu float64
        logit_contribs = torch.einsum("m,hm->h", v_for_cross, cross_attn)                 # (n_heads,)

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
    # Step 7: completeness check
    # -----------------------------------------------------------------------
    print(f"  [debug] logit_value={logit_value:.4f}  b_U={b_U_val:.4f}  "
          f"decomposable={decomposable_logit:.4f}", flush=True)
    print(f"  [debug] feat_sum={_debug_feat_logit_sum:.4f}  "
          f"error_sum={_debug_error_logit_sum:.4f}  "
          f"attn_sum={_debug_attn_logit_sum:.4f}  "
          f"embed_sum={_debug_embed_logit_sum:.4f}  "
          f"total={total_incoming_to_logit:.4f}", flush=True)

    if abs(decomposable_logit) > 1e-8:
        graph.completeness = total_incoming_to_logit / decomposable_logit
    else:
        graph.completeness = float("nan")

    return graph
