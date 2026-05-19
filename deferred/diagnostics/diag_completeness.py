"""
scripts/diag_completeness.py — Verify the completeness decomposition identity.

Checks that:
  embed_term + Σ_l mlp_term[l] + Σ_l cross_attn_term[l] == decomposable_logit

Runs on Pythia-70m (CPU, no CLT needed) to isolate the gradient math from
CLT reconstruction quality.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformer_lens import HookedTransformer

from graphs.build import _compute_attribution_gradients


def main():
    model = HookedTransformer.from_pretrained("pythia-70m", device="cpu")
    model.eval()

    prompt = "The capital of France is"
    tokens = model.to_tokens(prompt)
    target_position = -1
    target_token_idx = model.to_single_token(" Paris")
    L = model.cfg.n_layers

    # Full forward + cache
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    seq_len = tokens.shape[1]
    pos_abs = seq_len - 1

    logit_value = logits[0, pos_abs, target_token_idx].item()
    b_U = model.b_U[target_token_idx].item() if model.b_U is not None else 0.0
    decomposable = logit_value - b_U

    print(f"logit_value={logit_value:.6f}  b_U={b_U:.6f}  decomposable={decomposable:.6f}")

    # Get gradients
    grads = _compute_attribution_gradients(
        model, cache, tokens, target_position, target_token_idx, L
    )

    # 1. Embed term: grad_resid_pre[0] · embed
    embed = cache["hook_embed"][0, pos_abs].cpu().double()
    if "hook_pos_embed" in cache.cache_dict:
        embed = embed + cache["hook_pos_embed"][0, pos_abs].cpu().double()
    embed_term = (grads["resid_pre"][0] @ embed).item()

    # 2. MLP terms: grad_mlp_post[l] · true_mlp_post[l]
    mlp_total = 0.0
    mlp_per_layer = []
    for l in range(L):
        true_post = cache[f"blocks.{l}.mlp.hook_post"][0, pos_abs].cpu().double()
        mlp_term = (grads["mlp_post"][l] @ true_post).item()
        mlp_per_layer.append(mlp_term)
        mlp_total += mlp_term

    # 3. Cross-attn terms
    attn_total = 0.0
    for l in range(L):
        a_tt = cache[f"blocks.{l}.attn.hook_pattern"][0, :, pos_abs, pos_abs].cpu().double()
        z_all = cache[f"blocks.{l}.attn.hook_z"][0, pos_abs].cpu().double()
        W_O = model.blocks[l].attn.W_O.cpu().double()

        total_attn = torch.einsum("hd,hdm->hm", z_all, W_O)

        v_h = cache[f"blocks.{l}.attn.hook_v"][0, pos_abs].cpu().double()
        n_kv = v_h.shape[0]
        n_q = W_O.shape[0]
        if n_kv != n_q:
            v_h = v_h.repeat_interleave(n_q // n_kv, dim=0)
        self_loop = a_tt.unsqueeze(1) * torch.einsum("hd,hdm->hm", v_h, W_O)
        cross_attn = total_attn - self_loop

        attn_term = torch.einsum("m,hm->", grads["attn_out"][l], cross_attn).item()
        attn_total += attn_term

    total = embed_term + mlp_total + attn_total
    gap = total - decomposable

    print(f"\nembed_term:  {embed_term:.6f}")
    print(f"mlp_total:   {mlp_total:.6f}")
    for l, v in enumerate(mlp_per_layer):
        print(f"  mlp[{l}]:    {v:.6f}")
    print(f"attn_total:  {attn_total:.6f}")
    print(f"---")
    print(f"sum:         {total:.6f}")
    print(f"decomposable:{decomposable:.6f}")
    print(f"gap:         {gap:.6f}")
    print(f"completeness:{total/decomposable:.6f}")


if __name__ == "__main__":
    main()
