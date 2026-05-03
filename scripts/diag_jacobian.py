"""
Compare manual `_compute_attention_propagated_v` against torch autograd.

This is the definitive check: build a frozen forward pass (cached attn pattern,
cached LN/RMSNorm scales) and compute ∂(v · r_L)/∂r_l at one layer using both:
  (a) the manual Jacobian computation in graphs/build.py
  (b) torch autograd applied to the frozen forward

If they agree, the manual code is correct and the 8x completeness gap is
elsewhere. If they disagree, we've found the bug.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformer_lens import HookedTransformer

from graphs.build import _compute_attention_propagated_v, _compute_readout_vector


m = HookedTransformer.from_pretrained("google/medgemma-4b-pt")
m.to("cuda")

PROMPT = "The patient has received more than two prior lines of therapy and is therefore"
TARGET = " excluded"
target_token_idx = m.to_single_token(TARGET)

toks = m.to_tokens(PROMPT)
target_position = toks.shape[1] - 1
L = m.cfg.n_layers
print(f"L={L}, target_pos={target_position}, target_tok={target_token_idx}")

with torch.no_grad():
    _, cache = m.run_with_cache(toks)

# Manual computation of v_at_layer
v = _compute_readout_vector(m, cache, target_position, target_token_idx, L)
v_at_layer_manual = _compute_attention_propagated_v(m, cache, v, target_position, L)
print(f"\nManual v_at_layer norms:")
for l in [0, 5, 17, 23, 33, L]:
    print(f"  L{l}: norm={v_at_layer_manual[l].norm().item():.4e}")

# ----------------------------------------------------------------
# Autograd reference: rebuild forward with frozen nonlinearities,
# compute ∂(v · r_L)/∂r_l at each layer.
#
# We freeze: attention patterns, LN/RMSNorm scales, MLP gate.
# We let everything else differentiate.
# ----------------------------------------------------------------

import torch.nn.functional as F

print("\nComputing autograd reference ...")

# Freeze nonlinearities by replacing them with cached values via hooks.
# Then run forward with autograd enabled and call .backward() on (v · r_L).

frozen_hooks = {}

# Cache LN/RMSNorm scales — replace dynamic computation with frozen values
for l in range(L):
    for ln_name in ["ln1", "ln2", "ln1_post", "ln2_post"]:
        scale_key = f"blocks.{l}.{ln_name}.hook_scale"
        if scale_key in cache.cache_dict:
            frozen_hooks[scale_key] = cache.cache_dict[scale_key].detach().clone()
    pat_key = f"blocks.{l}.attn.hook_pattern"
    if pat_key in cache.cache_dict:
        frozen_hooks[pat_key] = cache.cache_dict[pat_key].detach().clone()

ln_final_scale_key = "ln_final.hook_scale"
if ln_final_scale_key in cache.cache_dict:
    frozen_hooks[ln_final_scale_key] = cache.cache_dict[ln_final_scale_key].detach().clone()

def replace_hook(value, hook):
    if hook.name in frozen_hooks:
        return frozen_hooks[hook.name].to(value.device).to(value.dtype)
    return value

# Run forward with frozen hooks, capturing resid_pre at each layer with grad
resid_pre_grad_targets = {}

def store_resid_pre(value, hook):
    value.requires_grad_(True)
    value.retain_grad()
    resid_pre_grad_targets[hook.name] = value
    return value

# Build the hook list
hook_list = []
for l in range(L):
    hook_list.append((f"blocks.{l}.hook_resid_pre", store_resid_pre))
    for ln_name in ["ln1", "ln2", "ln1_post", "ln2_post"]:
        hook_list.append((f"blocks.{l}.{ln_name}.hook_scale", replace_hook))
    hook_list.append((f"blocks.{l}.attn.hook_pattern", replace_hook))
hook_list.append(("ln_final.hook_scale", replace_hook))

# Forward with grad
m.zero_grad()
for p in m.parameters():
    p.requires_grad_(False)

with m.hooks(fwd_hooks=hook_list):
    logits = m(toks)
# logits: (1, seq, d_vocab)
target_logit = logits[0, target_position, target_token_idx]
target_logit.backward()

print(f"\nAutograd grad norms at hook_resid_pre[l]:")
print(f"{'L':>3}  {'manual norm':>14}  {'autograd norm':>14}  {'cosine':>8}  {'rel_diff':>10}")
for l in [0, 5, 11, 17, 23, 29, 33]:
    key = f"blocks.{l}.hook_resid_pre"
    if key not in resid_pre_grad_targets:
        continue
    g = resid_pre_grad_targets[key].grad
    if g is None:
        print(f"  L{l}: no grad")
        continue
    auto = g[0, target_position].cpu().double()
    manual = v_at_layer_manual[l].cpu().double()
    cos = torch.nn.functional.cosine_similarity(auto.unsqueeze(0), manual.unsqueeze(0)).item()
    rel = (auto - manual).norm().item() / (auto.norm().item() + 1e-8)
    print(f"  L{l}: manual={manual.norm().item():.4e}  auto={auto.norm().item():.4e}  cos={cos:+.4f}  rel_diff={rel:.4f}")
