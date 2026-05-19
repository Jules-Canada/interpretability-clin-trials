"""
Probe the actual structure of Gemma 3's residual update at one layer.

We want to know what `r_post - r_mid` actually equals, so build.py can
correctly compute the Jacobian of the post-MLP norm.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformer_lens import HookedTransformer

m = HookedTransformer.from_pretrained("google/medgemma-4b-pt")
m.to("cuda")

L = 17
toks = m.to_tokens("The patient has received more than two prior lines of therapy and is therefore")

# Cache everything for layer L
_, cache = m.run_with_cache(toks)

print("=== Available hooks for blocks.17 ===")
for k in sorted(cache.cache_dict.keys()):
    if f"blocks.{L}." in k:
        v = cache[k]
        print(f"  {k:55s}  shape={tuple(v.shape)}")

print("\n=== ln2_post structure ===")
ln2_post = m.blocks[L].ln2_post
print(f"  type: {type(ln2_post).__name__}")
print(f"  attrs: {[a for a in dir(ln2_post) if not a.startswith('_')]}")
print(f"  w.shape: {ln2_post.w.shape}, w mean: {ln2_post.w.float().mean():.4f}")
if hasattr(ln2_post, 'b'):
    print(f"  b.shape: {ln2_post.b.shape}, b mean: {ln2_post.b.float().mean():.4f}")
print(f"  forward source:")
import inspect
try:
    print(inspect.getsource(ln2_post.forward))
except Exception as e:
    print(f"  (can't get source: {e})")

print("\n=== Reconstruct r_post - r_mid ===")
mlp_out  = cache[f"blocks.{L}.hook_mlp_out"].float().cpu()
r_mid    = cache[f"blocks.{L}.hook_resid_mid"].float().cpu()
r_post   = cache[f"blocks.{L}.hook_resid_post"].float().cpu()
hook_scale_post = cache[f"blocks.{L}.ln2_post.hook_scale"].float().cpu()

# What's been added to residual at this layer
delta = (r_post - r_mid)[0, -1]   # (d_model,)
print(f"  delta = r_post - r_mid at target pos: rms={delta.pow(2).mean().sqrt():.4f}, max={delta.abs().max():.4f}")
print(f"  mlp_out at target pos:                rms={mlp_out[0,-1].pow(2).mean().sqrt():.4f}, max={mlp_out[0,-1].abs().max():.4f}")
print(f"  ln2_post.hook_scale at target pos:    {hook_scale_post[0,-1].squeeze().item():.4f}")

# Candidate 1: delta == mlp_out (post-norm not applied here)
err = (delta - mlp_out[0,-1]).abs().max().item()
print(f"\n  Candidate: delta == mlp_out                  → max err {err:.4f}")

# Candidate 2: delta == w * (mlp_out / scale)
w = ln2_post.w.float().cpu()
err = (delta - w * (mlp_out[0,-1] / hook_scale_post[0,-1].squeeze())).abs().max().item()
print(f"  Candidate: delta == w * (mlp_out / scale)    → max err {err:.4f}")

# Candidate 3: delta == (1+w) * (mlp_out / scale)
err = (delta - (1+w) * (mlp_out[0,-1] / hook_scale_post[0,-1].squeeze())).abs().max().item()
print(f"  Candidate: delta == (1+w) * (mlp_out/scale)  → max err {err:.4f}")

# Candidate 4: scale is not what I think — try computing rms ourselves
true_scale = mlp_out[0,-1].pow(2).mean().sqrt()
print(f"\n  Computed rms(mlp_out)={true_scale:.4f}  vs hook_scale={hook_scale_post[0,-1].squeeze().item():.4f}")
err = (delta - w * (mlp_out[0,-1] / true_scale)).abs().max().item()
print(f"  Candidate: delta == w * (mlp_out / true_rms) → max err {err:.4f}")
err = (delta - (1+w) * (mlp_out[0,-1] / true_scale)).abs().max().item()
print(f"  Candidate: delta == (1+w) * (mlp_out/true_rms) → max err {err:.4f}")
