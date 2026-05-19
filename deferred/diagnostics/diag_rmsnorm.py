"""
Check whether Gemma 3's RMSNorm uses `w * normalize(x)` or `(1+w) * normalize(x)`.

Compares the actual hooked output of ln2_post against both formulas.
The one that matches is the convention TransformerLens uses for this model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformer_lens import HookedTransformer

m = HookedTransformer.from_pretrained("google/medgemma-4b-pt")
m.to("cuda")

toks = m.to_tokens("The patient has received more than two prior lines of therapy and is therefore")

# Cache enough hooks to reconstruct ln2_post input/output at layer 17
names = [
    "blocks.17.hook_mlp_out",
    "blocks.17.hook_resid_mid",
    "blocks.17.ln2_post.hook_scale",
    "blocks.17.hook_resid_post",
]
_, cache = m.run_with_cache(toks, names_filter=lambda n: n in set(names) or "ln2_post" in n or "ln1_post" in n)

print("Layer 17 ln2_post.w values:")
w = m.blocks[17].ln2_post.w.detach().cpu()
print(f"  shape={w.shape}  min={w.min():.4f}  max={w.max():.4f}  mean={w.mean():.4f}")
print(f"  → values near 0 mean (1+w) convention; values near 1 mean w convention")

print("\nLayer 0 ln2_post.w values:")
w0 = m.blocks[0].ln2_post.w.detach().cpu()
print(f"  shape={w0.shape}  min={w0.min():.4f}  max={w0.max():.4f}  mean={w0.mean():.4f}")

print("\nLayer 33 ln2_post.w values:")
w33 = m.blocks[33].ln2_post.w.detach().cpu()
print(f"  shape={w33.shape}  min={w33.min():.4f}  max={w33.max():.4f}  mean={w33.mean():.4f}")

# Direct comparison: r_post - r_mid should equal LN_post(mlp_out)
mlp_out  = cache["blocks.17.hook_mlp_out"].float().cpu()
r_mid    = cache["blocks.17.hook_resid_mid"].float().cpu()
r_post   = cache["blocks.17.hook_resid_post"].float().cpu()
hook_scale = cache["blocks.17.ln2_post.hook_scale"].float().cpu()

actual_post_norm_out = (r_post - r_mid)  # what was added to residual

# normalize(x) = x / rms(x) where rms uses hook_scale
# Compute both candidate formulas
norm_x = mlp_out / hook_scale  # broadcast-divide
candidate_w     = norm_x * w.float()
candidate_1pw   = norm_x * (1.0 + w.float())

# Compare at target position
tgt_pos = mlp_out.shape[1] - 1
err_w   = (actual_post_norm_out[0, tgt_pos] - candidate_w[0, tgt_pos]).abs().max().item()
err_1pw = (actual_post_norm_out[0, tgt_pos] - candidate_1pw[0, tgt_pos]).abs().max().item()

print(f"\nMax-abs error at target position (lower = correct convention):")
print(f"  using `w * normalize(x)`     : {err_w:.6f}")
print(f"  using `(1+w) * normalize(x)` : {err_1pw:.6f}")
