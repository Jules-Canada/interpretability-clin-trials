"""
scripts/diag_scales.py — compare per-prompt resid RMS to saved CLT scales.

If saved scales match per-prompt RMS within ~2x, scales aren't the bug.
If they're off by ~8x, the saved scales are wrong (corpus sample bad).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformer_lens import HookedTransformer

PROMPT = "The patient has received more than two prior lines of therapy and is therefore"
CKPT   = "checkpoints/medgemma-4b-1024/clt_inference.pt"

m = HookedTransformer.from_pretrained("google/medgemma-4b-pt")
m.to("cuda")

toks = m.to_tokens(PROMPT)
_, cache = m.run_with_cache(toks)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)

print("\nL    per-prompt resid RMS    saved resid_scales    ratio (saved/prompt)")
print("-" * 75)
for l in [0, 5, 11, 17, 23, 29, 33]:
    rms   = (cache[f"blocks.{l}.hook_resid_pre"].float() ** 2).mean().sqrt().item()
    saved = ckpt["resid_scales"][l].item()
    print(f"{l:<4} {rms:>20.4f}    {saved:>18.4f}    {saved/rms:>6.2f}")

print("\nL    per-prompt mlp RMS      saved mlp_scales      ratio (saved/prompt)")
print("-" * 75)
for l in [0, 5, 11, 17, 23, 29, 33]:
    mrms  = (cache[f"blocks.{l}.mlp.hook_post"].float() ** 2).mean().sqrt().item()
    saved = ckpt["mlp_scales"][l].item()
    print(f"{l:<4} {mrms:>20.4f}    {saved:>18.4f}    {saved/mrms:>6.2f}")
