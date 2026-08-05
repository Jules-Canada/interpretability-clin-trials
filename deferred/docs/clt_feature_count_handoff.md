# ignis — CLT Feature Count & VRAM Analysis (Handoff)

**Context:** MedGemma-4B CLT (Gemma 3 4B base: L=34, d_model=2560, d_mlp=10240, 4× ratio).
Current `n_features = 1024` (per-layer dictionary width). Question was whether 1024 is the
right number and how it compares to the Anthropic CLT.

---

## TL;DR

1. The param math from the Claude Code session (`~6.18M × n_features`) is **correct** — but
   only because our decoder is non-standard.
2. Our decoder reconstructs the **MLP hidden activations (d_mlp = 10,240)**, not the MLP
   **output (d_model = 2,560)**. That makes every decoder 4× larger than a canonical CLT.
   This 4× is the dominant cost driver and the main addressable lever.
3. The "1024 vs 34M" comparison is a category error. `n_features` is *per-layer*; Anthropic's
   published figure is *total across the model*.
4. The real problem with 1024 is **not VRAM** — it's that the dictionary is undercomplete
   (0.4× over d_model), which caps interpretability regardless of compute.

---

## 1. Param formula (verified against `clt/model.py`)

- Encoders: `L × d_model × n_features` = 34 × 2560 × nf = **87,040 · nf**
- Decoders: `L(L+1)/2 × d_mlp × n_features` = 595 × 10240 × nf = **6,092,800 · nf**
- **Total ≈ 6.18M · nf**

Verified in code: `self._init_decoder(F, cfg.d_mlp, L, cfg.d_model)` → decoder out_features = d_mlp.
`decode()` returns `(b, s, d_mlp)`; `graphs/build.py` composes with frozen `W_out`
(`W_dec.T @ W_out`) to reach d_model.

**Consistency check:** 1024 → 6.33B params → 12.7 GB bf16 = our actual ~12 GB
`clt_inference.pt`. Confirms the formula.

### Decoder/encoder ratio decomposition
```
70× = 17.5× (= (L+1)/2, inherent to ANY CLT) × 4× (= d_mlp/d_model, OUR design choice)
```
The O(L²) factor is unavoidable. The 4× is the lever.

---

## 2. "34M" comparison is not apples-to-apples

- Our 1024 = per-layer width. **Total features = 34 × 1024 = 34,816.**
- Anthropic published **10M total** for their largest 18-layer run (~555K/layer), in a
  d_model-decoding architecture. (Could NOT verify a 34M Haiku figure — treat as unconfirmed.)
- The "34M → 210T params → 840 TB" calc plugs a *total* count into a *per-layer* slot in a
  4×-inflated architecture. Meaningless as an Anthropic comparison.

---

## 3. The real problem with 1024: undercomplete dictionary

- Expansion factor = 1024 / 2560 = **0.4× over d_model** (0.1× over the d_mlp we reconstruct).
- Good SAEs/transcoders use **overcomplete** dictionaries (4–32×) for sparse, monosemantic
  features. At 0.4× we are structurally forced into polysemantic features + high reconstruction
  error, independent of compute.
- **The ceiling is architectural, not hardware.**

---

## 4. The lever: decode to d_model instead of d_mlp

`params = 1.61M · nf` (canonical) vs `6.18M · nf` (current).

| n_features | current (→d_mlp) | canonical (→d_model) |
|---|---|---|
| 1024  | 6.33B / 12.7 GB bf16 | 1.65B |
| 4096  | 25.3B (multi-GPU only) | **6.6B — same budget as current 1024** |
| 16384 | 101B | 26.4B |

Switching to a d_model decoder buys **4× feature headroom for free** and *simplifies* the
attribution path (decoder becomes transfer-to-residual directly; drops the `W_dec.T @ W_out`
composition in `build.py`). Cost: lose reconstruction of the specific d_mlp hidden state.

**OPEN QUESTION TO RESOLVE:** is the d_mlp target deliberate (neuron-level tie-in, Gemma Scope 2
weight-init compatibility) or inherited convention?
- If inherited → switch to d_model decoding. Close to free money.
- If deliberate → alternative lever is **windowed cross-layer reach**: cap decoders at ~8
  downstream layers instead of all-downstream → 595 matrices drops to ~270 (~2.2× savings).
  Lossy: clips long-range features (Anthropic observed real features spanning 12+ layers).

---

## 5. VRAM reality (correction)

The "1024 fits in 79 GB on an H100" note is wrong as stated.

- Naive fp32 Adam: params 25 + grads 25 + Adam(m,v) 51 = **~101 GB → OOM on 80 GB H100**.
- bf16 mixed precision alone doesn't help (Adam states dominate, stay fp32).
- **8-bit Adam (bitsandbytes) ≈ 63 GB → fits.** Optimizer offload also works.

**Action:** confirm which optimizer the training run actually uses.

Note: the recent RunPod pod death was a **disk** limit (12 GB ckpt + model weights), a separate
bottleneck from training VRAM.

---

## Recommended next steps

1. Decide d_mlp vs d_model decoder target (the open question above). This gates everything.
2. If switching to d_model: re-target ~4096 features at current single-H100 budget; update
   `build.py` to drop the W_out composition.
3. Confirm optimizer = 8-bit Adam (or offload) for any nf ≥ 1024.
4. Only after that, size toward a healthy expansion (≥8× ≈ 20K features/layer) and price the
   hardware tier.

---

*Grounded:* code-verified facts (decoder dim, param formula, 12 GB ckpt match), Gemma 3 4B
L=34, Anthropic 18L = 10M total features, canonical CLT decodes to MLP output.
*Inferred / to verify:* d_model=2560 / d_mlp=10240 (standard Gemma 3 4B, not re-verified here);
8-bit Adam fit estimates; "34M" Anthropic figure (unconfirmed — found 10M for 18L).
