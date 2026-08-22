#!/usr/bin/env python3
"""Residual-stream patching: is the grade representation load-bearing for the answer?

Track C of docs/program/pod-run-2026-08-plan.md. Needs no transcoders, so unlike
attribution graphs it runs at any size that fits for inference.

The design falls out of the sweep set. Take two vignettes with different true
grades under the *same* criterion — one the protocol admits, one it excludes.
Run the donor, capture its residual stream at the final token position, inject
that into the receiver at layer L, and read the eligibility logit. If the grade
representation is what the answer is computed from, the answer flips at some
depth. If the answer is computed from something else, it never does.

Patching at the final position only, which is where the answer is read and the
one position whose content is comparable across two prompts of different lengths.

Two invariants make this checkable without a GPU, and `--selftest` asserts both
on a randomly initialised 3-layer model:

  identity   patching a prompt with its own activations reproduces its logits
  dominance  patching the LAST layer with the donor's activation reproduces the
             donor's logits exactly, because everything downstream of that point
             at that position is just the final norm and the unembed

    python scripts/patch_grade.py --selftest
    python scripts/patch_grade.py --model google/gemma-3-4b-it \\
        --stimuli specs/stimuli/ecog_sweep_v0.csv --pair E012_LE2:E001_LE2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# module tree
# --------------------------------------------------------------------------- #
def decoder_layers(model):
    """The list of decoder layer modules, across the wrappers Gemma 3 ships in.

    `AutoModelForCausalLM` on a multimodal checkpoint nests the text stack a
    level deeper than on a text-only one, so this cannot assume `model.model`.
    """
    for path in (("model", "layers"),
                 ("model", "language_model", "layers"),
                 ("language_model", "model", "layers"),
                 ("transformer", "h")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj):
            return obj
    raise RuntimeError(f"could not locate decoder layers on {type(model).__name__}")


def _split(out):
    """Decoder layers return a tensor on some versions and a tuple on others."""
    return (out[0], out[1:]) if isinstance(out, tuple) else (out, None)


def _rejoin(hidden, rest):
    return (hidden, *rest) if rest is not None else hidden


# --------------------------------------------------------------------------- #
# capture / patch
# --------------------------------------------------------------------------- #
def capture_resid(model, ids):
    """Final-position output of every decoder layer.

    returns: (n_layers, d_model) float32 on cpu
    """
    import torch
    layers = decoder_layers(model)
    grabbed: dict[int, "torch.Tensor"] = {}
    handles = []

    def make(i):
        def hook(_mod, _inp, out):
            hidden, _ = _split(out)
            grabbed[i] = hidden[0, -1, :].detach().float().cpu()   # (d_model,)
        return hook

    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make(i)))
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([grabbed[i] for i in range(len(layers))])


def patched_logits(model, ids, layer_idx: int, vector):
    """Logits at the final position with layer_idx's final-position output replaced.

    vector: (d_model,)   returns: (vocab,) float32 on cpu
    """
    import torch
    layers = decoder_layers(model)

    def hook(_mod, _inp, out):
        hidden, rest = _split(out)
        hidden = hidden.clone()
        hidden[0, -1, :] = vector.to(hidden.device, hidden.dtype)
        return _rejoin(hidden, rest)

    h = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids=ids)
    finally:
        h.remove()
    return out.logits[0, -1].detach().float().cpu()


def baseline_logits(model, ids):
    import torch
    with torch.no_grad():
        return model(input_ids=ids).logits[0, -1].detach().float().cpu()


def sweep(model, donor_ids, receiver_ids, tok_yes: int, tok_no: int) -> list[dict]:
    """Patch the donor's activation into the receiver at each layer in turn."""
    resid = capture_resid(model, donor_ids)                  # (n_layers, d_model)
    n = len(decoder_layers(model))

    def read(lg):
        return {"logit_diff": round((lg[tok_yes] - lg[tok_no]).item(), 4),
                "says": "Yes" if lg[tok_yes] > lg[tok_no] else "No"}

    base_r = read(baseline_logits(model, receiver_ids))
    base_d = read(baseline_logits(model, donor_ids))
    rows = []
    for L in range(n):
        r = read(patched_logits(model, receiver_ids, L, resid[L]))
        rows.append({"layer": L, **r,
                     "flipped": r["says"] != base_r["says"]})
    return [{"layer": "receiver_baseline", **base_r},
            {"layer": "donor_baseline", **base_d}, *rows]


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest() -> None:
    # Builds its model from a config and must never reach the network. Without
    # this, transformers stalls on hub lookups when there is no route out — a
    # 10-second check takes 90 seconds on a plane.
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM

    torch.manual_seed(0)
    cfg = Gemma3TextConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=3, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=8, sliding_window=8,
                           max_position_embeddings=64)
    model = Gemma3ForCausalLM(cfg).eval()
    n = len(decoder_layers(model))
    print(f"  built a {n}-layer random model; layer discovery OK")

    donor = torch.tensor([[3, 9, 14, 22, 7]])              # different lengths on
    recv = torch.tensor([[5, 1, 30, 2]])                   # purpose

    base_r = baseline_logits(model, recv)
    base_d = baseline_logits(model, donor)
    ok = True

    # identity — patching a prompt with its own activation changes nothing
    own = capture_resid(model, recv)
    for L in range(n):
        got = patched_logits(model, recv, L, own[L])
        same = torch.allclose(got, base_r, atol=1e-4)
        ok &= same
        print(f"  {'PASS' if same else 'FAIL'}  identity at layer {L}"
              f"  (max delta {(got - base_r).abs().max():.2e})")

    # dominance — the last layer's final-position output determines the logits
    dres = capture_resid(model, donor)
    got = patched_logits(model, recv, n - 1, dres[n - 1])
    same = torch.allclose(got, base_d, atol=1e-4)
    ok &= same
    print(f"  {'PASS' if same else 'FAIL'}  last-layer patch reproduces the donor"
          f"  (max delta {(got - base_d).abs().max():.2e})")

    # non-degeneracy — an early patch must not already be the donor's logits,
    # or the sweep carries no depth information
    early = patched_logits(model, recv, 0, dres[0])
    diff = (early - base_d).abs().max().item()
    good = diff > 1e-4
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  layer-0 patch differs from the donor"
          f"  (max delta {diff:.2e})")

    # hooks must not leak: a plain forward after all of the above is unchanged
    same = torch.allclose(baseline_logits(model, recv), base_r, atol=1e-6)
    ok &= same
    print(f"  {'PASS' if same else 'FAIL'}  no hook left installed")

    print("\nselftest:", "OK" if ok else "FAILURES")
    if not ok:
        sys.exit(1)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--stimuli", type=Path, default=Path("specs/stimuli/ecog_sweep_v0.csv"))
    ap.add_argument("--pair", action="append", default=[],
                    help="DONOR_ID:RECEIVER_ID, repeatable. Use ids from --stimuli; "
                         "pick two different true grades under the same criterion.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.pair:
        ap.error("give at least one --pair DONOR:RECEIVER, or --selftest")

    # Resolve the pairs before importing torch or touching the hub: a typo should
    # cost a second, not a model load on a paid pod. run_ecog_stimuli imports no
    # torch at module level, so this is cheap.
    from run_ecog_stimuli import load_stimuli

    rows = {r["id"]: r for r in load_stimuli(args.stimuli)}
    pairs = []
    for spec in args.pair:
        d, sep, r = spec.partition(":")
        if not sep:
            ap.error(f"--pair wants DONOR:RECEIVER, got {spec!r}")
        for i in (d, r):
            if i not in rows:
                ap.error(f"id {i!r} not in {args.stimuli}")
        if rows[d]["criterion_text"] != rows[r]["criterion_text"]:
            ap.error(f"{d} and {r} carry different criteria; hold the criterion "
                     f"fixed or the patch confounds grade with protocol")
        if rows[d]["expected_grade"] == rows[r]["expected_grade"]:
            ap.error(f"{d} and {r} have the same true grade — nothing to transfer")
        pairs.append((d, r))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_ecog_stimuli import (INTERMEDIATES, calibrate_surface, eligibility_chat,
                                  pick_device, resolve_tokens)

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=args.device_map,
            low_cpu_mem_usage=True)
        device = next(model.parameters()).device
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model.to(device)
    model.eval()

    # Same calibrated single-token readout as the runner. Case variants are for
    # evaluation only and are never mixed in here.
    cfg = INTERMEDIATES["ecog"]
    first = rows[pairs[0][0]]
    prefer = calibrate_surface(model, tokenizer, eligibility_chat(tokenizer, first),
                               ["Yes", "No"], device)
    tok = resolve_tokens(tokenizer, cfg, prefer, prefer)
    print(f"readout: yes={tok['yes_form']!r} no={tok['no_form']!r}")

    def encode(row):
        return tokenizer(eligibility_chat(tokenizer, row), add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)

    results = []
    for d, r in pairs:
        print(f"\n{d} (grade {rows[d]['expected_grade']}) -> "
              f"{r} (grade {rows[r]['expected_grade']})")
        table = sweep(model, encode(rows[d]), encode(rows[r]), tok["yes"], tok["no"])
        for t in table:
            flag = "  <- flips" if t.get("flipped") else ""
            print(f"   layer {str(t['layer']):>18}  {t['says']:>3}  "
                  f"logit_diff {t['logit_diff']:+8.3f}{flag}")
        results.append({"donor": d, "receiver": r,
                        "donor_grade": rows[d]["expected_grade"],
                        "receiver_grade": rows[r]["expected_grade"],
                        "criterion": rows[r]["criterion_text"], "sweep": table})

    if args.out:
        args.out.write_text(json.dumps(
            {"model": args.model, "stimuli": str(args.stimuli), "pairs": results},
            indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
