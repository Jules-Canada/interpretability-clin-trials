#!/usr/bin/env python
"""
scripts/find_top_activations.py

For each CLT feature, find the top-K token positions where it activates most
strongly, then save those positions with surrounding token context.

This is the first step of the feature labeling pipeline:
  1. find_top_activations.py  ← you are here
  2. label_features.py        — prompts Claude to label each feature
  3. graphs/export.py         — uses labels to populate clerp field

Output: data/feature_activations.jsonl
  One JSON object per feature, per layer. Each object contains:
    {
      "layer": 4,
      "feature": 211,
      "top_examples": [
        {
          "activation": 3.42,
          "token_idx": 18432,       # position in HDF5 file
          "context_tokens": ["The", " capital", " of", " France", ...],
          "target_token_pos": 4     # index within context_tokens where feature fired
        },
        ...
      ]
    }

Usage:
  python scripts/find_top_activations.py \\
      --checkpoint checkpoints/pythia-410m-4096/clt_final.pt \\
      --activation_path data/activations/pythia-410m.h5 \\
      --n_layers 24 --d_model 1024 --d_mlp 4096 --n_features 4096 \\
      --top_k 20 \\
      --output_path data/feature_activations.jsonl

To label only features that appear in attribution graphs (recommended):
  python scripts/find_top_activations.py \\
      ... \\
      --features_file data/graph_features.json

  where graph_features.json is: {"features": [[layer, feature_idx], ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from clt.config import CLTConfig
from clt.model import CrossLayerTranscoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find top activating token contexts per CLT feature.")

    p.add_argument("--checkpoint",       type=str, required=True)
    p.add_argument("--activation_path",  type=str, required=True,
                   help="HDF5 file from extract_activations.py")
    p.add_argument("--n_layers",         type=int, required=True)
    p.add_argument("--d_model",          type=int, required=True)
    p.add_argument("--d_mlp",            type=int, required=True)
    p.add_argument("--n_features",       type=int, default=512)
    p.add_argument("--top_k",            type=int, default=20,
                   help="Top activating examples to keep per feature")
    p.add_argument("--context_window",   type=int, default=10,
                   help="Tokens on each side of the target position to include as context")
    p.add_argument("--batch_size",       type=int, default=2048,
                   help="Token positions to encode at once — reduce if OOM")
    p.add_argument("--output_path",      type=str, default="data/feature_activations.jsonl")
    p.add_argument("--features_file",    type=str, default=None,
                   help="Optional JSON file listing specific (layer, feature) pairs to label. "
                        "If omitted, all features in all layers are processed.")
    p.add_argument("--model_name",       type=str, default="EleutherAI/pythia-410m",
                   help="Used to reconstruct token strings from the HDF5 token indices")

    return p.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clt(args, device) -> CrossLayerTranscoder:
    cfg = CLTConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        n_features=args.n_features,
    )
    clt = CrossLayerTranscoder(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    clt.load_state_dict(ckpt["model_state_dict"])
    clt = clt.to(device)
    clt.eval()
    print(f"Loaded CLT from step {ckpt['step']}")
    return clt


def collect_top_activations(
    clt: CrossLayerTranscoder,
    h5_path: str,
    target_layers: list[int],
    target_features: dict[int, list[int]],  # layer → list of feature indices
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> dict[tuple[int, int], list[tuple[float, int]]]:
    """
    Scan the HDF5 file in batches, encode with CLT, track top-K activations.

    Returns: {(layer, feature): [(activation_value, token_idx), ...]} sorted descending.
    """
    # Heap structure: for each (layer, feature), keep top_k (value, token_idx) pairs
    # Using a simple list + sort since top_k is small (20)
    tops: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for layer in target_layers:
        for feat in target_features[layer]:
            tops[(layer, feat)] = []

    with h5py.File(h5_path, "r") as f:
        n_tokens = f["resid_pre_0"].shape[0]
        n_batches = (n_tokens + batch_size - 1) // batch_size

        with tqdm(total=n_tokens, unit="tok", desc="Scanning activations") as pbar:
            for b in range(n_batches):
                start = b * batch_size
                end   = min(start + batch_size, n_tokens)
                idx   = slice(start, end)

                # Load resid streams for all target layers
                resid_batch = [
                    torch.from_numpy(f[f"resid_pre_{l}"][idx].astype("float32")).to(device)
                    for l in range(clt.cfg.n_layers)
                ]

                with torch.no_grad():
                    # encode() expects list of (batch, d_model) — HDF5 is already flat tokens
                    # Unsqueeze seq dim: (n_tokens, d_model) → (1, n_tokens, d_model)
                    resid_unseq = [r.unsqueeze(0) for r in resid_batch]
                    feature_acts = clt.encode(resid_unseq)
                    # feature_acts[l]: (1, n_tokens, n_features) → (n_tokens, n_features)
                    feature_acts = [a.squeeze(0).cpu() for a in feature_acts]

                # Update top-k for each tracked (layer, feature)
                for layer in target_layers:
                    acts = feature_acts[layer]  # (batch_n_tokens, n_features)
                    for feat in target_features[layer]:
                        key = (layer, feat)
                        vals = acts[:, feat].tolist()
                        for local_pos, val in enumerate(vals):
                            if val <= 0:
                                continue
                            global_pos = start + local_pos
                            tops[key].append((val, global_pos))

                        # Trim to top_k to bound memory
                        if len(tops[key]) > top_k * 10:
                            tops[key].sort(reverse=True)
                            tops[key] = tops[key][:top_k]

                pbar.update(end - start)

    # Final sort + trim
    for key in tops:
        tops[key].sort(reverse=True)
        tops[key] = tops[key][:top_k]

    return tops


def build_examples(
    tops: dict[tuple[int, int], list[tuple[float, int]]],
    h5_path: str,
    tokenizer,
    context_window: int,
) -> list[dict]:
    """
    For each (layer, feature), fetch token context around each top activation position.

    If the HDF5 file contains a 'token_ids' dataset (written by extract_activations.py),
    decodes token strings for context. Otherwise stores raw token indices only.

    Returns list of dicts ready to write as JSONL.
    """
    results = []

    with h5py.File(h5_path, "r") as f:
        n_tokens = f["resid_pre_0"].shape[0]
        has_token_ids = "token_ids" in f

        if not has_token_ids:
            print("Warning: HDF5 file has no 'token_ids' dataset. Context strings will be unavailable.")
            print("  Re-run extract_activations.py to generate a new HDF5 with token IDs stored.")

        for (layer, feat), examples in sorted(tops.items()):
            if not examples:
                continue

            top_examples = []
            for act_val, tok_idx in examples:
                ctx_start = max(0, tok_idx - context_window)
                ctx_end   = min(n_tokens, tok_idx + context_window + 1)
                target_pos_in_ctx = tok_idx - ctx_start

                example = {
                    "activation": round(float(act_val), 4),
                    "token_idx": int(tok_idx),
                    "target_token_pos": int(target_pos_in_ctx),
                }

                if has_token_ids and tokenizer is not None:
                    ids = f["token_ids"][ctx_start:ctx_end].tolist()
                    example["context_tokens"] = tokenizer.convert_ids_to_tokens(ids)
                elif has_token_ids:
                    example["context_token_ids"] = f["token_ids"][ctx_start:ctx_end].tolist()

                top_examples.append(example)

            results.append({
                "layer": layer,
                "feature": feat,
                "top_examples": top_examples,
            })

    return results


def main() -> None:
    args = parse_args()
    device = _device()
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load CLT
    # -----------------------------------------------------------------------
    clt = load_clt(args, device)
    L = args.n_layers
    F = args.n_features

    # -----------------------------------------------------------------------
    # Determine which (layer, feature) pairs to process
    # -----------------------------------------------------------------------
    if args.features_file:
        data = json.loads(Path(args.features_file).read_text())
        pairs = [(int(l), int(f)) for l, f in data["features"]]
        print(f"Targeting {len(pairs)} specific (layer, feature) pairs from {args.features_file}")
    else:
        pairs = [(l, f) for l in range(L) for f in range(F)]
        print(f"Targeting all {len(pairs)} features ({L} layers × {F} features)")

    target_layers = sorted(set(l for l, _ in pairs))
    target_features: dict[int, list[int]] = {l: [] for l in target_layers}
    for l, f in pairs:
        target_features[l].append(f)

    # -----------------------------------------------------------------------
    # Scan HDF5 for top activations
    # -----------------------------------------------------------------------
    print(f"\nScanning {args.activation_path} ...")
    tops = collect_top_activations(
        clt=clt,
        h5_path=args.activation_path,
        target_layers=target_layers,
        target_features=target_features,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Load tokenizer for context string decoding
    # -----------------------------------------------------------------------
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print(f"Loaded tokenizer for {args.model_name}")
    except Exception as e:
        print(f"Warning: could not load tokenizer ({e}). Context strings will be token IDs only.")

    # -----------------------------------------------------------------------
    # Build examples and write output
    # -----------------------------------------------------------------------
    print("\nBuilding examples ...")
    results = build_examples(tops, args.activation_path, tokenizer=tokenizer, context_window=args.context_window)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    n_with_examples = sum(1 for r in results if r["top_examples"])
    print(f"\nWrote {len(results)} features ({n_with_examples} with activations) → {out_path}")
    print("Next step: python scripts/label_features.py --activations_path", args.output_path)


if __name__ == "__main__":
    main()
