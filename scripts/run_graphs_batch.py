#!/usr/bin/env python
"""
scripts/run_graphs_batch.py

Build and export attribution graphs for a list of prompts in one shot.
Loads the model and CLT checkpoint once, then iterates over all prompts.

Usage:
    python scripts/run_graphs_batch.py \
        --checkpoint checkpoints/pythia-410m-4096/clt_final.pt \
        --n_layers 24 --d_model 1024 --d_mlp 4096 --n_features 4096 \
        --prompts_file prompts/trial_prompts.json

Prompts file format (JSON array):
    [
      {
        "id": "output_slug",
        "prompt": "The prompt text",
        "target_token": " token",
        "domain_tags": ["optional", "tags"]
      },
      ...
    ]

JSONs are written to frontend/graph_data/<id>.json.
A summary of successes and failures is printed at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformer_lens import HookedTransformer

from clt.config import AttributionConfig, CLTConfig
from clt.model import CrossLayerTranscoder
from graphs.build import build_attribution_graph
from graphs.export import save_graph
from graphs.prune import prune_graph, node_influence_scores


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build attribution graphs for a batch of prompts.")

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_layers",   type=int, required=True)
    p.add_argument("--d_model",    type=int, required=True)
    p.add_argument("--d_mlp",      type=int, required=True)
    p.add_argument("--n_features", type=int, default=512)
    p.add_argument("--prompts_file", type=str, required=True,
                   help="Path to JSON file containing list of prompt dicts")
    p.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")

    # Graph settings
    p.add_argument("--top_k_nodes",     type=int,   default=30)
    p.add_argument("--top_k_edges",     type=int,   default=100)
    p.add_argument("--max_path_length", type=int,   default=3)
    p.add_argument("--min_activation",  type=float, default=1e-4)

    p.add_argument("--output_dir", type=str, default="frontend/graph_data")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _device()
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load prompts
    # -----------------------------------------------------------------------
    prompts = json.loads(Path(args.prompts_file).read_text())
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}\n")

    # -----------------------------------------------------------------------
    # Load model (once)
    # -----------------------------------------------------------------------
    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name)
    model.eval()
    model = model.to(device)
    print(f"  Loaded: {args.n_layers} layers, d_model={args.d_model}\n")

    # -----------------------------------------------------------------------
    # Load CLT (once)
    # -----------------------------------------------------------------------
    clt_cfg = CLTConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        n_features=args.n_features,
    )
    clt = CrossLayerTranscoder(clt_cfg)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    clt.load_state_dict(ckpt["model_state_dict"])
    clt = clt.to(device)
    clt.eval()
    print(f"  Loaded from step {ckpt['step']}\n")

    attr_cfg = AttributionConfig(
        min_activation=args.min_activation,
        top_k_nodes=args.top_k_nodes,
        top_k_edges=args.top_k_edges,
        max_path_length=args.max_path_length,
    )

    # -----------------------------------------------------------------------
    # Process each prompt
    # -----------------------------------------------------------------------
    results = []

    for i, entry in enumerate(prompts):
        slug       = entry["id"]
        prompt     = entry["prompt"]
        target_tok = entry["target_token"]

        print(f"[{i+1}/{len(prompts)}] {slug}")
        print(f"  Prompt: {prompt!r}  →  {target_tok!r}")

        try:
            # Verify target token is single-token
            try:
                target_token_idx = model.to_single_token(target_tok)
            except Exception:
                raise ValueError(
                    f"'{target_tok}' is not a single token. "
                    "Check spelling and leading space."
                )

            tokens = model.to_tokens(prompt)

            graph = build_attribution_graph(
                model, clt, tokens, target_token_idx, cfg=attr_cfg
            )
            pruned = prune_graph(graph, cfg=attr_cfg)

            with torch.no_grad():
                logits = model(tokens)
            probs = torch.softmax(logits[0, -1], dim=-1)
            logit_prob = probs[target_token_idx].item()

            output_path = f"{args.output_dir}/{slug}.json"
            saved = save_graph(
                pruned,
                output_path,
                model_name=args.model_name,
                logit_probability=logit_prob,
            )

            print(f"  Completeness: {graph.completeness:.4f}  |  p({target_tok})={logit_prob:.4f}")
            print(f"  Saved: {saved}")
            results.append({"id": slug, "status": "ok", "completeness": graph.completeness})

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({"id": slug, "status": "failed", "error": str(e)})

        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    ok      = [r for r in results if r["status"] == "ok"]
    failed  = [r for r in results if r["status"] == "failed"]

    print("=" * 60)
    print(f"Done: {len(ok)}/{len(prompts)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed slugs:")
        for r in failed:
            print(f"  {r['id']}: {r['error']}")
    print()
    print("Add these slugs to frontend/data/graph-metadata.json to view in the frontend.")


if __name__ == "__main__":
    main()
