#!/usr/bin/env python
"""
scripts/run_graph.py

Build and export an attribution graph for a single prompt + target token.
Loads a trained CLT checkpoint and Pythia-410m, runs the forward pass,
builds the attribution graph, prunes it, and writes frontend-ready JSON.

Usage:
    python scripts/run_graph.py \\
        --checkpoint checkpoints/pythia-410m/clt_final.pt \\
        --n_layers 24 --d_model 1024 --d_mlp 4096 --n_features 512 \\
        --prompt "Patient is a 58-year-old with NSCLC. Is the patient eligible?" \\
        --target_token " eligible" \\
        --output_slug eligibility_nsclc_001

The JSON is written to frontend/graph_data/<output_slug>.json, where the
frontend (frontend/index.html) can pick it up directly.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from transformer_lens import HookedTransformer

from clt.config import AttributionConfig, CLTConfig
from clt.model import CrossLayerTranscoder
from graphs.build import build_attribution_graph
from graphs.export import save_graph
from graphs.prune import prune_graph, node_influence_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an attribution graph from a prompt + CLT checkpoint.")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to CLT checkpoint (.pt file)")

    # Model architecture — must match the checkpoint
    p.add_argument("--n_layers",   type=int, required=True, help="Number of transformer layers")
    p.add_argument("--d_model",    type=int, required=True, help="Residual stream dimension")
    p.add_argument("--d_mlp",      type=int, required=True, help="MLP hidden dimension")
    p.add_argument("--n_features", type=int, default=512,   help="CLT features per layer")

    # Prompt
    p.add_argument("--prompt",        type=str, required=True, help="Input prompt text")
    p.add_argument("--target_token",  type=str, required=True,
                   help="Token to trace (must be a single token, e.g. ' eligible')")
    p.add_argument("--target_position", type=int, default=-1,
                   help="Sequence position to trace from (default: last token, -1)")

    # Graph settings
    p.add_argument("--top_k_nodes",    type=int, default=30,  help="Max nodes after pruning")
    p.add_argument("--top_k_edges",    type=int, default=100, help="Max edges after pruning")
    p.add_argument("--max_path_length", type=int, default=3,  help="Indirect influence path depth")
    p.add_argument("--min_activation", type=float, default=1e-4,
                   help="Minimum feature activation to include as a node")

    # Output
    p.add_argument("--output_slug", type=str, required=True,
                   help="Filename slug for output JSON (e.g. 'eligibility_nsclc_001')")
    p.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m",
                   help="Model name written to graph metadata")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = _device()
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load Pythia-410m
    # -----------------------------------------------------------------------
    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name)
    model.eval()
    model = model.to(device)
    print(f"  Loaded: {args.n_layers} layers, d_model={args.d_model}")

    # -----------------------------------------------------------------------
    # Load CLT checkpoint
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
    print(f"  Loaded from step {ckpt['step']}")

    # -----------------------------------------------------------------------
    # Tokenize prompt + resolve target token
    # -----------------------------------------------------------------------
    tokens = model.to_tokens(args.prompt)  # (1, seq)
    str_tokens = model.to_str_tokens(tokens[0])
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Tokens ({len(str_tokens)}): {str_tokens}")

    # Resolve target token to vocabulary index
    try:
        target_token_idx = model.to_single_token(args.target_token)
    except Exception:
        raise ValueError(
            f"'{args.target_token}' is not a single token in the vocabulary. "
            "Check spelling and leading space (e.g. ' eligible' not 'eligible')."
        )
    print(f"Target token: '{args.target_token}' (vocab idx={target_token_idx})")

    # -----------------------------------------------------------------------
    # Build attribution graph
    # -----------------------------------------------------------------------
    attr_cfg = AttributionConfig(
        target_position=args.target_position,
        min_activation=args.min_activation,
        top_k_nodes=args.top_k_nodes,
        top_k_edges=args.top_k_edges,
        max_path_length=args.max_path_length,
    )

    print("\nBuilding attribution graph...")
    graph = build_attribution_graph(model, clt, tokens, target_token_idx, cfg=attr_cfg)

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Target logit: {graph.logit_value:.4f}")
    print(f"  Completeness: {graph.completeness:.4f}  (1.0 = perfect attribution)")

    # -----------------------------------------------------------------------
    # Show top features before pruning
    # -----------------------------------------------------------------------
    print("\nTop 10 nodes by indirect influence on target logit:")
    scores = node_influence_scores(graph, cfg=attr_cfg)
    for node_id, score in scores[:10]:
        print(f"  {node_id:<40s}  {score:.4f}")

    # -----------------------------------------------------------------------
    # Prune
    # -----------------------------------------------------------------------
    print(f"\nPruning to top-{args.top_k_nodes} nodes, top-{args.top_k_edges} edges...")
    pruned = prune_graph(graph, cfg=attr_cfg)
    print(f"  After pruning: {len(pruned.nodes)} nodes, {len(pruned.edges)} edges")

    # -----------------------------------------------------------------------
    # Compute softmax probability for logit clerp label
    # -----------------------------------------------------------------------
    with torch.no_grad():
        logits = model(tokens)
    probs = torch.softmax(logits[0, args.target_position], dim=-1)
    logit_prob = probs[target_token_idx].item()
    print(f"  Target token probability: {logit_prob:.4f}")

    # -----------------------------------------------------------------------
    # Export to frontend JSON
    # -----------------------------------------------------------------------
    output_path = f"frontend/graph_data/{args.output_slug}.json"
    saved = save_graph(
        pruned,
        output_path,
        model_name=args.model_name,
        logit_probability=logit_prob,
    )
    print(f"\nGraph saved to: {saved}")
    print(f"Open frontend/index.html and load '{args.output_slug}' to visualize.")


if __name__ == "__main__":
    main()
