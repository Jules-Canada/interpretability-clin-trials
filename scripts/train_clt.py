#!/usr/bin/env python3
"""
scripts/train_clt.py

Train a Cross-Layer Transcoder on pre-cached activations.
Run scripts/extract_activations.py first to produce the activation files.

Usage — dev run on pythia-70m (no W&B):
    python scripts/train_clt.py \\
        --activation_dir data/activations/pythia-70m \\
        --n_layers 6 \\
        --d_model 512 \\
        --d_mlp 2048 \\
        --n_features 512 \\
        --n_steps 5000 \\
        --no_wandb

Usage — full run on pythia-410m with W&B:
    python scripts/train_clt.py \\
        --activation_dir data/activations/pythia-410m \\
        --n_layers 24 \\
        --d_model 1024 \\
        --d_mlp 4096 \\
        --n_features 4096 \\
        --wandb_group pythia-410m
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clt.config import CLTConfig
from clt.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CLT on pre-cached activations.")

    # Model architecture — must match the model used in extract_activations.py
    p.add_argument("--n_layers",    type=int, required=True, help="Number of transformer layers (e.g. 6 for pythia-70m, 24 for pythia-410m)")
    p.add_argument("--d_model",     type=int, required=True, help="Residual stream dimension (e.g. 512 for pythia-70m, 1024 for pythia-410m)")
    p.add_argument("--d_mlp",       type=int, required=True, help="MLP hidden dimension (e.g. 2048 for pythia-70m, 4096 for pythia-410m)")
    p.add_argument("--n_features",  type=int, default=512,   help="CLT features per layer. Start with d_model; scale up for richer decompositions.")

    # CLT hyperparameters (all have defaults in CLTConfig)
    p.add_argument("--jumprelu_threshold", type=float, default=0.03,  help="Initial JumpReLU threshold θ (learned during training)")
    p.add_argument("--sparsity_coeff",     type=float, default=2e-4,  help="λ: weight on the sparsity loss term")
    p.add_argument("--sparsity_c",         type=float, default=1.0,   help="c: denominator offset in sparsity penalty")

    # Data
    p.add_argument("--activation_dir", type=str, required=True, help="Directory containing resid_stream_l*.pt and mlp_output_l*.pt files")

    # Optimisation
    p.add_argument("--lr",         type=float, default=2e-4,   help="Adam learning rate")
    p.add_argument("--batch_size", type=int,   default=512,    help="Tokens per training step")
    p.add_argument("--n_steps",    type=int,   default=50_000, help="Total training steps")

    # Checkpointing
    p.add_argument("--save_dir",   type=str, default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--save_every", type=int, default=5_000,         help="Save a checkpoint every N steps")

    # W&B
    p.add_argument("--no_wandb",      action="store_true",              help="Disable W&B logging (useful for dev runs)")
    p.add_argument("--wandb_project", type=str, default="clt-replication")
    p.add_argument("--wandb_group",   type=str, default="pythia-70m",   help="W&B run group — use the model size as the group name")
    p.add_argument("--log_every",     type=int, default=50,             help="Log metrics every N steps")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    clt_cfg = CLTConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        n_features=args.n_features,
        jumprelu_threshold=args.jumprelu_threshold,
        sparsity_coeff=args.sparsity_coeff,
        sparsity_c=args.sparsity_c,
    )

    train_cfg = TrainConfig(
        activation_dir=args.activation_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        save_dir=args.save_dir,
        save_every=args.save_every,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        log_every=args.log_every,
        use_wandb=not args.no_wandb,
    )

    print("CLT config:")
    print(f"  n_layers={clt_cfg.n_layers}, d_model={clt_cfg.d_model}, "
          f"d_mlp={clt_cfg.d_mlp}, n_features={clt_cfg.n_features}")
    print(f"  jumprelu_threshold={clt_cfg.jumprelu_threshold}, "
          f"sparsity_coeff={clt_cfg.sparsity_coeff}")
    print()
    print("Train config:")
    print(f"  activation_dir={train_cfg.activation_dir}")
    print(f"  lr={train_cfg.lr}, batch_size={train_cfg.batch_size}, n_steps={train_cfg.n_steps}")
    print(f"  wandb={'disabled' if not train_cfg.use_wandb else train_cfg.wandb_project + '/' + train_cfg.wandb_group}")
    print()

    train(clt_cfg, train_cfg)


if __name__ == "__main__":
    main()
