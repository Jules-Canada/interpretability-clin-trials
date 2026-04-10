#!/usr/bin/env python3
"""
scripts/train_clt.py

Train a Cross-Layer Transcoder on pre-cached activations.
Run scripts/extract_activations.py first to produce the HDF5 activation file.

Usage — dev run on pythia-70m (no W&B):
    python scripts/train_clt.py \\
        --activation_path data/activations/pythia-70m.h5 \\
        --n_layers 6 \\
        --d_model 512 \\
        --d_mlp 2048 \\
        --n_features 512 \\
        --n_steps 5000 \\
        --no_wandb

Usage — full run on pythia-410m with W&B:
    python scripts/train_clt.py \\
        --activation_path data/activations/pythia-410m.h5 \\
        --n_layers 24 \\
        --d_model 1024 \\
        --d_mlp 4096 \\
        --n_features 4096 \\
        --wandb_group pythia-410m
"""

from __future__ import annotations

import argparse

import torch
import wandb

from clt.config import CLTConfig, TrainConfig
from clt.loader import HDF5ActivationLoader
from clt.model import CrossLayerTranscoder
from clt.train import train


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CLT on pre-cached activations.")

    # Model architecture — must match the model used in extract_activations.py
    p.add_argument("--n_layers",   type=int, required=True, help="Number of transformer layers")
    p.add_argument("--d_model",    type=int, required=True, help="Residual stream dimension")
    p.add_argument("--d_mlp",      type=int, required=True, help="MLP hidden dimension")
    p.add_argument("--n_features", type=int, default=512,   help="CLT features per layer")

    # CLT hyperparameters
    p.add_argument("--jumprelu_threshold", type=float, default=0.03,  help="Initial JumpReLU threshold θ")
    p.add_argument("--jumprelu_bandwidth", type=float, default=0.1,   help="STE bandwidth for JumpReLU gradient")
    p.add_argument("--sparsity_coeff",     type=float, default=2e-4,  help="λ: weight on sparsity loss")
    p.add_argument("--sparsity_c",         type=float, default=1.0,   help="c: denominator offset in sparsity penalty")

    # Data
    p.add_argument("--activation_path", type=str, required=True, help="Path to HDF5 file from extract_activations.py")

    # Optimization
    p.add_argument("--lr",         type=float, default=2e-4,   help="Adam learning rate")
    p.add_argument("--batch_size", type=int,   default=512,    help="Token positions per training step")
    p.add_argument("--n_steps",    type=int,   default=50_000, help="Total training steps")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--save_every",     type=int, default=5_000,         help="Save a checkpoint every N steps")

    # W&B
    p.add_argument("--no_wandb",      action="store_true",                help="Disable W&B logging")
    p.add_argument("--wandb_project", type=str, default="ignis-clt",      help="W&B project name")
    p.add_argument("--wandb_group",   type=str, default="",               help="W&B run group (e.g. 'pythia-410m')")
    p.add_argument("--log_every",     type=int, default=50,               help="Log metrics every N steps")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _device()

    clt_cfg = CLTConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        n_features=args.n_features,
        jumprelu_threshold=args.jumprelu_threshold,
        jumprelu_bandwidth=args.jumprelu_bandwidth,
        sparsity_coeff=args.sparsity_coeff,
        sparsity_c=args.sparsity_c,
    )

    train_cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        log_every=args.log_every,
        use_wandb=not args.no_wandb,
    )

    print(f"Device: {device}")
    print(f"CLT config: {clt_cfg}")
    print(f"Train config: {train_cfg}")
    print()

    clt = CrossLayerTranscoder(clt_cfg).to(device)
    loader = HDF5ActivationLoader(args.activation_path, clt_cfg, train_cfg, device)

    wandb_run = None
    if train_cfg.use_wandb:
        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            group=train_cfg.wandb_group,
            config={**vars(clt_cfg), **vars(train_cfg)},
        )

    train(train_cfg, clt_cfg, clt, loader, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
