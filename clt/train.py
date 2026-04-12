"""
clt/train.py — CLT training loop.

Entry points:
  train_step() — single gradient step, accepts tensors directly. Used by tests.
  train()      — full training loop over an ActivationLoader. Logs to wandb.

Usage:
    from clt.config import CLTConfig, TrainConfig
    from clt.model import CrossLayerTranscoder
    from clt.loader import HDF5ActivationLoader
    from clt.train import train
    import wandb

    clt = CrossLayerTranscoder(clt_cfg).to(device)
    loader = HDF5ActivationLoader("data/activations/pythia-410m.h5", clt_cfg, train_cfg, device)
    run = wandb.init(project=train_cfg.wandb_project, group=train_cfg.wandb_group)
    train(train_cfg, clt_cfg, clt, loader, wandb_run=run)
"""

from __future__ import annotations

import itertools
import os
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor

from clt.config import CLTConfig, TrainConfig
from clt.model import CrossLayerTranscoder

if TYPE_CHECKING:
    from clt.loader import ActivationLoader


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    clt: CrossLayerTranscoder,
    optimizer: torch.optim.Optimizer,
    resid_batch: list[Float[Tensor, "batch d_model"]],
    mlp_batch: list[Float[Tensor, "batch d_mlp"]],
) -> dict[str, float | list[float]]:
    """
    One forward + backward + optimizer step.

    Args:
        clt:         The CrossLayerTranscoder being trained.
        optimizer:   Optimizer holding clt.parameters().
        resid_batch: list[L] of (batch, d_model) — residual stream inputs.
        mlp_batch:   list[L] of (batch, d_mlp)   — MLP output targets.

    Returns dict with keys:
        total           float      — combined loss value
        reconstruction  float      — MSE summed over layers
        sparsity        float      — sparsity penalty
        per_layer_mse   list[L]    — per-layer reconstruction MSE
        per_layer_l0    list[L]    — per-layer average active features per token
    """
    optimizer.zero_grad()

    # Forward pass
    # feature_acts: list[L] of (batch, n_features)
    # mlp_recons:   list[L] of (batch, d_mlp)
    feature_acts, mlp_recons = clt(resid_batch)

    # Per-layer MSE — computed before reducing so we can log each layer separately
    # list[L] of scalar tensors
    per_layer_mse = [
        ((y_hat - y) ** 2).mean()
        for y_hat, y in zip(mlp_recons, mlp_batch)
    ]

    rec_loss = sum(per_layer_mse)
    spar_loss = clt.sparsity_loss(feature_acts)
    total = rec_loss + spar_loss

    total.backward()
    optimizer.step()

    # Detach to plain floats — graph is no longer needed after backward
    return {
        "total": total.item(),
        "reconstruction": rec_loss.item(),
        "sparsity": spar_loss.item(),
        # list[float] — one value per layer
        "per_layer_mse": [m.item() for m in per_layer_mse],
        "per_layer_l0": clt.l0_per_layer(feature_acts),
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Return the path of the most recent step checkpoint in checkpoint_dir, or None.

    Scans for files matching `clt_step{N}.pt` and returns the one with the
    highest step number. Ignores `clt_final.pt` — that marks a completed run.
    """
    import glob
    import re

    pattern = os.path.join(checkpoint_dir, "clt_step*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def step_number(path: str) -> int:
        m = re.search(r"clt_step(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    return max(candidates, key=step_number)


def train(
    train_cfg: TrainConfig,
    clt_cfg: CLTConfig,
    clt: CrossLayerTranscoder,
    loader: ActivationLoader,
    wandb_run=None,
    resume_from: str | None = None,
) -> None:
    """
    Train the CLT for train_cfg.n_steps steps, optionally resuming from a checkpoint.

    Args:
        train_cfg:    TrainConfig (lr, n_steps, log_every, save_every, etc.)
        clt_cfg:      CLTConfig (used for checkpoint metadata only).
        clt:          CrossLayerTranscoder to train (modified in place).
        loader:       Any ActivationLoader — yields (resid_streams, mlp_outputs).
        wandb_run:    Optional active wandb run. If None, metrics are printed only.
        resume_from:  Path to a checkpoint file to resume from. If None, starts fresh.
                      Use find_latest_checkpoint() to auto-detect the latest checkpoint.
    """
    optimizer = torch.optim.Adam(clt.parameters(), lr=train_cfg.lr)
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=next(clt.parameters()).device, weights_only=True)
        clt.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {resume_from} (step {ckpt['step']})")

    if start_step >= train_cfg.n_steps:
        print(f"Already trained to {start_step} steps — nothing to do")
        return

    # islice limits consumption to exactly (n_steps - start_step) batches from the
    # loader, regardless of how many items the loader would otherwise yield.
    # This means the loop always ends with step = n_steps - 1, and the final
    # checkpoint is always labeled with the last actually-trained step.
    remaining = train_cfg.n_steps - start_step
    step = start_step - 1  # guard: defined even if remaining == 0
    for step, (resid_batch, mlp_batch) in enumerate(
        itertools.islice(loader, remaining), start=start_step
    ):
        metrics = train_step(clt, optimizer, resid_batch, mlp_batch)

        if step % train_cfg.log_every == 0:
            _log(step, metrics, wandb_run)

        if step % train_cfg.save_every == 0 and step > 0:
            _save_checkpoint(clt, optimizer, step, train_cfg)

    # step is always the last actually-trained step
    _save_checkpoint(clt, optimizer, step, train_cfg, tag="final")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(step: int, metrics: dict, wandb_run) -> None:
    """Log metrics to wandb if available, otherwise print to stdout."""
    flat: dict[str, float] = {
        "loss/total": metrics["total"],
        "loss/reconstruction": metrics["reconstruction"],
        "loss/sparsity": metrics["sparsity"],
    }
    for l, mse in enumerate(metrics["per_layer_mse"]):
        flat[f"layer/{l}/mse"] = mse
    for l, l0 in enumerate(metrics["per_layer_l0"]):
        flat[f"layer/{l}/l0"] = l0

    if wandb_run is not None:
        wandb_run.log(flat, step=step)

    mse_str = ", ".join(f"L{l}={v:.4f}" for l, v in enumerate(metrics["per_layer_mse"]))
    l0_str  = ", ".join(f"L{l}={v:.1f}"  for l, v in enumerate(metrics["per_layer_l0"]))
    print(
        f"step {step:>6d} | "
        f"total={metrics['total']:.4f} | "
        f"recon={metrics['reconstruction']:.4f} | "
        f"sparsity={metrics['sparsity']:.6f} | "
        f"mse=[{mse_str}] | "
        f"l0=[{l0_str}]"
    )


def _save_checkpoint(
    clt: CrossLayerTranscoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_cfg: TrainConfig,
    tag: str | None = None,
) -> None:
    """Save model and optimizer state to disk."""
    name = f"clt_{tag}.pt" if tag else f"clt_step{step:07d}.pt"
    path = os.path.join(train_cfg.checkpoint_dir, name)
    torch.save({
        "step": step,
        "model_state_dict": clt.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")
