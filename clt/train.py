"""
clt/train.py — Training loop for the Cross-Layer Transcoder (CLT)

Expects pre-cached activations on disk (see scripts/extract_activations.py).
Logs per-layer reconstruction MSE, L0 sparsity, and total loss to wandb.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from clt.config import CLTConfig
from clt.model import CrossLayerTranscoder


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    activation_dir: str          # Directory written by scripts/extract_activations.py

    # Optimization
    lr: float = 2e-4
    batch_size: int = 512
    n_steps: int = 50_000

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 5_000

    # W&B
    wandb_project: str = "clt-replication"
    wandb_group: str = "pythia-70m"
    log_every: int = 50
    use_wandb: bool = True       # set False to disable W&B logging


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations(
    activation_dir: str,
    n_layers: int,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Load pre-cached residual streams and MLP outputs from disk.

    Expected files (written by scripts/extract_activations.py):
        {activation_dir}/resid_stream_l{l}.pt  — (n_tokens, d_model)
        {activation_dir}/mlp_output_l{l}.pt    — (n_tokens, d_mlp)

    Returns:
        resid_streams: list of length n_layers, each (n_tokens, d_model)
        mlp_outputs:   list of length n_layers, each (n_tokens, d_mlp)
    """
    base = Path(activation_dir)
    resid_streams: list[Tensor] = []
    mlp_outputs: list[Tensor] = []
    for l in range(n_layers):
        # (n_tokens, d_model)
        resid = torch.load(base / f"resid_stream_l{l}.pt", map_location=device, weights_only=True)
        # (n_tokens, d_mlp)
        mlp = torch.load(base / f"mlp_output_l{l}.pt", map_location=device, weights_only=True)
        resid_streams.append(resid)
        mlp_outputs.append(mlp)
    return resid_streams, mlp_outputs


def make_dataloader(
    resid_streams: list[Tensor],
    mlp_outputs: list[Tensor],
    batch_size: int,
) -> DataLoader:
    """
    Wrap flat token-level tensors into a shuffled DataLoader.

    Yields tuples of 2*L tensors: first L are resid streams, next L are MLP outputs.
    Each element shape: (batch_size, d_model) or (batch_size, d_mlp).
    """
    # TensorDataset indexes along dim 0 (token dimension)
    dataset = TensorDataset(*(resid_streams + mlp_outputs))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(clt_cfg: CLTConfig, train_cfg: TrainConfig) -> CrossLayerTranscoder:
    """
    Train a CrossLayerTranscoder on pre-cached activations.

    Activations are loaded once into memory; CLT is trained offline from them.
    Returns the trained model.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    L = clt_cfg.n_layers

    wandb.init(
        project=train_cfg.wandb_project,
        group=train_cfg.wandb_group,
        config={**clt_cfg.__dict__, **train_cfg.__dict__},
        mode="online" if train_cfg.use_wandb else "disabled",
    )

    # Load activations once — don't re-forward Pythia during training
    resid_streams, mlp_outputs = load_activations(train_cfg.activation_dir, L, device)
    loader = make_dataloader(resid_streams, mlp_outputs, train_cfg.batch_size)

    clt = CrossLayerTranscoder(clt_cfg).to(device)
    optimizer = torch.optim.Adam(clt.parameters(), lr=train_cfg.lr)

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    loader_iter = iter(loader)

    while step < train_cfg.n_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        # Re-split into per-layer lists; unsqueeze seq dim (seq=1) for model API
        # batch[:L]: (batch_size, d_model) → (batch_size, 1, d_model)
        batch_resid: list[Float[Tensor, "batch 1 d_model"]] = [
            batch[l].unsqueeze(1) for l in range(L)
        ]
        # batch[L:]: (batch_size, d_mlp) → (batch_size, 1, d_mlp)
        batch_mlp: list[Float[Tensor, "batch 1 d_mlp"]] = [
            batch[L + l].unsqueeze(1) for l in range(L)
        ]

        optimizer.zero_grad()
        losses = clt.loss(batch_resid, batch_mlp)
        losses["total"].backward()
        optimizer.step()

        if step % train_cfg.log_every == 0:
            with torch.no_grad():
                feature_acts, mlp_recons = clt(batch_resid)
                # Per-layer MSE on the current batch
                per_layer_mse = [
                    ((mlp_recons[l] - batch_mlp[l]) ** 2).mean().item()
                    for l in range(L)
                ]
                # L0: average number of active features per token, per layer
                l0_per_layer = clt.l0_per_layer(feature_acts)

            log_dict: dict = {
                "loss/total": losses["total"].item(),
                "loss/reconstruction": losses["reconstruction"].item(),
                "loss/sparsity": losses["sparsity"].item(),
                "step": step,
            }
            for l in range(L):
                log_dict[f"mse/layer_{l}"] = per_layer_mse[l]
                log_dict[f"l0/layer_{l}"] = l0_per_layer[l]

            wandb.log(log_dict, step=step)
            mean_l0 = sum(l0_per_layer) / len(l0_per_layer)
            print(
                f"step {step:>6d} | "
                f"total={losses['total'].item():.4f} | "
                f"recon={losses['reconstruction'].item():.4f} | "
                f"sparsity={losses['sparsity'].item():.6f} | "
                f"mean_L0={mean_l0:.1f}"
            )

        if step > 0 and step % train_cfg.save_every == 0:
            ckpt_path = save_dir / f"clt_step_{step}.pt"
            torch.save(clt.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        step += 1

    final_path = save_dir / "clt_final.pt"
    torch.save(clt.state_dict(), final_path)
    print(f"Training complete. Final checkpoint: {final_path}")
    wandb.finish()

    return clt
