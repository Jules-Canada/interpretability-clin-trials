"""
clt/loader.py — Activation data loaders for CLT training.

Two implementations share the same iterator interface:
  - LiveActivationLoader: extracts activations on the fly from a HookedTransformer.
    Use for local dev / small models (pythia-70m, toy).
  - HDF5ActivationLoader: reads cached activations from disk.
    Use for large models (pythia-410m) where re-running the forward pass per step
    would be prohibitively slow.

Both yield (resid_streams, mlp_outputs) batches of shape:
  resid_streams: list[L] of (batch_size, d_model)
  mlp_outputs:   list[L] of (batch_size, d_mlp)

Note: batch dimension here is over token positions (not sequences). Positions are
sampled randomly across the full dataset on each step.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from clt.config import CLTConfig, TrainConfig


# ---------------------------------------------------------------------------
# Loader protocol — the interface train() depends on
# ---------------------------------------------------------------------------

class ActivationLoader(Protocol):
    """
    Any object that yields (resid_streams, mlp_outputs) batches.
    train() depends only on this interface, not on a specific loader class.
    """
    def __iter__(self) -> Iterator[
        tuple[
            list[Float[Tensor, "batch d_model"]],
            list[Float[Tensor, "batch d_mlp"]],
        ]
    ]: ...


# ---------------------------------------------------------------------------
# LiveActivationLoader
# ---------------------------------------------------------------------------

class LiveActivationLoader:
    """
    Extracts residual streams and MLP outputs from a HookedTransformer on the fly.

    Each iteration samples a random batch of sequences from tokens, runs a forward
    pass with caching, and returns the activations at each layer.

    Args:
        model:      A frozen HookedTransformer. Will be set to eval mode.
        tokens:     Integer token ids, shape (n_seqs, seq_len).
        clt_cfg:    CLTConfig describing the model's layer count and dimensions.
        train_cfg:  TrainConfig providing batch_size and n_steps.
        device:     Device to move activations to before yielding.
    """

    def __init__(
        self,
        model: HookedTransformer,
        tokens: Float[Tensor, "n_seqs seq_len"],
        clt_cfg: CLTConfig,
        train_cfg: TrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.model.eval()
        self.tokens = tokens
        self.clt_cfg = clt_cfg
        self.train_cfg = train_cfg
        self.device = device

    def __iter__(self) -> Iterator[
        tuple[
            list[Float[Tensor, "batch d_model"]],
            list[Float[Tensor, "batch d_mlp"]],
        ]
    ]:
        cfg = self.clt_cfg
        n_seqs = self.tokens.shape[0]
        batch_size = self.train_cfg.batch_size

        for _ in range(self.train_cfg.n_steps):
            # Sample a random batch of sequences
            # (batch_size,)
            seq_idx = torch.randint(0, n_seqs, (batch_size,))
            # (batch_size, seq_len)
            batch_tokens = self.tokens[seq_idx]

            with torch.no_grad():
                _, cache = self.model.run_with_cache(batch_tokens)

            # Extract and move to target device
            # Each tensor: (batch_size, seq_len, d_model) or (batch_size, seq_len, d_mlp)
            # We flatten seq into the batch dimension so downstream treats each
            # token position independently: (batch_size * seq_len, d_{model,mlp})
            resid_streams = []
            mlp_outputs = []
            for l in range(cfg.n_layers):
                # (batch_size, seq_len, d_model) → (batch_size * seq_len, d_model)
                resid = cache[f"blocks.{l}.hook_resid_pre"].to(self.device)
                resid_streams.append(resid.flatten(0, 1))

                # (batch_size, seq_len, d_mlp) → (batch_size * seq_len, d_mlp)
                mlp = cache[f"blocks.{l}.mlp.hook_post"].to(self.device)
                mlp_outputs.append(mlp.flatten(0, 1))

            yield resid_streams, mlp_outputs


# ---------------------------------------------------------------------------
# HDF5ActivationLoader
# ---------------------------------------------------------------------------

class HDF5ActivationLoader:
    """
    Reads cached residual streams and MLP outputs from an HDF5 file.

    Expected file layout (written by scripts/extract_activations.py):
        resid_pre_{l}  — dataset, shape (n_tokens, d_model), float32
        mlp_post_{l}   — dataset, shape (n_tokens, d_mlp),   float32

    where n_tokens is the total number of token positions across the corpus
    (sequences × seq_len, flattened).

    Args:
        path:       Path to the HDF5 file.
        clt_cfg:    CLTConfig describing layer count and dimensions.
        train_cfg:  TrainConfig providing batch_size and n_steps.
        device:     Device to move sampled tensors to before yielding.
    """

    def __init__(
        self,
        path: str,
        clt_cfg: CLTConfig,
        train_cfg: TrainConfig,
        device: torch.device,
    ):
        self.path = path
        self.clt_cfg = clt_cfg
        self.train_cfg = train_cfg
        self.device = device

    def __iter__(self) -> Iterator[
        tuple[
            list[Float[Tensor, "batch d_model"]],
            list[Float[Tensor, "batch d_mlp"]],
        ]
    ]:
        import h5py

        cfg = self.clt_cfg
        batch_size = self.train_cfg.batch_size

        with h5py.File(self.path, "r") as f:
            n_tokens = f[f"resid_pre_0"].shape[0]

            for _ in range(self.train_cfg.n_steps):
                # Sample unique random token positions from the full corpus.
                # h5py fancy indexing requires sorted, unique indices.
                # (batch_size,)
                idx = np.random.choice(n_tokens, size=batch_size, replace=False)
                idx.sort()

                resid_streams = []
                mlp_outputs = []
                for l in range(cfg.n_layers):
                    sorted_idx = idx

                    # (batch_size, d_model)
                    resid = torch.from_numpy(
                        f[f"resid_pre_{l}"][sorted_idx]
                    ).to(self.device)
                    resid_streams.append(resid)

                    # (batch_size, d_mlp)
                    mlp = torch.from_numpy(
                        f[f"mlp_post_{l}"][sorted_idx]
                    ).to(self.device)
                    mlp_outputs.append(mlp)

                yield resid_streams, mlp_outputs
