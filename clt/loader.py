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

import threading
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

def _rms_scale(
    tensors: list[Tensor],
    dim: int,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute the per-layer RMS scale factor from a list of token-batch tensors.

    For each tensor x of shape (n_tokens, d):
        scale = sqrt(mean(||x||^2 / d))   over all tokens

    Returns a 1-D tensor of length len(tensors).

    Used to normalize residual streams and MLP outputs to unit RMS before
    feeding them to the CLT (paper § Building an Interpretable Replacement Model).
    """
    scales = []
    for x in tensors:
        # (n_tokens, d) → scalar: mean of squared norms / dim
        # (n_tokens, d) → (n_tokens,) via mean over last dim, then scalar mean
        mean_sq = (x.float() ** 2).mean()
        scales.append(mean_sq.sqrt().clamp(min=eps))
    return torch.stack(scales)


class LiveActivationLoader:
    """
    Extracts residual streams and MLP outputs from a HookedTransformer on the fly.

    Each iteration samples a random batch of sequences from tokens, runs a forward
    pass with caching, and returns the activations at each layer.

    If clt_cfg.normalize_activations is True, activations are normalized by the
    per-layer RMS scale estimated from the first batch of each epoch.

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
        # RMS scales computed on first call to __iter__; None until then.
        # (n_layers,) float tensors, one per activation type.
        self._resid_scales: Tensor | None = None
        self._mlp_scales: Tensor | None = None

    def _estimate_scales(self) -> None:
        """
        Estimate per-layer RMS scales from a warm-up batch of all sequences.
        Called once on first iteration; subsequent iterations reuse the scales.
        """
        cfg = self.clt_cfg
        n_seqs = min(self.tokens.shape[0], 256)  # cap at 256 seqs for speed
        sample_tokens = self.tokens[:n_seqs]
        with torch.no_grad():
            _, cache = self.model.run_with_cache(sample_tokens)
        resid_list = [
            cache[f"blocks.{l}.hook_resid_pre"].to(self.device).flatten(0, 1)
            for l in range(cfg.n_layers)
        ]
        mlp_list = [
            cache[f"blocks.{l}.mlp.hook_post"].to(self.device).flatten(0, 1)
            for l in range(cfg.n_layers)
        ]
        self._resid_scales = _rms_scale(resid_list, dim=cfg.d_model).to(self.device)
        self._mlp_scales = _rms_scale(mlp_list, dim=cfg.d_mlp).to(self.device)

    def __iter__(self) -> Iterator[
        tuple[
            list[Float[Tensor, "batch d_model"]],
            list[Float[Tensor, "batch d_mlp"]],
        ]
    ]:
        cfg = self.clt_cfg
        n_seqs = self.tokens.shape[0]
        batch_size = self.train_cfg.batch_size

        if cfg.normalize_activations and self._resid_scales is None:
            self._estimate_scales()

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
                resid = cache[f"blocks.{l}.hook_resid_pre"].to(self.device).flatten(0, 1)
                if cfg.normalize_activations:
                    resid = resid / self._resid_scales[l]
                resid_streams.append(resid)

                # (batch_size, seq_len, d_mlp) → (batch_size * seq_len, d_mlp)
                mlp = cache[f"blocks.{l}.mlp.hook_post"].to(self.device).flatten(0, 1)
                if cfg.normalize_activations:
                    mlp = mlp / self._mlp_scales[l]
                mlp_outputs.append(mlp)

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

    If clt_cfg.normalize_activations is True, per-layer RMS scales are estimated
    from a 4096-token random sample at __init__ time and applied to every batch.

    Args:
        path:       Path to the HDF5 file.
        clt_cfg:    CLTConfig describing layer count and dimensions.
        train_cfg:  TrainConfig providing batch_size and n_steps.
        device:     Device to move sampled tensors to before yielding.
    """

    _SCALE_SAMPLE = 4096  # tokens used to estimate RMS scales at init

    def __init__(
        self,
        path: str,
        clt_cfg: CLTConfig,
        train_cfg: TrainConfig,
        device: torch.device,
    ):
        import h5py

        self.path = path
        self.clt_cfg = clt_cfg
        self.train_cfg = train_cfg
        self.device = device

        # Pre-compute RMS scales if normalization is requested
        self._resid_scales: Tensor | None = None
        self._mlp_scales: Tensor | None = None
        if clt_cfg.normalize_activations:
            self._compute_scales(h5py)

    def _compute_scales(self, h5py) -> None:
        """
        Estimate per-layer RMS scales from a random sample of the HDF5 file.
        Scales are stored as (n_layers,) CPU tensors and moved to device on use.
        """
        cfg = self.clt_cfg
        with h5py.File(self.path, "r") as f:
            n_tokens = f["resid_pre_0"].shape[0]
            sample_size = min(self._SCALE_SAMPLE, n_tokens)
            # Contiguous read for speed (avoids random seeks during init)
            start = np.random.randint(0, n_tokens - sample_size)
            idx = slice(start, start + sample_size)

            resid_list = [
                torch.from_numpy(f[f"resid_pre_{l}"][idx].astype("float32"))
                for l in range(cfg.n_layers)
            ]
            mlp_list = [
                torch.from_numpy(f[f"mlp_post_{l}"][idx].astype("float32"))
                for l in range(cfg.n_layers)
            ]

        self._resid_scales = _rms_scale(resid_list, dim=cfg.d_model).to(self.device)
        self._mlp_scales = _rms_scale(mlp_list, dim=cfg.d_mlp).to(self.device)

    # Tokens to load into CPU RAM per buffer fill (~17 GB at float16, 34 layers)
    _RAM_BUFFER_TOKENS = 20_000

    def _fill_buffer(self, f, n_tokens: int) -> tuple[list[Tensor], list[Tensor]]:
        """
        Load a contiguous chunk of tokens from HDF5, shuffle in-place, and
        pin to page-locked memory for fast CPU→GPU transfer.
        Shuffling once lets __iter__ serve sequential slices instead of
        random gathers — sequential slice is ~50x faster on CPU.
        """
        cfg = self.clt_cfg
        buf_size = min(self._RAM_BUFFER_TOKENS, n_tokens)
        start = np.random.randint(0, n_tokens - buf_size)
        idx = slice(start, start + buf_size)

        perm = torch.randperm(buf_size)
        resid_buf = [
            torch.from_numpy(f[f"resid_pre_{l}"][idx])[perm].pin_memory()
            for l in range(cfg.n_layers)
        ]
        mlp_buf = [
            torch.from_numpy(f[f"mlp_post_{l}"][idx])[perm].pin_memory()
            for l in range(cfg.n_layers)
        ]
        return resid_buf, mlp_buf

    def __iter__(self) -> Iterator[
        tuple[
            list[Float[Tensor, "batch d_model"]],
            list[Float[Tensor, "batch d_mlp"]],
        ]
    ]:
        import h5py

        cfg = self.clt_cfg
        batch_size = self.train_cfg.batch_size
        n_steps = self.train_cfg.n_steps

        with h5py.File(self.path, "r") as f:
            n_tokens = f["resid_pre_0"].shape[0]
            buf_size = min(self._RAM_BUFFER_TOKENS, n_tokens)

            print(f"Loading {buf_size:,} token buffer into CPU RAM...", flush=True)
            resid_buf, mlp_buf = self._fill_buffer(f, n_tokens)
            buf_pos = 0

            for step in range(n_steps):
                # Refill when buffer exhausted
                if buf_pos + batch_size > buf_size:
                    resid_buf, mlp_buf = self._fill_buffer(f, n_tokens)
                    buf_pos = 0

                s = slice(buf_pos, buf_pos + batch_size)
                buf_pos += batch_size

                resid_streams = []
                mlp_outputs = []
                for l in range(cfg.n_layers):
                    # Contiguous slice + non_blocking transfer overlaps H2D with compute
                    resid = resid_buf[l][s].to(self.device, non_blocking=True)
                    if cfg.normalize_activations:
                        resid = resid / self._resid_scales[l]
                    resid_streams.append(resid)

                    mlp = mlp_buf[l][s].to(self.device, non_blocking=True)
                    if cfg.normalize_activations:
                        mlp = mlp / self._mlp_scales[l]
                    mlp_outputs.append(mlp)

                yield resid_streams, mlp_outputs
