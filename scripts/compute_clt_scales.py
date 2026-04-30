"""
scripts/compute_clt_scales.py — populate dataset-level RMS scales in a CLT checkpoint.

The CLT trains with per-layer RMS scales computed from a sample of the activation
HDF5 (clt/loader.py:_compute_scales). These scales were never persisted alongside
the model, so attribution at inference fell back to per-prompt RMS — which drifts
across prompts and inflates feature contributions, breaking completeness.

This script reads the same HDF5 the CLT was trained on, computes per-layer
RMS scales over a fixed sample, and writes them into the existing checkpoint
under keys "resid_scales" and "mlp_scales". The model weights are not modified.

Usage:
    python scripts/compute_clt_scales.py \\
        --hdf5 /workspace/medgemma-4b.h5 \\
        --checkpoint checkpoints/medgemma-4b-1024/clt_inference.pt \\
        --n_layers 34
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import h5py
import numpy as np
import torch

from clt.loader import _rms_scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add dataset-level RMS scales to a CLT checkpoint.")
    p.add_argument("--hdf5",       type=str, required=True, help="Path to activation HDF5")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to CLT .pt to update")
    p.add_argument("--n_layers",   type=int, required=True, help="Number of CLT layers")
    p.add_argument("--sample_size", type=int, default=100_000,
                   help="Number of tokens used to estimate RMS (default 100k for stability)")
    return p.parse_args()


def compute_scales(
    hdf5_path: str, n_layers: int, sample_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read a contiguous block of activations from the HDF5 and compute per-layer
    RMS scales for resid_pre_l and mlp_post_l.
    """
    with h5py.File(hdf5_path, "r") as f:
        n_tokens = f["resid_pre_0"].shape[0]
        sample = min(sample_size, n_tokens)
        # Contiguous slice — same access pattern as the loader. RMS is stable
        # over any 100k+ random sample so the start position doesn't matter.
        start = np.random.randint(0, max(1, n_tokens - sample))
        idx = slice(start, start + sample)

        resid_list = [
            torch.from_numpy(f[f"resid_pre_{l}"][idx].astype("float32"))
            for l in range(n_layers)
        ]
        mlp_list = [
            torch.from_numpy(f[f"mlp_post_{l}"][idx].astype("float32"))
            for l in range(n_layers)
        ]

    d_model = resid_list[0].shape[-1]
    d_mlp = mlp_list[0].shape[-1]

    resid_scales = _rms_scale(resid_list, dim=d_model)  # (n_layers,)
    mlp_scales = _rms_scale(mlp_list, dim=d_mlp)        # (n_layers,)
    return resid_scales, mlp_scales


def main() -> None:
    args = parse_args()

    print(f"Reading HDF5: {args.hdf5}")
    print(f"Sample size:  {args.sample_size:,} tokens")
    print(f"Layers:       {args.n_layers}")

    resid_scales, mlp_scales = compute_scales(
        args.hdf5, args.n_layers, args.sample_size
    )
    print(f"resid_scales: min={resid_scales.min():.4f}  max={resid_scales.max():.4f}  mean={resid_scales.mean():.4f}")
    print(f"mlp_scales:   min={mlp_scales.min():.4f}  max={mlp_scales.max():.4f}  mean={mlp_scales.mean():.4f}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "resid_scales" in ckpt or "mlp_scales" in ckpt:
        print("  WARNING: checkpoint already has scales; overwriting.")

    ckpt["resid_scales"] = resid_scales
    ckpt["mlp_scales"] = mlp_scales

    # Atomic write: temp file → move (avoids partial writes on NFS-backed dirs).
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(args.checkpoint) or ".",
        prefix=".scales_tmp_",
        suffix=".pt",
        delete=False,
    ) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(ckpt, tmp_path)
        shutil.move(tmp_path, args.checkpoint)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"Updated: {args.checkpoint}")


if __name__ == "__main__":
    main()
