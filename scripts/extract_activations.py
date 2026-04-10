#!/usr/bin/env python3
"""
scripts/extract_activations.py

Extract and cache residual streams and MLP hidden states from a Pythia model.
Streams text from a HuggingFace dataset, tokenises it, runs the model in batches,
and writes activations to a single HDF5 file:

    {output_path}
        resid_pre_{l}  — dataset, shape (n_tokens, d_model), float32
        mlp_post_{l}   — dataset, shape (n_tokens, d_mlp),   float32

The HDF5 file is the direct input to HDF5ActivationLoader and scripts/train_clt.py.

Usage — dev run on pythia-70m (~2 min on CPU, ~50k tokens):
    python scripts/extract_activations.py \\
        --model_name EleutherAI/pythia-70m \\
        --output_path data/activations/pythia-70m.h5 \\
        --max_tokens 50000

Usage — full run on pythia-410m (needs GPU):
    python scripts/extract_activations.py \\
        --model_name EleutherAI/pythia-410m \\
        --output_path data/activations/pythia-410m.h5 \\
        --max_tokens 5000000 \\
        --batch_size 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CLT training activations from a Pythia model.")

    p.add_argument(
        "--model_name", type=str, default="EleutherAI/pythia-70m",
        help="HuggingFace model ID. pythia-70m for dev; pythia-410m for full runs.",
    )
    p.add_argument(
        "--output_path", type=str, default="data/activations/pythia-70m.h5",
        help="Path to write the HDF5 output file. Parent directory is created if needed.",
    )
    p.add_argument(
        "--max_tokens", type=int, default=50_000,
        help="Total tokens to extract. 50k for dev; 5M+ for a real training run.",
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Sequences per forward pass. Reduce if you run out of memory.",
    )
    p.add_argument(
        "--seq_len", type=int, default=128,
        help="Tokens per sequence.",
    )
    p.add_argument(
        "--dataset", type=str, default="monology/pile-uncopyrighted",
        help="HuggingFace dataset name.",
    )
    p.add_argument(
        "--dataset_split", type=str, default="train",
        help="Dataset split to use.",
    )
    p.add_argument(
        "--model_cache_dir", type=str, default=None,
        help="Optional local directory to cache downloaded model weights.",
    )
    p.add_argument(
        "--flush_every", type=int, default=500,
        help="Flush accumulated activations to HDF5 every N batches.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Token stream
# ---------------------------------------------------------------------------

def token_batches(
    model: HookedTransformer,
    dataset_name: str,
    split: str,
    seq_len: int,
    batch_size: int,
    max_tokens: int,
):
    """
    Generator that yields (batch_size, seq_len) token tensors from a streaming dataset.

    Concatenates all text into one long token stream, then slices into fixed-length
    sequences. Stops after max_tokens total tokens have been yielded.
    """
    ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

    token_buffer: list[int] = []
    tokens_yielded = 0
    tokens_per_batch = seq_len * batch_size

    for example in ds:
        text = example.get("text", "")
        if not text or not text.strip():
            continue

        # (1, n) → list[int]
        ids = model.to_tokens(text, prepend_bos=False).squeeze(0).tolist()
        token_buffer.extend(ids)

        while len(token_buffer) >= tokens_per_batch:
            # (batch_size, seq_len)
            batch_ids = token_buffer[:tokens_per_batch]
            token_buffer = token_buffer[tokens_per_batch:]

            yield torch.tensor(batch_ids, dtype=torch.long).view(batch_size, seq_len)

            tokens_yielded += tokens_per_batch
            if tokens_yielded >= max_tokens:
                return


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

class HDF5Writer:
    """
    Opens an HDF5 file and provides append() to incrementally write activations.

    Uses resizable datasets so we can stream-append without re-loading existing data.
    Buffers batches in memory and flushes to disk periodically to amortise I/O cost.

    Dataset keys match HDF5ActivationLoader expectations:
        resid_pre_{l}  — shape (n_tokens, d_model)
        mlp_post_{l}   — shape (n_tokens, d_mlp)
    """

    def __init__(self, path: str, n_layers: int, d_model: int, d_mlp: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.n_layers = n_layers
        self.f = h5py.File(path, "w")

        # Create resizable datasets — initial size 0, unlimited on axis 0
        for l in range(n_layers):
            self.f.create_dataset(
                f"resid_pre_{l}",
                shape=(0, d_model),
                maxshape=(None, d_model),
                dtype="float32",
                chunks=(1024, d_model),   # chunk along token axis for efficient random reads
            )
            self.f.create_dataset(
                f"mlp_post_{l}",
                shape=(0, d_mlp),
                maxshape=(None, d_mlp),
                dtype="float32",
                chunks=(1024, d_mlp),
            )

        # In-memory buffers — list of numpy arrays per layer, flushed periodically
        # resid_buffers[l]: list of (batch*seq, d_model) arrays
        self.resid_buffers: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        # mlp_buffers[l]: list of (batch*seq, d_mlp) arrays
        self.mlp_buffers:   list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.tokens_buffered = 0

    def append(
        self,
        resid_streams: list[torch.Tensor],
        mlp_outputs: list[torch.Tensor],
    ) -> None:
        """
        Buffer one batch of activations.

        Args:
            resid_streams: list[n_layers] of (batch, seq, d_model)
            mlp_outputs:   list[n_layers] of (batch, seq, d_mlp)
        """
        batch, seq, _ = resid_streams[0].shape
        for l in range(self.n_layers):
            # (batch, seq, d_model) → (batch*seq, d_model), moved to CPU
            self.resid_buffers[l].append(
                resid_streams[l].reshape(batch * seq, -1).cpu().float().numpy()
            )
            self.mlp_buffers[l].append(
                mlp_outputs[l].reshape(batch * seq, -1).cpu().float().numpy()
            )
        self.tokens_buffered += batch * seq

    def flush(self) -> None:
        """Append buffered activations to HDF5 datasets, then clear buffers."""
        if self.tokens_buffered == 0:
            return

        for l in range(self.n_layers):
            for key, buffers in [
                (f"resid_pre_{l}", self.resid_buffers[l]),
                (f"mlp_post_{l}",  self.mlp_buffers[l]),
            ]:
                # (tokens_buffered, d)
                chunk = np.concatenate(buffers, axis=0)
                ds = self.f[key]
                n_existing = ds.shape[0]
                ds.resize(n_existing + chunk.shape[0], axis=0)
                ds[n_existing:] = chunk

            self.resid_buffers[l].clear()
            self.mlp_buffers[l].clear()

        self.f.flush()
        self.tokens_buffered = 0

    def total_tokens(self) -> int:
        """Return how many tokens have been written to disk so far."""
        return self.f["resid_pre_0"].shape[0]

    def close(self) -> None:
        self.flush()
        self.f.close()


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {args.model_name} ...")

    model = HookedTransformer.from_pretrained(
        args.model_name,
        cache_dir=args.model_cache_dir,
    )
    model.eval()
    model.to(device)

    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    d_mlp    = model.cfg.d_mlp
    print(f"Model: {n_layers} layers, d_model={d_model}, d_mlp={d_mlp}")
    print(f"Dataset: {args.dataset} (streaming)")
    print(f"Target: {args.max_tokens:,} tokens → {args.output_path}")
    print()

    # Hook names to extract — only cache what we need to save memory
    resid_hooks = [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)]
    mlp_hooks   = [f"blocks.{l}.mlp.hook_post"  for l in range(n_layers)]
    all_hooks   = set(resid_hooks + mlp_hooks)

    writer = HDF5Writer(args.output_path, n_layers, d_model, d_mlp)

    batches = token_batches(
        model=model,
        dataset_name=args.dataset,
        split=args.dataset_split,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    tokens_processed = 0
    t_start = time.time()

    with tqdm(total=args.max_tokens, unit="tok", desc="Extracting") as pbar:
        for batch_idx, batch_tokens in enumerate(batches):
            # (batch_size, seq_len) → device
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    batch_tokens,
                    names_filter=lambda name: name in all_hooks,
                )

            # resid_streams[l]: (batch, seq, d_model)
            resid_streams = [cache[h] for h in resid_hooks]
            # mlp_outputs[l]: (batch, seq, d_mlp)
            mlp_outputs   = [cache[h] for h in mlp_hooks]

            writer.append(resid_streams, mlp_outputs)

            n_new = batch_tokens.numel()
            tokens_processed += n_new
            pbar.update(n_new)

            if (batch_idx + 1) % args.flush_every == 0:
                writer.flush()
                pbar.set_postfix({"on_disk": f"{writer.total_tokens():,}"})

    writer.close()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s  ({tokens_processed / elapsed:,.0f} tok/s)")
    print(f"Tokens written: {tokens_processed:,}")
    print(f"Output: {Path(args.output_path).resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    extract(args)
