#!/usr/bin/env python
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
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        "--flush_every", type=int, default=5,
        help="Flush accumulated activations to HDF5 every N batches. Keep low (5-10) to avoid OOM.",
    )
    p.add_argument(
        "--resid_only", action="store_true",
        help="Store only resid_pre (skip mlp_post). Use when HDF5 is for find_top_activations "
             "only (not CLT training). Reduces storage from ~2.5TB to ~491GB for 5M tokens.",
    )
    p.add_argument(
        "--local_dataset", type=str, default=None,
        help="Path to a local JSONL file to use instead of a HuggingFace dataset. "
             "Each line must be a JSON object with a text field (see --text_field).",
    )
    p.add_argument(
        "--text_field", type=str, default="text",
        help="JSON field containing the document text. Default 'text' (Pile). "
             "Use 'full_text' for clinical trial protocol JSONL files.",
    )
    p.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float16"],
        help="HDF5 storage dtype. float16 halves disk usage — use for large models "
             "(e.g. Gemma 3 4B) where float32 exceeds available disk. "
             "CLT training reads and upcasts to float32 automatically.",
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

def _text_source(args):
    """Yield raw text strings from either a local JSONL file or a HuggingFace dataset."""
    if args.local_dataset:
        import json as _json
        with open(args.local_dataset) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    text = obj.get(args.text_field, "")
                    if text and text.strip():
                        yield text
                except _json.JSONDecodeError:
                    continue
    else:
        ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
        for example in ds:
            text = example.get(args.text_field, "")
            if text and text.strip():
                yield text


def token_batches(
    model: HookedTransformer,
    args,
    seq_len: int,
    batch_size: int,
    max_tokens: int,
):
    """
    Generator that yields (batch_size, seq_len) token tensors.

    Reads from a local JSONL file (--local_dataset) or a HuggingFace streaming
    dataset. Concatenates all text into one long token stream, then slices into
    fixed-length sequences. Stops after max_tokens total tokens have been yielded.
    """
    token_buffer: list[int] = []
    tokens_yielded = 0
    tokens_per_batch = seq_len * batch_size

    for text in _text_source(args):
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
        token_ids      — shape (n_tokens,), int32 — vocabulary token IDs
                         Used by find_top_activations.py to reconstruct token
                         strings for feature labeling.
    """

    def __init__(self, path: str, n_layers: int, d_model: int, d_mlp: int,
                 resid_only: bool = False, dtype: str = "float32"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.n_layers   = n_layers
        self.resid_only = resid_only
        self.dtype      = dtype
        self.f = h5py.File(path, "w")

        # Create resizable datasets — initial size 0, unlimited on axis 0
        for l in range(n_layers):
            self.f.create_dataset(
                f"resid_pre_{l}",
                shape=(0, d_model),
                maxshape=(None, d_model),
                dtype=dtype,
                chunks=(1024, d_model),   # chunk along token axis for efficient random reads
            )
            if not resid_only:
                self.f.create_dataset(
                    f"mlp_post_{l}",
                    shape=(0, d_mlp),
                    maxshape=(None, d_mlp),
                    dtype=dtype,
                    chunks=(1024, d_mlp),
                )

        # Token IDs — needed for feature labeling (reconstruct token strings)
        self.f.create_dataset(
            "token_ids",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=(4096,),
        )

        # In-memory buffers — list of numpy arrays per layer, flushed periodically
        # resid_buffers[l]: list of (batch*seq, d_model) arrays
        self.resid_buffers: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        # mlp_buffers[l]: list of (batch*seq, d_mlp) arrays — empty when resid_only
        self.mlp_buffers:   list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.token_id_buffer: list[np.ndarray] = []
        self.tokens_buffered = 0

    def append(
        self,
        resid_streams: list[torch.Tensor],
        mlp_outputs: list[torch.Tensor],
        token_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Buffer one batch of activations.

        Args:
            resid_streams: list[n_layers] of (batch, seq, d_model)
            mlp_outputs:   list[n_layers] of (batch, seq, d_mlp)
            token_ids:     (batch, seq) int tensor of vocabulary token IDs (optional)
        """
        batch, seq, _ = resid_streams[0].shape
        np_dtype = np.float16 if self.dtype == "float16" else np.float32
        for l in range(self.n_layers):
            # (batch, seq, d_model) → (batch*seq, d_model), moved to CPU
            self.resid_buffers[l].append(
                resid_streams[l].reshape(batch * seq, -1).cpu().to(torch.float32).numpy().astype(np_dtype)
            )
            if not self.resid_only:
                self.mlp_buffers[l].append(
                    mlp_outputs[l].reshape(batch * seq, -1).cpu().to(torch.float32).numpy().astype(np_dtype)
                )
        if token_ids is not None:
            # (batch, seq) → (batch*seq,)
            self.token_id_buffer.append(
                token_ids.reshape(-1).cpu().numpy().astype("int32")
            )
        self.tokens_buffered += batch * seq

    def flush(self) -> None:
        """Append buffered activations to HDF5 datasets, then clear buffers."""
        if self.tokens_buffered == 0:
            return

        for l in range(self.n_layers):
            datasets = [(f"resid_pre_{l}", self.resid_buffers[l])]
            if not self.resid_only:
                datasets.append((f"mlp_post_{l}", self.mlp_buffers[l]))

            for key, buffers in datasets:
                # (tokens_buffered, d)
                chunk = np.concatenate(buffers, axis=0)
                ds = self.f[key]
                n_existing = ds.shape[0]
                ds.resize(n_existing + chunk.shape[0], axis=0)
                ds[n_existing:] = chunk

            self.resid_buffers[l].clear()
            self.mlp_buffers[l].clear()

        if self.token_id_buffer:
            chunk = np.concatenate(self.token_id_buffer, axis=0)
            ds = self.f["token_ids"]
            n_existing = ds.shape[0]
            ds.resize(n_existing + chunk.shape[0], axis=0)
            ds[n_existing:] = chunk
            self.token_id_buffer.clear()

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
    all_hooks   = set(resid_hooks + ([] if args.resid_only else mlp_hooks))

    if args.resid_only:
        print("Mode: resid_only — skipping mlp_post (for find_top_activations, not CLT training)")

    writer = HDF5Writer(args.output_path, n_layers, d_model, d_mlp,
                        resid_only=args.resid_only, dtype=args.dtype)

    batches = token_batches(
        model=model,
        args=args,
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
            # mlp_outputs[l]: (batch, seq, d_mlp) — empty list when resid_only
            mlp_outputs   = [] if args.resid_only else [cache[h] for h in mlp_hooks]

            writer.append(resid_streams, mlp_outputs, token_ids=batch_tokens)

            n_new = batch_tokens.numel()
            tokens_processed += n_new
            pbar.update(n_new)

            if (batch_idx + 1) % args.flush_every == 0:
                writer.flush()
                on_disk = writer.total_tokens()
                elapsed = time.time() - t_start
                tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                eta_hrs = ((args.max_tokens - tokens_processed) / tok_per_sec) / 3600 if tok_per_sec > 0 else 0
                pbar.set_postfix({
                    "on_disk": f"{on_disk:,}",
                    "tok/s": f"{tok_per_sec:,.0f}",
                    "eta": f"{eta_hrs:.2f}h",
                })

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
