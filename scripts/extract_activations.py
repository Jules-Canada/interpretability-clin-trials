#!/usr/bin/env python3
"""
scripts/extract_activations.py

Extract and cache residual streams and MLP hidden states from a Pythia model.
Reads text from a HuggingFace dataset, tokenises it, runs the model in batches,
and writes one .pt file per layer per activation type:

    {output_dir}/resid_stream_l{l}.pt   — shape (n_tokens, d_model)
    {output_dir}/mlp_output_l{l}.pt     — shape (n_tokens, d_mlp)

These files are the direct input to scripts/train_clt.py.

Usage — fast dev run on pythia-70m (~2 min on CPU):
    python scripts/extract_activations.py \\
        --model_name EleutherAI/pythia-70m \\
        --output_dir data/activations/pythia-70m \\
        --max_tokens 200000

Usage — full run on pythia-410m (needs GPU):
    python scripts/extract_activations.py \\
        --model_name EleutherAI/pythia-410m \\
        --output_dir data/activations/pythia-410m \\
        --max_tokens 5000000 \\
        --batch_size 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CLT training activations from a Pythia model.")

    p.add_argument(
        "--model_name", type=str, default="EleutherAI/pythia-70m",
        help="HuggingFace model ID. Any Pythia size works: pythia-70m, pythia-160m, pythia-410m, pythia-1b.",
    )
    p.add_argument(
        "--output_dir", type=str, default="data/activations/pythia-70m",
        help="Directory to write .pt files into. Created if it doesn't exist.",
    )
    p.add_argument(
        "--max_tokens", type=int, default=200_000,
        help="Total number of tokens to extract. 200k is enough for dev; use 5M+ for a real training run.",
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Sequences per forward pass. Reduce if you run out of memory.",
    )
    p.add_argument(
        "--seq_len", type=int, default=128,
        help="Tokens per sequence. Longer = more context per sample but more memory.",
    )
    p.add_argument(
        "--dataset", type=str, default="wikitext",
        help="HuggingFace dataset name. 'wikitext' for dev; 'monology/pile-uncopyrighted' for full runs.",
    )
    p.add_argument(
        "--dataset_split", type=str, default="train",
        help="Dataset split to use.",
    )
    p.add_argument(
        "--dataset_config", type=str, default="wikitext-103-raw-v1",
        help="Dataset config name, if required (e.g. 'wikitext-103-raw-v1').",
    )
    p.add_argument(
        "--model_cache_dir", type=str, default=None,
        help="Optional local directory to cache downloaded model weights.",
    )
    p.add_argument(
        "--chunk_size", type=int, default=50_000,
        help="Accumulate this many tokens in memory before flushing to disk. Keeps RAM usage bounded.",
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
# Dataset loading and tokenisation
# ---------------------------------------------------------------------------

def token_batches(
    model: HookedTransformer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    seq_len: int,
    batch_size: int,
    max_tokens: int,
):
    """
    Generator that yields batches of token sequences from a HuggingFace text dataset.

    Concatenates all text into one long stream, then slices into fixed-length
    sequences of `seq_len` tokens. Yields (batch_size, seq_len) integer tensors.

    Args:
        model:          loaded HookedTransformer (used for its tokeniser)
        dataset_name:   HuggingFace dataset identifier
        dataset_config: dataset config name (can be None)
        split:          dataset split ('train', 'validation', etc.)
        seq_len:        tokens per sequence
        batch_size:     sequences per batch
        max_tokens:     stop after yielding this many tokens total
    """
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    token_buffer: list[int] = []
    tokens_yielded = 0

    for example in ds:
        text = example.get("text", "")
        if not text or not text.strip():
            continue

        # Tokenise the text and append to the buffer
        ids = model.to_tokens(text, prepend_bos=False).squeeze(0).tolist()
        token_buffer.extend(ids)

        # Slice complete sequences out of the buffer
        while len(token_buffer) >= seq_len * batch_size:
            # (batch_size, seq_len)
            batch_ids = token_buffer[: seq_len * batch_size]
            token_buffer = token_buffer[seq_len * batch_size :]

            batch_tensor = torch.tensor(batch_ids, dtype=torch.long).view(batch_size, seq_len)
            yield batch_tensor

            tokens_yielded += seq_len * batch_size
            if tokens_yielded >= max_tokens:
                return


# ---------------------------------------------------------------------------
# Activation accumulator
# ---------------------------------------------------------------------------

class ActivationAccumulator:
    """
    Collects per-layer residual streams and MLP hidden states in memory,
    then flushes to disk in chunks to keep RAM usage bounded.

    Files are appended to, so multiple flush() calls build up the final tensors.
    """

    def __init__(self, output_dir: Path, n_layers: int):
        self.output_dir = output_dir
        self.n_layers = n_layers
        output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffers: list of tensors to concatenate before flushing
        # resid_buffers[l]: list of (batch*seq, d_model) tensors
        self.resid_buffers:  list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        # mlp_buffers[l]: list of (batch*seq, d_mlp) tensors
        self.mlp_buffers:    list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        self.tokens_in_buffer = 0

    def add(
        self,
        resid_streams: list[torch.Tensor],
        mlp_outputs: list[torch.Tensor],
    ) -> None:
        """
        Add one batch of activations to the in-memory buffers.

        Args:
            resid_streams: list of length n_layers, each (batch, seq, d_model)
            mlp_outputs:   list of length n_layers, each (batch, seq, d_mlp)
        """
        batch, seq, _ = resid_streams[0].shape
        for l in range(self.n_layers):
            # Flatten batch and seq into a single token dimension: (batch*seq, d)
            # (batch, seq, d_model) → (batch*seq, d_model)
            self.resid_buffers[l].append(resid_streams[l].reshape(batch * seq, -1).cpu())
            # (batch, seq, d_mlp) → (batch*seq, d_mlp)
            self.mlp_buffers[l].append(mlp_outputs[l].reshape(batch * seq, -1).cpu())
        self.tokens_in_buffer += batch * seq

    def flush(self) -> None:
        """
        Concatenate buffers and append to on-disk .pt files, then clear the buffers.
        Uses torch.cat + torch.save each flush, building up the full tensors incrementally.
        """
        if self.tokens_in_buffer == 0:
            return

        for l in range(self.n_layers):
            resid_path = self.output_dir / f"resid_stream_l{l}.pt"
            mlp_path   = self.output_dir / f"mlp_output_l{l}.pt"

            # (chunk_tokens, d_model)
            new_resid = torch.cat(self.resid_buffers[l], dim=0)
            # (chunk_tokens, d_mlp)
            new_mlp   = torch.cat(self.mlp_buffers[l],   dim=0)

            # Append to existing file if it exists, else create
            if resid_path.exists():
                existing_resid = torch.load(resid_path, weights_only=True)
                new_resid = torch.cat([existing_resid, new_resid], dim=0)
            if mlp_path.exists():
                existing_mlp = torch.load(mlp_path, weights_only=True)
                new_mlp = torch.cat([existing_mlp, new_mlp], dim=0)

            torch.save(new_resid, resid_path)
            torch.save(new_mlp,   mlp_path)

            self.resid_buffers[l].clear()
            self.mlp_buffers[l].clear()

        self.tokens_in_buffer = 0

    def total_tokens_on_disk(self) -> int:
        """Return how many tokens have been flushed to disk so far."""
        path = self.output_dir / "resid_stream_l0.pt"
        if not path.exists():
            return 0
        return torch.load(path, weights_only=True).shape[0]


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    # Load model
    # Consume the dataset generator once to trigger the "Loading dataset" print
    # before the progress bar starts — purely cosmetic ordering fix.
    print(f"Loading dataset: {args.dataset} ...")
    # (actual load happens inside token_batches on first iteration)

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

    # Hook names to extract
    resid_hooks = [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)]
    mlp_hooks   = [f"blocks.{l}.mlp.hook_post"  for l in range(n_layers)]
    all_hooks   = resid_hooks + mlp_hooks

    output_dir = Path(args.output_dir)
    accumulator = ActivationAccumulator(output_dir, n_layers)

    # Remove any existing output files so we start fresh
    for l in range(n_layers):
        for path in [output_dir / f"resid_stream_l{l}.pt", output_dir / f"mlp_output_l{l}.pt"]:
            if path.exists():
                path.unlink()
                print(f"Removed existing file: {path}")

    batches = token_batches(
        model=model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config if args.dataset_config else None,
        split=args.dataset_split,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    tokens_processed = 0
    t_start = time.time()
    n_batches = args.max_tokens // (args.batch_size * args.seq_len)

    with tqdm(total=args.max_tokens, unit="tok", desc="Extracting") as pbar:
        for batch_tokens in batches:
            # (batch_size, seq_len) → move to device
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    batch_tokens,
                    names_filter=lambda name: name in all_hooks,
                )

            # Pull activations out of cache
            # resid_streams[l]: (batch, seq, d_model)
            resid_streams = [cache[h] for h in resid_hooks]
            # mlp_outputs[l]: (batch, seq, d_mlp)
            mlp_outputs   = [cache[h] for h in mlp_hooks]

            accumulator.add(resid_streams, mlp_outputs)

            n_new = batch_tokens.numel()
            tokens_processed += n_new
            pbar.update(n_new)

            # Flush to disk periodically to keep RAM bounded
            if accumulator.tokens_in_buffer >= args.chunk_size:
                accumulator.flush()
                pbar.set_postfix({"flushed": f"{accumulator.total_tokens_on_disk():,}"})

    # Final flush
    accumulator.flush()

    elapsed = time.time() - t_start
    total_on_disk = accumulator.total_tokens_on_disk()

    print(f"\nDone in {elapsed:.1f}s  ({tokens_processed / elapsed:,.0f} tok/s)")
    print(f"Tokens on disk: {total_on_disk:,}")
    print(f"Output directory: {output_dir.resolve()}")
    print()

    # Print a summary of saved files
    for l in range(n_layers):
        r = output_dir / f"resid_stream_l{l}.pt"
        m = output_dir / f"mlp_output_l{l}.pt"
        r_shape = tuple(torch.load(r, weights_only=True).shape)
        m_shape = tuple(torch.load(m, weights_only=True).shape)
        print(f"  Layer {l}:  resid_stream {r_shape}  |  mlp_output {m_shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    extract(args)
