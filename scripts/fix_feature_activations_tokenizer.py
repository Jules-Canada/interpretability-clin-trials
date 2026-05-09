"""
scripts/fix_feature_activations_tokenizer.py — re-decode context_tokens using
the correct tokenizer, in-place, without re-scanning the HDF5.

When find_top_activations.py is run with --model_name defaulting to a different
model than the one that produced the activations, context_tokens get decoded
under the wrong vocab. The activation values and token_idx fields are
correct — only the string decoding needs fixing. This script reads each
example's token_idx, fetches the context window from the HDF5's token_ids
dataset, and re-decodes with the correct tokenizer.

Usage:
    python scripts/fix_feature_activations_tokenizer.py \\
        --activations data/feature_activations.jsonl \\
        --hdf5 /workspace/medgemma-4b.h5 \\
        --model_name google/medgemma-4b-pt \\
        --context_window 10
"""

from __future__ import annotations

import argparse
import json
import shutil

import h5py
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--context_window", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading tokenizer: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name)

    backup = args.activations + ".bak"
    shutil.copyfile(args.activations, backup)
    print(f"Backed up original to: {backup}")

    with open(args.activations) as f:
        records = [json.loads(line) for line in f if line.strip()]

    fixed = 0
    with h5py.File(args.hdf5, "r") as h5:
        if "token_ids" not in h5:
            raise RuntimeError("HDF5 has no token_ids dataset — re-extract first")
        n_tokens = h5["token_ids"].shape[0]
        ids_dataset = h5["token_ids"]

        for rec in records:
            for ex in rec.get("top_examples", []):
                idx = ex["token_idx"]
                start = max(0, idx - args.context_window)
                end = min(n_tokens, idx + args.context_window + 1)
                ids = ids_dataset[start:end].tolist()
                ex["context_tokens"] = tok.convert_ids_to_tokens(ids)
                fixed += 1

    with open(args.activations, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Re-decoded {fixed} examples across {len(records)} features → {args.activations}")


if __name__ == "__main__":
    main()
