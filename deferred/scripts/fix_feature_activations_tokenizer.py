"""
scripts/fix_feature_activations_tokenizer.py — re-decode context_tokens using
the correct tokenizer, in-place, without re-scanning the HDF5.

If context_tokens were ever decoded under the wrong vocab (the Phase 4 bug:
a Pythia tokenizer decoding MedGemma token ids), this re-decodes them in
place. The activation values and token_idx fields are correct — only the
string decoding needs fixing. This script reads each example's token_idx,
fetches the context window from the HDF5's token_ids dataset, and re-decodes
with the correct tokenizer.

The tokenizer is taken from the HDF5's 'model_name' attr automatically (same
resolution as find_top_activations.py, shared via scripts/_tokenizer_resolve).
Pass --model_name only for legacy HDF5 files written before that attr existed;
if passed it is asserted against the attr and a mismatch is a hard error.

Usage (normal — model read from HDF5):
    python scripts/fix_feature_activations_tokenizer.py \\
        --activations data/feature_activations.jsonl \\
        --hdf5 /workspace/medgemma-4b.h5 \\
        --context_window 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (deferred/scripts/)
from scripts._tokenizer_resolve import resolve_model_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--model_name", default=None,
                   help="Normally OMITTED — read from the HDF5's 'model_name' "
                        "attr. If passed, asserted against the attr (mismatch "
                        "is a hard error). Required only for legacy HDF5 files "
                        "written before the attr existed.")
    p.add_argument("--context_window", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = resolve_model_name(args.hdf5, args.model_name)
    print(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)

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
