#!/usr/bin/env python3
"""Can circuit-tracer build a ReplacementModel at this size, and attribute one prompt?

Written 2026-08-23 to pressure-test the claim — inherited from
`docs/program/pod-run-2026-08-plan.md` and false — that attribution graphs stop at 4B.
They do not: 12B peaks at 36.8GB and 27B at 76.5GB of 79.2GB usable on one H100.

Reports VRAM at each stage so a size/card combination can be judged before committing a
session to it. Run this before a graph batch at any new size.

Two things this exists to catch, both of which cost a session when missed:

  reference format  A transcoder set is a plain path whose first two components are the
                    repo id: `mwhanna/gemma-scope-2-27b-it/transcoder_all/width_16k_l0_small_affine`.
                    A bare repo id fails with "Could not download config.yaml" — the
                    config lives inside each width/sparsity subfolder. Naming the variant
                    also keeps the download to that variant; these repos total 1.5-2.4TB.
  margin            27B fits at minimal settings and OOMs at real ones. If peak comes back
                    within a few GB of capacity, plan on --offload cpu or a bigger card.

    python scripts/graph_feasibility.py --model google/gemma-3-12b-it \
        --transcoders mwhanna/gemma-scope-2-12b-it/transcoder_all/width_16k_l0_small_affine
"""
from __future__ import annotations

import argparse
import time

import torch


def _gb(x: float) -> float:
    return x / 1024**3


def vram(tag: str) -> None:
    torch.cuda.synchronize()
    print(f"  [{tag:22s}] alloc {_gb(torch.cuda.memory_allocated()):6.1f} GB   "
          f"reserved {_gb(torch.cuda.memory_reserved()):6.1f} GB   "
          f"peak {_gb(torch.cuda.max_memory_allocated()):6.1f} GB", flush=True)


PROMPT = ("<bos><start_of_turn>user\nInclusion: ECOG <= 1 required for enrollment.\n"
          "Patient: Fully active, no restriction\n"
          "Is the patient eligible? Answer Yes or No.<end_of_turn>\n<start_of_turn>model\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcoders", required=True,
                    help="plain path incl. the width/sparsity variant subfolder")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-n-logits", type=int, default=5)
    ap.add_argument("--max-feature-nodes", type=int, default=2000)
    ap.add_argument("--offload", choices=["cpu", "disk"], default=None)
    ap.add_argument("--skip-attribute", action="store_true")
    a = ap.parse_args()

    from circuit_tracer import ReplacementModel, attribute

    print(f"model={a.model}\ntranscoders={a.transcoders}\n"
          f"batch_size={a.batch_size} offload={a.offload}", flush=True)
    vram("start")
    t0 = time.time()
    model = ReplacementModel.from_pretrained(
        a.model, a.transcoders, backend="nnsight", dtype=torch.bfloat16)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
    vram("after load")
    print(f"  transcoder count: {len(getattr(model, 'transcoders', []) or [])}", flush=True)

    if a.skip_attribute:
        print("RESULT: LOAD_OK (attribution skipped)")
        return

    t0 = time.time()
    try:
        graph = attribute(PROMPT, model, max_n_logits=a.max_n_logits,
                          batch_size=a.batch_size, offload=a.offload,
                          max_feature_nodes=a.max_feature_nodes, verbose=True)
    except torch.cuda.OutOfMemoryError as e:
        vram("at OOM")
        print("RESULT: OOM during attribute()")
        print(str(e)[:300])
        return
    print(f"  attribute() in {time.time() - t0:.0f}s", flush=True)
    vram("after attribute")
    print("RESULT: ATTRIBUTE_OK")
    for attr in ("active_features", "selected_features", "adjacency_matrix"):
        v = getattr(graph, attr, None)
        if v is not None:
            print(f"   {attr}: {tuple(v.shape) if hasattr(v, 'shape') else len(v)}")


if __name__ == "__main__":
    main()
