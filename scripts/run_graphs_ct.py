#!/usr/bin/env python
"""
scripts/run_graphs_ct.py

Stage 1 attribution graphs via Anthropic's circuit-tracer (nnsight backend) +
Gemma Scope 2 pretrained transcoders. Replaces the retired
graphs/build.py + CrossLayerTranscoder path (see docs/clt_era_archive.md).

Single-target attribution: trace the model's predicted Yes/No token. Because
attribution targets the top-K logits, one run yields logit nodes for BOTH "Yes"
and "No" (the "free" dual-logit read); true contrastive logit(Yes)-logit(No) is
a deliberate follow-up.

Three modes (run them in this order on a fresh pod):

  --probe   Print the installed circuit-tracer API signatures (attribute,
            ReplacementModel.from_pretrained, the graph-export helper). No model
            load, no GPU — instant. Use this FIRST to confirm the calls below
            match the installed version; if they differ, paste the output back.

  --smoke   Reproduce a known graph on the documented gemma-2-2b + built-in
            "gemma" transcoders (transformerlens backend). Confirms the full
            attribute -> export flow end-to-end before loading 4B.

  (default) Real run: --model + --transcoders over the eligibility pairs.

Usage:
    python scripts/run_graphs_ct.py --probe
    python scripts/run_graphs_ct.py --smoke
    python scripts/run_graphs_ct.py --model google/gemma-3-4b-it \
        --transcoders mwhanna/gemma-scope-2-4b-it

NOTE: the attribute()/export calls below reflect the documented API as of
2026-06-14 but were not runnable locally (no GPU). --probe/--smoke are the
guardrail: they surface any signature drift cheaply. Calls that fail print the
real signature instead of crashing silently.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path

# Repo root on sys.path so `prompts` imports when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from prompts.eligibility import ELIGIBILITY_PAIRS, to_chat


# --------------------------------------------------------------------------- #
# circuit-tracer imports (lazy, with helpful errors)
# --------------------------------------------------------------------------- #
def _import_attr():
    from circuit_tracer import ReplacementModel, attribute  # type: ignore
    return ReplacementModel, attribute


def _import_export():
    """Locate the graph-file export helper across plausible module paths."""
    for modpath in ("circuit_tracer.utils", "circuit_tracer.frontend",
                    "circuit_tracer.frontend.graph_files",
                    "circuit_tracer.utils.create_graph_files"):
        try:
            mod = __import__(modpath, fromlist=["create_graph_files"])
            if hasattr(mod, "create_graph_files"):
                return mod.create_graph_files
        except Exception:
            continue
    raise ImportError(
        "Could not find create_graph_files. Run --probe and paste the output so "
        "the export call can be fixed to the installed API."
    )


# --------------------------------------------------------------------------- #
# --probe: print installed signatures (no model load)
# --------------------------------------------------------------------------- #
def probe() -> None:
    print("=== circuit-tracer API probe ===")
    import circuit_tracer  # noqa: F401
    print(f"circuit_tracer: {getattr(circuit_tracer, '__version__', 'unknown')}\n")

    ReplacementModel, attribute = _import_attr()
    print("attribute", inspect.signature(attribute))
    print()
    print("ReplacementModel.from_pretrained",
          inspect.signature(ReplacementModel.from_pretrained))
    print()
    try:
        cgf = _import_export()
        print(f"create_graph_files [{cgf.__module__}]", inspect.signature(cgf))
    except Exception as e:
        print(f"create_graph_files: NOT FOUND ({e})")
    print("\nIf any of these differ from the calls in run_one()/export, paste this output.")


# --------------------------------------------------------------------------- #
# core: one attribution graph
# --------------------------------------------------------------------------- #
def _completeness_from_json(json_path: Path):
    """Completeness proxy from the exported graph (Dev Rule 8 analog): share of
    node influence flowing through transcoder features vs mlp-reconstruction-error
    nodes. circuit-tracer stores no scalar score, but every node carries
    feature_type + influence (confirmed 2026-06-18)."""
    if not json_path.exists():
        return None
    nodes = json.loads(json_path.read_text()).get("nodes", [])
    feat = sum(n.get("influence", 0.0) for n in nodes
               if n.get("feature_type") == "cross layer transcoder")
    err = sum(n.get("influence", 0.0) for n in nodes
              if "error" in (n.get("feature_type") or ""))
    return feat / (feat + err) if (feat + err) > 0 else None


def _logit_read(graph, tokenizer) -> str:
    """Top output logits (the free dual-logit Yes/No read)."""
    try:
        ids = graph.logit_token_ids
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        probs = graph.logit_probabilities
        probs = probs.tolist() if hasattr(probs, "tolist") else list(probs)
        return ", ".join(f"{tokenizer.decode([i]).strip()!r}:{p:.2f}"
                         for i, p in list(zip(ids, probs))[:6])
    except Exception as e:  # format drift — surfaced cheaply in --smoke
        return f"(unavailable: {e})"


def _export_and_score(graph, create_graph_files, *, slug, out_dir,
                      node_threshold, edge_threshold):
    """Write the frontend graph JSON, then read back a completeness proxy.

    Pass the Graph object straight to create_graph_files (it accepts Graph|str).
    We deliberately do NOT call graph.to_pt: the raw .pt for a 4B graph is large
    and the 20GB root disk fills up — to_pt's zip write failed with
    'unexpected pos' (out of space). Only the small frontend JSON is written.
    """
    create_graph_files(graph, slug=slug, output_path=out_dir,
                       node_threshold=node_threshold, edge_threshold=edge_threshold)
    return _completeness_from_json(Path(out_dir) / f"{slug}.json")


def run_one(model, attribute, create_graph_files, *, slug, attr_input, display,
            out_dir, max_n_logits, node_threshold, edge_threshold,
            batch_size, offload, max_feature_nodes) -> dict:
    """Attribute one input, export the graph, return a summary dict."""
    print(f"  input: {display}")
    graph = attribute(attr_input, model, max_n_logits=max_n_logits,
                      max_feature_nodes=max_feature_nodes,
                      batch_size=batch_size, offload=offload, verbose=False)
    print(f"  logits: {_logit_read(graph, model.tokenizer)}")
    comp = _export_and_score(graph, create_graph_files, slug=slug, out_dir=out_dir,
                             node_threshold=node_threshold, edge_threshold=edge_threshold)
    msg = f"completeness~{comp:.3f}" if comp is not None else "completeness~NA"
    print(f"  {msg}  -> {out_dir}/{slug}.json")
    return {"id": slug, "status": "ok", "completeness": comp}


# --------------------------------------------------------------------------- #
# --smoke: gemma-2-2b known graph
# --------------------------------------------------------------------------- #
def smoke(out_dir: str) -> None:
    print("=== smoke: gemma-2-2b known graph ===")
    ReplacementModel, attribute = _import_attr()
    create_graph_files = _import_export()
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "gemma", dtype=torch.bfloat16, backend="transformerlens"
    )
    prompt = "The capital of the state containing Dallas is"
    try:
        graph = attribute(prompt, model, max_n_logits=10, verbose=False)
        print(f"  logits: {_logit_read(graph, model.tokenizer)}")
        comp = _export_and_score(graph, create_graph_files, slug="smoke_dallas",
                                 out_dir=out_dir, node_threshold=0.8, edge_threshold=0.98)
        print(f"  completeness~{comp}  -> {out_dir}/smoke_dallas.json")
        print("\nSmoke OK — the attribute->export flow works.")
    except Exception:
        traceback.print_exc()
        print("\nSmoke FAILED — paste the traceback so the calls can be fixed.")


# --------------------------------------------------------------------------- #
# default: real batch over the eligibility pairs
# --------------------------------------------------------------------------- #
def batch(args) -> None:
    if not args.transcoders:
        sys.exit("--transcoders is required for a real run (e.g. mwhanna/gemma-scope-2-4b-it)")

    ReplacementModel, attribute = _import_attr()
    create_graph_files = _import_export()

    print(f"Loading {args.model}\n  transcoder_set={args.transcoders}\n  backend=nnsight")
    # transcoder_set is the 2nd positional arg (confirmed via --probe 2026-06-18).
    model = ReplacementModel.from_pretrained(
        args.model, args.transcoders, backend="nnsight", dtype=torch.bfloat16
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for i, p in enumerate(ELIGIBILITY_PAIRS):
        slug = p["id"]
        print(f"[{i+1}/{len(ELIGIBILITY_PAIRS)}] {slug}  (expected {p['expected']})")
        try:
            # IT model -> IT transcoders -> chat template, consistently. Tokenize
            # ourselves with add_special_tokens=False: the template already
            # includes <bos>, so this avoids a doubled BOS at the front.
            chat = to_chat(model.tokenizer, p)
            input_ids = model.tokenizer(chat, add_special_tokens=False)["input_ids"]
            results.append(run_one(
                model, attribute, create_graph_files, slug=slug,
                attr_input=input_ids, display=f"{p['patient']} (expect {p['expected']})",
                out_dir=args.output_dir, max_n_logits=args.max_n_logits,
                node_threshold=args.node_threshold, edge_threshold=args.edge_threshold,
                batch_size=args.batch_size, offload=args.offload,
                max_feature_nodes=args.max_feature_nodes,
            ))
        except Exception as e:
            traceback.print_exc()
            results.append({"id": slug, "status": "failed", "error": str(e)})
        print()

    ok = [r for r in results if r["status"] == "ok"]
    print("=" * 60)
    print(f"Done: {len(ok)}/{len(ELIGIBILITY_PAIRS)} graphs.")
    weak = [r for r in ok if (r["completeness"] or 0) < 0.5]
    if weak:
        print("Below completeness 0.5 (Dev Rule 8 analog):",
              ", ".join(r["id"] for r in weak))


def main() -> None:
    ap = argparse.ArgumentParser(description="circuit-tracer attribution graphs (Stage 1).")
    ap.add_argument("--probe", action="store_true", help="Print installed API signatures and exit")
    ap.add_argument("--smoke", action="store_true", help="Reproduce a known gemma-2-2b graph")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--transcoders", default=None,
                    help="HF transcoder ref, e.g. mwhanna/gemma-scope-2-4b-it")
    ap.add_argument("--output_dir", default="frontend/graph_data")
    ap.add_argument("--max_n_logits", type=int, default=10,
                    help="Top-K logits to attribute (>=2 so both Yes and No are captured)")
    ap.add_argument("--max_feature_nodes", type=int, default=8192,
                    help="Cap feature nodes kept by attribute(). 4B x 16k-width x ~30 "
                         "positions otherwise yields >46k nodes, whose edge matrix "
                         "(nodes^2) exceeds torch.sort's INT_MAX limit during pruning.")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="attribute() batch size; lower if nnsight OOMs on 4B (lib default 512)")
    ap.add_argument("--offload", choices=["cpu", "disk"], default=None,
                    help="attribute() offload to free VRAM on 4B+nnsight (default: none)")
    ap.add_argument("--node_threshold", type=float, default=0.8)
    ap.add_argument("--edge_threshold", type=float, default=0.98)
    args = ap.parse_args()

    if args.probe:
        probe()
    elif args.smoke:
        smoke(args.output_dir)
    else:
        batch(args)


if __name__ == "__main__":
    main()
