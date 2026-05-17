"""
scripts/screen_prompts.py — sanity-check prompts before extracting activations.

For each prompt:
  1. Resolve target_token to a single token id (auto-prefix with a leading space
     since prompts ending in "Answer:" emit a space-prefixed continuation).
  2. Forward MedGemma on the prompt, read off softmax probabilities at the last
     position.
  3. Print p(target), top-K predicted tokens, and a PASS/FAIL flag against
     --min_prob.

Catches the two failure modes that bit us in Phase 4:
  - target_token doesn't tokenize to a single Gemma token (silent mismatch)
  - p(target) < threshold, meaning the model isn't actually predicting the
    target — graphs would be meaningless.

Supports both the new "text" key (categorical_prompts) and the legacy "prompt"
key (eligibility / adverse_events / endpoints).

Usage:
    python scripts/screen_prompts.py \\
        --prompts prompts/categorical_prompts.py \\
        --var_name CATEGORICAL_PROMPTS \\
        --model_name google/medgemma-4b-pt \\
        --min_prob 0.2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path: str, var_name: str) -> list[dict]:
    spec = importlib.util.spec_from_file_location("prompts_module", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return list(getattr(mod, var_name))


def prompt_text(p: dict) -> str:
    if "text" in p:
        return p["text"]
    if "prompt" in p:
        return p["prompt"]
    raise KeyError(f"Prompt {p.get('id', '?')} has neither 'text' nor 'prompt' key")


def resolve_target(tok, text: str, target: str) -> tuple[str, int | None]:
    """
    Return (literal_used, token_id) — token_id is None if no single-token form found.

    Tries leading-space form first when text doesn't end in whitespace, since
    that's the standard sentencepiece/Gemma pattern after punctuation.
    """
    candidates: list[str] = []
    if not text.endswith((" ", "\n", "\t")):
        candidates.append(" " + target)
    candidates.append(target)
    # Also try the user's literal as-is even if it has a leading space already.
    if target not in candidates:
        candidates.append(target)
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        ids = tok.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return cand, ids[0]
    return target, None


def surface_variants(word: str) -> list[str]:
    """Casing variants of an answer word we treat as the same answer.

    `medgemma-4b-pt` is a pretrained base model; after an "Answer:" scaffold it
    spreads probability across Yes/yes/YES (etc.), so scoring a single canonical
    token understates the model's true semantic confidence. We aggregate over
    these forms — see auto-memory project_prompt_design_flaw, Effect 1.
    """
    base = word.strip()
    return sorted({base, base.lower(), base.upper(), base.capitalize()})


def resolve_variant_ids(tok, text: str, word: str) -> tuple[list[str], list[int]]:
    """Single-token ids for every casing variant of `word`.

    Honors the same leading-space rule as resolve_target. Dedups ids so a
    variant that tokenizes identically to another isn't double-counted.
    Returns (literals_used, token_ids) — empty ids means nothing scoreable.
    """
    want_space = not text.endswith((" ", "\n", "\t"))
    literals: list[str] = []
    ids: list[int] = []
    seen: set[int] = set()
    for form in surface_variants(word):
        for cand in ([" " + form, form] if want_space else [form]):
            enc = tok.encode(cand, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in seen:
                seen.add(enc[0])
                literals.append(cand)
                ids.append(enc[0])
    return literals, ids


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, help="Path to Python file with prompts list")
    p.add_argument("--var_name", default="CATEGORICAL_PROMPTS")
    p.add_argument("--model_name", default="google/medgemma-4b-pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--min_prob", type=float, default=0.2,
                   help="Flag prompts where p(target) is below this threshold")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--output", default=None,
                   help="Optional: write per-prompt results to JSON here")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prompts = load_prompts(args.prompts, args.var_name)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}::{args.var_name}")

    device = pick_device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    # MPS + float16 produces all-NaN logits on Gemma 3 (large activations
    # overflow fp16's range). The float16 default is for CUDA pods; auto-switch
    # to bfloat16 on MPS so a local run doesn't silently waste a model load.
    if device.type == "mps" and dtype is torch.float16:
        print("WARNING: MPS + float16 yields NaN on Gemma 3 — switching to bfloat16")
        dtype = torch.bfloat16
    print(f"Device: {device} | dtype: {dtype}")

    print(f"Loading tokenizer + model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")
    print()

    results: list[dict] = []
    multi_token: list[tuple[str, str]] = []
    failures: list[tuple[str, str, float]] = []

    header = (f"{'id':<32} {'target':<8} {'flag':<6} "
              f"{'p_raw':>8} {'p_agg':>8}   top-{args.top_k}")
    print(header)
    print("-" * (len(header) + 28))

    for prompt in prompts:
        pid = prompt["id"]
        text = prompt_text(prompt)
        target = prompt["target_token"]
        target_literal, target_id = resolve_target(tok, text, target)
        agg_literals, agg_ids = resolve_variant_ids(tok, text, target)

        # Truly unscoreable only if no casing variant is single-token either.
        if target_id is None and not agg_ids:
            multi_token.append((pid, target))
            print(f"{pid:<32} {target!r:<8} {'MULTI':<6} "
                  f"{'----':>8} {'----':>8}   "
                  f"(no single-token form under {args.model_name})")
            results.append({
                "id": pid, "target": target, "target_literal": target_literal,
                "single_token": False, "p_target": None, "p_agg": None,
                "agg_literals": [], "top_k": [],
            })
            continue

        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
            probs = F.softmax(logits.float(), dim=-1)

        p_target = probs[target_id].item() if target_id is not None else None
        idx = torch.tensor(agg_ids, device=probs.device)
        p_agg = float(probs[idx].sum().item())

        top_p, top_i = probs.topk(args.top_k)
        top = [(tok.decode([i]), float(p)) for i, p in zip(top_i.tolist(), top_p.tolist())]

        # Gate on aggregated mass — that's the model's true answer confidence.
        flag = "PASS" if p_agg >= args.min_prob else "FAIL"
        if flag == "FAIL":
            failures.append((pid, target_literal, p_agg))

        p_raw_str = f"{p_target:>8.4f}" if p_target is not None else f"{'n/a':>8}"
        top_str = ", ".join(f"{repr(t)}={p:.2f}" for t, p in top)
        print(f"{pid:<32} {target_literal!r:<8} {flag:<6} "
              f"{p_raw_str} {p_agg:>8.4f}   {top_str}")

        results.append({
            "id": pid, "target": target, "target_literal": target_literal,
            "single_token": target_id is not None, "p_target": p_target,
            "p_agg": p_agg, "agg_literals": agg_literals,
            "top_k": [{"token": t, "p": p} for t, p in top],
        })

    print()
    print(f"Unscoreable (no single-token form): {len(multi_token)}")
    for pid, t in multi_token:
        print(f"  {pid}: {t!r}")

    print(f"Below threshold p_agg={args.min_prob}: {len(failures)}")
    for pid, lit, p in failures:
        print(f"  {pid}: p_agg({lit!r}-class)={p:.3f}")

    pair_check(prompts, results, args.min_prob)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model_name, "results": results}, f, indent=2)
        print(f"\nWrote per-prompt results to {args.output}")


def pair_check(prompts: list[dict], results: list[dict], min_prob: float) -> None:
    """If ids end in _pos/_neg, report per-pair both-pass rate (on p_agg)."""
    by_stem: dict[str, dict[str, dict]] = {}
    for r in results:
        pid = r["id"]
        for suffix in ("_pos", "_neg"):
            if pid.endswith(suffix):
                stem = pid[: -len(suffix)]
                by_stem.setdefault(stem, {})[suffix] = r
                break
    if not by_stem:
        return
    print()
    print("Pair check (both sides must PASS for the pair to be usable):")
    both_pass = 0
    for stem, pair in by_stem.items():
        pos, neg = pair.get("_pos"), pair.get("_neg")
        if pos is None or neg is None:
            print(f"  {stem}: MISSING half ({list(pair.keys())})")
            continue
        ok_pos = (pos.get("p_agg") or 0) >= min_prob
        ok_neg = (neg.get("p_agg") or 0) >= min_prob
        status = "BOTH" if ok_pos and ok_neg else ("POS_ONLY" if ok_pos else ("NEG_ONLY" if ok_neg else "NEITHER"))
        if ok_pos and ok_neg:
            both_pass += 1
        pp, np_ = pos.get("p_agg"), neg.get("p_agg")
        print(f"  {stem:<30} {status:<10} "
              f"pos p_agg={pp:.3f}  neg p_agg={np_:.3f}"
              if pp is not None and np_ is not None else
              f"  {stem:<30} {status:<10} pos p_agg={pp!r}  neg p_agg={np_!r}")
    print(f"  → {both_pass}/{len(by_stem)} pairs usable")


if __name__ == "__main__":
    main()
