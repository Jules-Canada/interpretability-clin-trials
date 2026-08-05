"""
scripts/screen_contrastive.py — contrastive logit-difference screening.

For binary-decision prompts (pos/neg pairs), the relevant signal is NOT
absolute p(target) but the logit difference between answer classes:

    logit_diff(prompt) = mean(logit[Yes_variants]) − mean(logit[No_variants])

A pair separates if:
    gap = logit_diff(pos_prompt) − logit_diff(neg_prompt) > 0

This is the standard metric for binary-decision circuit analysis (IOI, sparse
feature circuits) and the correct screening criterion for contrastive
attribution graphs.  Absolute softmax peakiness is irrelevant — a flat base
model can still separate pairs on the logit difference.

Usage:
    python scripts/screen_contrastive.py \
        --prompts prompts/categorical_prompts.py \
        --var_name CATEGORICAL_PROMPTS \
        --model_name google/medgemma-4b-pt \
        --dtype float32 \
        --output data/contrastive_screen.json

    # Also screen easy controls for dynamic-range comparison:
    python scripts/screen_contrastive.py \
        --prompts prompts/categorical_prompts.py \
        --var_name EASY_INCLUSION_PROMPTS \
        --model_name google/medgemma-4b-pt \
        --dtype float32 \
        --output data/contrastive_screen_easy.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
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


def surface_variants(word: str) -> list[str]:
    base = word.strip()
    return sorted({base, base.lower(), base.upper(), base.capitalize()})


def resolve_class_ids(tok, text: str, word: str) -> list[int]:
    """All single-token IDs for casing variants of `word`, with leading-space handling."""
    want_space = not text.endswith((" ", "\n", "\t"))
    ids: list[int] = []
    seen: set[int] = set()
    for form in surface_variants(word):
        for cand in ([" " + form, form] if want_space else [form]):
            enc = tok.encode(cand, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in seen:
                seen.add(enc[0])
                ids.append(enc[0])
    return ids


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
    p.add_argument("--prompts", required=True)
    p.add_argument("--var_name", default="CATEGORICAL_PROMPTS")
    p.add_argument("--model_name", default="google/medgemma-4b-pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output", default=None)
    p.add_argument("--pos_word", default="Yes", help="Positive answer class word")
    p.add_argument("--neg_word", default="No", help="Negative answer class word")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prompts = load_prompts(args.prompts, args.var_name)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}::{args.var_name}")

    device = pick_device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    if device.type == "mps" and dtype is torch.float16:
        print("WARNING: MPS + float16 yields NaN on Gemma 3 — switching to bfloat16")
        dtype = torch.bfloat16
    print(f"Device: {device} | dtype: {dtype}")

    print(f"Loading tokenizer + model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)\n")

    # Resolve answer-class token IDs using a representative prompt text
    sample_text = prompt_text(prompts[0])
    pos_ids = resolve_class_ids(tok, sample_text, args.pos_word)
    neg_ids = resolve_class_ids(tok, sample_text, args.neg_word)
    print(f"Positive class ({args.pos_word}): {len(pos_ids)} token IDs "
          f"{[tok.decode([i]) for i in pos_ids]}")
    print(f"Negative class ({args.neg_word}): {len(neg_ids)} token IDs "
          f"{[tok.decode([i]) for i in neg_ids]}\n")

    if not pos_ids or not neg_ids:
        print("ERROR: could not resolve answer-class tokens. Aborting.")
        return

    pos_idx = torch.tensor(pos_ids, device=device)
    neg_idx = torch.tensor(neg_ids, device=device)

    # Compute logit_diff for each prompt
    results: list[dict] = []

    header = f"{'id':<34} {'logit_diff':>10} {'p_pos':>7} {'p_neg':>7}   top-3"
    print(header)
    print("-" * (len(header) + 30))

    for prompt in prompts:
        pid = prompt["id"]
        text = prompt_text(prompt)

        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()

        logit_pos = logits[pos_idx].mean().item()
        logit_neg = logits[neg_idx].mean().item()
        logit_diff = logit_pos - logit_neg

        probs = torch.softmax(logits, dim=-1)
        p_pos = probs[pos_idx].sum().item()
        p_neg = probs[neg_idx].sum().item()

        top_p, top_i = probs.topk(3)
        top = [(tok.decode([i]), float(p)) for i, p in zip(top_i.tolist(), top_p.tolist())]
        top_str = ", ".join(f"{repr(t)}={p:.3f}" for t, p in top)

        print(f"{pid:<34} {logit_diff:>+10.3f} {p_pos:>7.3f} {p_neg:>7.3f}   {top_str}")

        results.append({
            "id": pid,
            "logit_diff": logit_diff,
            "logit_pos": logit_pos,
            "logit_neg": logit_neg,
            "p_pos": p_pos,
            "p_neg": p_neg,
            "top_3": [{"token": t, "p": p} for t, p in top],
        })

    # Pair analysis
    print()
    pair_results = pair_analysis(results)

    # Summary stats
    if pair_results:
        gaps = [pr["gap"] for pr in pair_results]
        correct = sum(1 for g in gaps if g > 0)
        print(f"\nSummary: {correct}/{len(gaps)} pairs separate correctly (gap > 0)")
        print(f"  mean gap: {sum(gaps)/len(gaps):+.3f}")
        print(f"  min gap:  {min(gaps):+.3f}")
        print(f"  max gap:  {max(gaps):+.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model_name,
                "dtype": str(dtype),
                "device": str(device),
                "pos_word": args.pos_word,
                "neg_word": args.neg_word,
                "pos_ids": pos_ids,
                "neg_ids": neg_ids,
                "per_prompt": results,
                "pairs": pair_results,
            }, f, indent=2)
        print(f"\nWrote results to {args.output}")


def pair_analysis(results: list[dict]) -> list[dict]:
    """Match _pos/_neg pairs and report logit-diff gap."""
    by_stem: dict[str, dict[str, dict]] = {}
    for r in results:
        pid = r["id"]
        for suffix in ("_pos", "_neg"):
            if pid.endswith(suffix):
                stem = pid[: -len(suffix)]
                by_stem.setdefault(stem, {})[suffix] = r
                break

    if not by_stem:
        return []

    print("Pair separation (gap = logit_diff_pos − logit_diff_neg; want > 0):")
    pair_results: list[dict] = []

    for stem, pair in sorted(by_stem.items()):
        pos, neg = pair.get("_pos"), pair.get("_neg")
        if pos is None or neg is None:
            print(f"  {stem}: MISSING half")
            continue

        gap = pos["logit_diff"] - neg["logit_diff"]
        sign = "OK" if gap > 0 else "FLIP"

        print(f"  {stem:<30} {sign:<5} gap={gap:+.3f}  "
              f"(pos_ld={pos['logit_diff']:+.3f}  neg_ld={neg['logit_diff']:+.3f})")

        pair_results.append({
            "stem": stem,
            "gap": gap,
            "pos_logit_diff": pos["logit_diff"],
            "neg_logit_diff": neg["logit_diff"],
            "separates": gap > 0,
        })

    return pair_results


if __name__ == "__main__":
    main()
