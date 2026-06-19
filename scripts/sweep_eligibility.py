#!/usr/bin/env python
"""
scripts/sweep_eligibility.py

Stage 1 behavioral GATE (forward-pass only — no circuit-tracer needed).

The hosted Neuronpedia sweep (raw-cloze, OOD) showed gemma-3-4b-it ignoring the
stated age bounds (effective band ~[16,97] vs stated [18,75]). This script
re-tests that behavior under the CORRECT Gemma-3 chat template, to decide whether
there is a coherent decision worth graphing before spending GPU on attribution.

It measures p(Yes)/p(No) (summed over single-token surface variants) at the
generation position after the chat template, over:
  (a) the 5 ELIGIBILITY_PAIRS,
  (b) an age sweep against a fixed "age 18-75" criterion,
  (c) bound-variation probes (fixed age 60, varied criterion) — does the model
      read the stated bounds at all, or run on a generic adult-age prior?

Gate: if the chat-formatted boundary snaps to ~[18,75], graph the now-coherent
decision (run_graphs_ct.py). If it stays incoherent, that incoherence is the
finding and the prompts need a redesign before more compute.

Usage:
    python scripts/sweep_eligibility.py --model google/gemma-3-4b-it
    python scripts/sweep_eligibility.py --model google/medgemma-4b-it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path so `prompts` imports when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Route the HF cache to /workspace on pods (20GB root disk fills otherwise).
# Must be set before transformers/huggingface_hub import. No-op on the Mac.
import os
os.environ.setdefault(
    "HF_HOME",
    "/workspace/.cache/huggingface" if os.path.isdir("/workspace")
    else os.path.expanduser("~/.cache/huggingface"),
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts.eligibility import ELIGIBILITY_PAIRS, build_body, to_chat


def resolve_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Single-token surface variants for Yes / No in this tokenizer."""
    def variants(word: str) -> list[int]:
        forms = sorted({word, word.lower(), word.upper(), word.capitalize()})
        ids: list[int] = []
        seen: set[int] = set()
        for form in forms:
            for cand in (" " + form, form):
                enc = tokenizer.encode(cand, add_special_tokens=False)
                if len(enc) == 1 and enc[0] not in seen:
                    seen.add(enc[0])
                    ids.append(enc[0])
        return ids

    pos, neg = variants("Yes"), variants("No")
    if not pos or not neg:
        raise ValueError("Could not resolve single-token Yes/No ids for this tokenizer")
    return pos, neg


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def decision(model, tokenizer, criterion: str, patient: str,
             pos_ids: list[int], neg_ids: list[int], device) -> dict:
    """p(Yes)/p(No) at the generation position for one criterion+patient."""
    p = {"criterion": criterion, "patient": patient}
    chat = to_chat(tokenizer, p)  # includes <bos> via the template
    ids = tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(device)
    logits = model(**ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p_yes = probs[pos_ids].sum().item()
    p_no = probs[neg_ids].sum().item()
    top_id = int(logits.argmax())
    return {
        "p_yes": round(p_yes, 4),
        "p_no": round(p_no, 4),
        "says": "Yes" if p_yes > p_no else "No",
        "top_token": tokenizer.decode([top_id]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavioral gate for eligibility prompts.")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="JSON output path (default data/eligibility_sweep_<model>.json)")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Model:  {args.model}\nDevice: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    pos_ids, neg_ids = resolve_yes_no_ids(tokenizer)
    print(f"Yes ids={pos_ids}  No ids={neg_ids}\n")

    results: dict = {"model": args.model, "pairs": [], "age_sweep": [], "bound_variation": []}

    def run(crit, pat):
        return decision(model, tokenizer, crit, pat, pos_ids, neg_ids, device)

    # (a) the 5 matched pairs ------------------------------------------------
    print("=== (a) ELIGIBILITY_PAIRS ===")
    for p in ELIGIBILITY_PAIRS:
        d = run(p["criterion"], p["patient"])
        correct = d["says"] == p["expected"]
        row = {"id": p["id"], "expected": p["expected"], **d, "correct": correct}
        results["pairs"].append(row)
        flag = "ok " if correct else "XX "
        print(f"  {flag}{p['id']:22} exp={p['expected']:3} says={d['says']:3} "
              f"p(Yes)={d['p_yes']:.2f} p(No)={d['p_no']:.2f}")

    # (b) age sweep against fixed 18-75 criterion ----------------------------
    print("\n=== (b) age sweep  (criterion: age 18-75) ===")
    crit = "Inclusion: age 18-75 years."
    for age in [5, 10, 15, 16, 17, 18, 20, 25, 50, 70, 73, 75, 76, 80, 90, 95, 99, 105]:
        pat = f"{age}-year-old with stage IIIA non-small cell lung cancer."
        d = run(crit, pat)
        truth = "Yes" if 18 <= age <= 75 else "No"
        row = {"age": age, "truth": truth, **d, "correct": d["says"] == truth}
        results["age_sweep"].append(row)
        flag = "ok " if d["says"] == truth else "XX "
        print(f"  {flag}age={age:3} truth={truth:3} says={d['says']:3} p(Yes)={d['p_yes']:.2f}")

    # (c) bound-variation: fixed age 60, move the stated bounds ---------------
    print("\n=== (c) bound variation  (patient age 60) ===")
    pat = "60-year-old with stage IIIA non-small cell lung cancer."
    for lo, hi in [(18, 40), (65, 90), (18, 90), (18, 75)]:
        crit = f"Inclusion: age {lo}-{hi} years."
        d = run(crit, pat)
        truth = "Yes" if lo <= 60 <= hi else "No"
        row = {"bounds": [lo, hi], "truth": truth, **d, "correct": d["says"] == truth}
        results["bound_variation"].append(row)
        flag = "ok " if d["says"] == truth else "XX "
        print(f"  {flag}{lo}-{hi:<3} truth={truth:3} says={d['says']:3} p(Yes)={d['p_yes']:.2f}")

    # save -------------------------------------------------------------------
    slug = args.model.split("/")[-1]
    out = Path(args.out) if args.out else Path("data") / f"eligibility_sweep_{slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    n_ok = sum(r["correct"] for r in results["age_sweep"])
    print(f"\nGate read: age sweep {n_ok}/{len(results['age_sweep'])} correct. "
          "If the boundary now tracks [18,75], proceed to run_graphs_ct.py.")


if __name__ == "__main__":
    main()
