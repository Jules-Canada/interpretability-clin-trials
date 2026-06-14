"""
scripts/stage0_tokenizer_check.py

Stage 0 pre-flight for the contrastive eligibility prompts (CLAUDE.md Phase 2,
active path = circuit-tracer + Gemma Scope 2). Tokenizer-only: no GPU, no model
weights, no circuit-tracer. Runs on a laptop CPU in seconds.

Confirms two things before any compute is spent:

  1. " Yes" / " No" (and variants) are SINGLE clean tokens in the Gemma-3
     tokenizer -> decides POS_TOKEN / NEG_TOKEN in prompts/eligibility.py.
     The Gemma-3 4B family (it / pt / MedGemma) shares one tokenizer, so this
     runs against the already-cached google/medgemma-4b-pt — no new download.

  2. The canonical Gemma-3 IT chat template matches our hand-rolled
     to_gemma_chat(), so the Yes/No logit is read at the right position. The
     chat template lives only on the IT model, so this step needs
     google/gemma-3-4b-it (separately gated — accept its license once). If that
     model isn't accessible yet, this step is SKIPPED with instructions and the
     token check still runs.

Run from the repo root:
    python3 scripts/stage0_tokenizer_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from prompts...` when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer  # noqa: E402

TOKEN_MODEL = "google/medgemma-4b-pt"  # cached; same vocab as gemma-3-4b-it
IT_MODEL = "google/gemma-3-4b-it"      # carries the chat template

CANDIDATES = ["Yes", " Yes", "No", " No", "yes", " yes", "no", " no"]


def check_tokens() -> None:
    print(f"# single-token check  (vocab from {TOKEN_MODEL})")
    tok = AutoTokenizer.from_pretrained(TOKEN_MODEL)
    for s in CANDIDATES:
        ids = tok.encode(s, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        flag = "OK single" if len(ids) == 1 else f"!! {len(ids)} tokens"
        print(f"  {s!r:8} -> ids={ids} pieces={pieces}  {flag}")


def check_template() -> None:
    from prompts.eligibility import ELIGIBILITY_PAIRS, build_body, to_gemma_chat

    p = ELIGIBILITY_PAIRS[0]
    print(f"\n# chat-template check  ({IT_MODEL})")
    try:
        tok = AutoTokenizer.from_pretrained(IT_MODEL)
    except Exception as e:  # gated / not accepted / offline
        print(f"  SKIPPED — could not load {IT_MODEL}:")
        print(f"  {type(e).__name__}: {e}")
        print("  -> accept the license once at "
              "https://huggingface.co/google/gemma-3-4b-it")
        return

    canonical = tok.apply_chat_template(
        [{"role": "user", "content": build_body(p)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ours = to_gemma_chat(p)
    print("--- canonical (apply_chat_template) ---")
    print(repr(canonical))
    print("--- ours (to_gemma_chat) ---")
    print(repr(ours))
    if canonical == ours:
        print("MATCH — to_gemma_chat is faithful")
    else:
        print("DIFFER — switch to_gemma_chat() to use apply_chat_template "
              "(never hand-roll IT formatting)")


if __name__ == "__main__":
    check_tokens()
    check_template()
