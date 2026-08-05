#!/usr/bin/env python
"""
scripts/run_ecog_stimuli.py

Pillar-1 probe, forward-pass only: does the model compute the ECOG INTERMEDIATE,
or does it answer the eligibility question some other way?

Each stimulus row is run as TWO INDEPENDENT prompts (separate forward passes, no
shared context, neither turn sees the other's answer):

  (1) ELIGIBILITY — protocol requires ECOG <=1, here is the patient, Yes/No.
  (2) GRADING     — same patient, what is the ECOG grade, 0-4.

The pair is the point. Eligibility-correct + grading-wrong is a model that gets
the answer without the abstraction the spec is written in; that dissociation is
the pillar-1 measurement, not the eligibility accuracy on its own.

Both prompts go through the IT chat template (CLAUDE.md: IT model -> chat
template, always). Prompt 1 reuses prompts.eligibility.build_body so the
eligibility wording is byte-identical to the rest of the program and the numbers
stay comparable to sweep_eligibility.py.

READOUT (CLAUDE.md token rule)
  Primary p(Yes)/p(No) is the CANONICAL LEADING-SPACE token only (" Yes", " No")
  — the attribution-grade readout. The case/spacing-variant sum is reported
  alongside as *_agg for evaluation only; never mix the two. Grades likewise:
  " 0".." 4" canonical, bare "0".."4" aggregated separately.

WHAT ecog_v0.csv ENCODES, AND HOW IT IS SCORED
  lexical_distance  verbatim | near | far  — `far` is the load-bearing arm
                    (CLAUDE.md: clinical notes never use protocol phrasing).
  set_id            S01..S05 group near/far restatements of the SAME patient.
                    A model holding the concept must return the same grade
                    across a set; drift is paraphrase-generalisation failure.
  decisive          the grade actually settles eligibility (grade >=2 -> No).
  ambiguous         the GRADE is contestable (e.g. 0-vs-1) while eligibility is
                    not. Those rows are excluded from headline grading accuracy
                    and reported separately — scoring a contestable grade as
                    wrong would manufacture a pillar-1 gap that isn't there.

Usage:
    python scripts/run_ecog_stimuli.py --model google/medgemma-4b-it
    python scripts/run_ecog_stimuli.py --model google/gemma-3-4b-it
    python scripts/run_ecog_stimuli.py --dry-run      # render prompts, no model

Output is model-slugged (data/ecog_v0_results_<slug>.json), so the two runs above
land side by side rather than overwriting each other. Run the base-Gemma
comparator in the same session as MedGemma: per CLAUDE.md the Round-1
MedGemma-vs-Gemma claims failed re-verification, and a MedGemma number without a
matched baseline from the same stimuli can't say anything about medical tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

from prompts.eligibility import build_body

DEFAULT_STIMULI = Path("specs/stimuli/ecog_v0.csv")
# Output is model-slugged (as in sweep_eligibility.py) so running the MedGemma
# and Gemma comparators back to back on the same pod cannot clobber each other.
OUT_TEMPLATE = "data/ecog_v0_results_{slug}.json"

GRADES = [0, 1, 2, 3, 4]
LEX_ORDER = {"verbatim": 0, "near": 1, "far": 2}   # print order, easy -> hard

# Grading-turn wording. Deliberately NOT parallel to the eligibility QUESTION:
# this prompt names the intermediate ("ECOG performance status") because we are
# asking the model to report it. The vignette itself must never name it — that is
# the stimulus-side rule (CLAUDE.md: vary only the clinical description).
GRADE_INSTRUCTION = "You are assessing a patient for a clinical trial."
GRADE_QUESTION = (
    "What is the patient's ECOG performance status grade? "
    "Answer with a single digit: 0, 1, 2, 3, or 4."
)

# Defining vocabulary a vignette must NOT contain. A hit means the row leaks the
# abstraction and both prompts are testing lookup, not recovery.
DEFINING_VOCAB = re.compile(r"\becog\b|\bperformance status\b|\bkarnofsky\b", re.I)

TRUTHY = {"yes", "y", "true", "1"}


def norm_key(k: str) -> str:
    """CSV headers carry stray spaces and trailing colons; normalise to a slug."""
    return re.sub(r"[^a-z0-9]+", "_", (k or "").strip().lower()).strip("_")


def as_bool(v: str) -> bool:
    return (v or "").strip().lower() in TRUTHY


def load_stimuli(path: Path) -> list[dict]:
    """Read ecog_v0.csv into rows with stable field names.

    Tolerates the header variants seen across revisions of this file
    (`patient_detail`/`patient`, `inclusion_rule`/`inclusion`).
    """
    with path.open(newline="", encoding="utf-8-sig") as fh:
        raw_rows = list(csv.DictReader(fh))

    rows: list[dict] = []
    for i, raw in enumerate(raw_rows):
        r = {norm_key(k): (v or "").strip() for k, v in raw.items() if k is not None}
        patient = r.get("patient_detail") or r.get("patient") or ""
        if not patient:
            continue  # skip blank/trailing lines
        grade_raw = r.get("expected_ecog", "")
        expected_incl = (r.get("expected_inclusion") or "").strip().capitalize() or None
        rows.append({
            "id": r.get("id") or f"row{i:03d}",
            "set_id": r.get("set_id") or None,
            "criterion_text": r.get("inclusion_rule") or r.get("inclusion") or "ECOG <= 1",
            "patient": patient,
            "expected_grade": int(grade_raw) if grade_raw.isdigit() else None,
            "expected_eligible": expected_incl,          # "Yes" | "No" | None
            "lexical_distance": (r.get("lexical_distance") or "unknown").lower(),
            "decisive": as_bool(r.get("decisive", "")),
            "ambiguous": as_bool(r.get("ambiguous", "")),
            "notes": r.get("notes") or "",
            "leaks_vocab": bool(DEFINING_VOCAB.search(patient)),
        })
    return rows


def eligibility_chat(tokenizer, row: dict) -> str:
    """Prompt 1. Reuses build_body so wording matches the rest of the program."""
    body = build_body({
        "criterion": f"Inclusion: {row['criterion_text']}.",
        "patient": row["patient"],
    })
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True,
    )


def grading_chat(tokenizer, row: dict) -> str:
    """Prompt 2. Independent turn — no eligibility criterion, no prior answer."""
    body = (
        f"{GRADE_INSTRUCTION}\n"
        f"Patient: {row['patient']}\n"
        f"{GRADE_QUESTION}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True,
    )


def single_token_id(tokenizer, s: str) -> int | None:
    enc = tokenizer.encode(s, add_special_tokens=False)
    return enc[0] if len(enc) == 1 else None


def resolve_tokens(tokenizer) -> dict:
    """Canonical leading-space ids (primary) + variant ids (evaluation only)."""
    def variants(word: str) -> list[int]:
        forms = sorted({word, word.lower(), word.upper(), word.capitalize()})
        ids, seen = [], set()
        for form in forms:
            for cand in (" " + form, form):
                tid = single_token_id(tokenizer, cand)
                if tid is not None and tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
        return ids

    tok = {
        "yes": single_token_id(tokenizer, " Yes"),
        "no": single_token_id(tokenizer, " No"),
        "yes_agg": variants("Yes"),
        "no_agg": variants("No"),
        "grade": {g: single_token_id(tokenizer, f" {g}") for g in GRADES},
        "grade_agg": {g: [t for t in (single_token_id(tokenizer, f" {g}"),
                                      single_token_id(tokenizer, f"{g}"))
                          if t is not None] for g in GRADES},
    }
    if tok["yes"] is None or tok["no"] is None:
        raise ValueError("Canonical ' Yes'/' No' are not single tokens in this tokenizer")
    missing = [g for g, t in tok["grade"].items() if t is None]
    if missing:
        raise ValueError(f"Canonical grade digits {missing} are not single tokens")
    return tok


def pick_device(arg: str):
    import torch
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def next_token_logits(model, tokenizer, chat: str, device):
    """Logits at the generation position. chat already carries <bos>.

    returns: (vocab,) float32
    """
    import torch
    ids = tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids).logits          # (1, seq, vocab)
    return out[0, -1].float()              # (vocab,)


def run_eligibility(model, tokenizer, row, tok, device) -> dict:
    import torch
    logits = next_token_logits(model, tokenizer, eligibility_chat(tokenizer, row), device)
    probs = torch.softmax(logits, dim=-1)  # (vocab,)
    p_yes = probs[tok["yes"]].item()
    p_no = probs[tok["no"]].item()
    return {
        "p_yes": round(p_yes, 5),                     # canonical " Yes" only
        "p_no": round(p_no, 5),                       # canonical " No" only
        "logit_diff": round((logits[tok["yes"]] - logits[tok["no"]]).item(), 4),
        "p_yes_agg": round(probs[tok["yes_agg"]].sum().item(), 5),   # eval only
        "p_no_agg": round(probs[tok["no_agg"]].sum().item(), 5),     # eval only
        "says": "Yes" if p_yes > p_no else "No",
        "top_token": tokenizer.decode([int(logits.argmax())]),
    }


def run_grading(model, tokenizer, row, tok, device) -> dict:
    import torch
    logits = next_token_logits(model, tokenizer, grading_chat(tokenizer, row), device)
    probs = torch.softmax(logits, dim=-1)  # (vocab,)
    canon = {g: probs[tok["grade"][g]].item() for g in GRADES}
    agg = {g: probs[tok["grade_agg"][g]].sum().item() for g in GRADES}
    mass = sum(canon.values())
    dist = {g: (canon[g] / mass if mass > 0 else 0.0) for g in GRADES}
    pred = max(dist, key=dist.get)
    return {
        # renormalised over the 5 canonical grade tokens — the grade distribution
        "grade_dist": {str(g): round(dist[g], 5) for g in GRADES},
        "grade_p_raw": {str(g): round(canon[g], 6) for g in GRADES},
        "grade_p_agg": {str(g): round(agg[g], 6) for g in GRADES},
        # how much of the full next-token distribution lands on a grade digit at
        # all; low mass = the model is not answering in the requested format and
        # the argmax below is an artefact of renormalising noise
        "grade_token_mass": round(mass, 5),
        "pred_grade": pred,
        "pred_eligible_from_grade": "Yes" if pred <= 1 else "No",
        "top_token": tokenizer.decode([int(logits.argmax())]),
    }


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:5.1f}%)" if d else "     -    "


def breakdown(rows: list[dict], key) -> dict:
    """Group by key(row). Grading counts exclude ambiguous rows from the headline
    and tally them under grade_ambiguous_* instead."""
    out: dict = {}
    for r in rows:
        c = out.setdefault(str(key(r)), {
            "n": 0, "elig_n": 0, "elig_correct": 0,
            "grade_n": 0, "grade_correct": 0,
            "grade_ambiguous_n": 0, "grade_ambiguous_correct": 0,
        })
        c["n"] += 1
        if r.get("eligibility_correct") is not None:
            c["elig_n"] += 1
            c["elig_correct"] += int(r["eligibility_correct"])
        if r.get("grade_correct") is not None:
            bucket = "grade_ambiguous" if r["ambiguous"] else "grade"
            c[f"{bucket}_n"] += 1
            c[f"{bucket}_correct"] += int(r["grade_correct"])
    return out


def print_breakdown(title: str, table: dict, label: str, order=None) -> None:
    keys = sorted(table, key=order) if order else sorted(table)
    print(f"\n{title}")
    print(f"  {label:<10} {'n':>3}  {'eligibility':>16} {'grading':>16} {'grading(amb)':>16}")
    for k in keys:
        c = table[k]
        print(f"  {k:<10} {c['n']:>3}  {pct(c['elig_correct'], c['elig_n']):>16} "
              f"{pct(c['grade_correct'], c['grade_n']):>16} "
              f"{pct(c['grade_ambiguous_correct'], c['grade_ambiguous_n']):>16}")


def paraphrase_sets(rows: list[dict]) -> dict:
    """Within-set consistency: the load-bearing paraphrase-generalisation test.

    Members of a set_id are restatements of the SAME patient at different lexical
    distance, so the ground-truth grade is identical by construction. A model
    holding the concept returns one grade across the set; one that pattern-matches
    the scale wording drifts as the phrasing moves away from it.
    """
    sets: dict = {}
    for r in rows:
        if r["set_id"]:
            sets.setdefault(r["set_id"], []).append(r)
    out: dict = {}
    for sid, members in sets.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda m: LEX_ORDER.get(m["lexical_distance"], 9))
        grades = [m["grading"]["pred_grade"] for m in members]
        answers = [m["eligibility"]["says"] for m in members]
        out[sid] = {
            "expected_grade": members[0]["expected_grade"],
            "members": [
                {"id": m["id"], "lexical_distance": m["lexical_distance"],
                 "pred_grade": m["grading"]["pred_grade"],
                 "says": m["eligibility"]["says"],
                 "logit_diff": m["eligibility"]["logit_diff"]}
                for m in members
            ],
            "grade_consistent": len(set(grades)) == 1,
            "eligibility_consistent": len(set(answers)) == 1,
            "grade_drift": max(grades) - min(grades),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ECOG stimuli: eligibility vs grading.")
    ap.add_argument("--model", default="google/medgemma-4b-it")
    ap.add_argument("--stimuli", default=str(DEFAULT_STIMULI))
    ap.add_argument("--out", default=None,
                    help=f"JSON output path (default {OUT_TEMPLATE})")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="first N rows only")
    ap.add_argument("--dry-run", action="store_true",
                    help="render prompts and stimulus parse, load no model")
    ap.add_argument("--write-csv", nargs="?", const="AUTO", default=None,
                    help="also write a copy of the CSV with model_inclusion / "
                         "model_ecog filled in; bare flag uses "
                         "data/ecog_v0_scored_<slug>.csv")
    args = ap.parse_args()

    stim_path = Path(args.stimuli)
    if not stim_path.exists():
        sys.exit(f"Stimulus file not found: {stim_path}")
    rows = load_stimuli(stim_path)
    if args.limit:
        rows = rows[:args.limit]

    model_slug = args.model.split("/")[-1]
    n_far = sum(r["lexical_distance"] == "far" for r in rows)
    n_amb = sum(r["ambiguous"] for r in rows)
    print(f"Stimuli: {stim_path}  ({len(rows)} rows, {n_far} far, {n_amb} ambiguous)")

    # Design warnings — these change how the numbers may be read, so they print
    # before the run, not buried after it.
    warnings: list[str] = []
    if len({r["expected_eligible"] for r in rows}) < 2:
        warnings.append("Only one eligibility class present: accuracy is degenerate.")
    grades_present = sorted({r["expected_grade"] for r in rows
                             if r["expected_grade"] is not None})
    missing_grades = [g for g in GRADES if g not in grades_present]
    if missing_grades:
        warnings.append(f"No stimuli at grade(s) {missing_grades}; grading accuracy "
                        "says nothing about that end of the scale.")
    leaks = [r["id"] for r in rows if r["leaks_vocab"]]
    if leaks:
        warnings.append("Vignettes naming the defining vocabulary (tests lookup, not "
                        f"recovery): {', '.join(leaks)}")
    unknown_lex = [r["id"] for r in rows if r["lexical_distance"] not in LEX_ORDER]
    if unknown_lex:
        warnings.append(f"Unannotated lexical_distance on: {', '.join(unknown_lex)}")
    for w in warnings:
        print(f"  !! {w}")

    if args.dry_run:
        class _T:  # stand-in so prompts render without downloading a model
            @staticmethod
            def apply_chat_template(msgs, tokenize=False, add_generation_prompt=True):
                return (f"<bos><start_of_turn>user\n{msgs[0]['content']}"
                        f"<end_of_turn>\n<start_of_turn>model\n")
        for r in rows:
            print(f"\n--- {r['id']} set={r['set_id']} lex={r['lexical_distance']} "
                  f"decisive={r['decisive']} ambiguous={r['ambiguous']} "
                  f"expected: elig={r['expected_eligible']} grade={r['expected_grade']}")
            print("[1 eligibility]\n" + eligibility_chat(_T, r))
            print("[2 grading]\n" + grading_chat(_T, r))
        print("\nDry run: no model loaded, nothing written.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(args.device)
    print(f"\nModel:  {args.model}\nDevice: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    tok = resolve_tokens(tokenizer)
    print(f"Canonical ids: ' Yes'={tok['yes']} ' No'={tok['no']} "
          f"grades={ {g: tok['grade'][g] for g in GRADES} }")

    for r in rows:
        r["eligibility"] = run_eligibility(model, tokenizer, r, tok, device)
        r["grading"] = run_grading(model, tokenizer, r, tok, device)
        r["eligibility_correct"] = (
            r["eligibility"]["says"] == r["expected_eligible"]
            if r["expected_eligible"] else None)
        r["grade_correct"] = (
            r["grading"]["pred_grade"] == r["expected_grade"]
            if r["expected_grade"] is not None else None)
        # Does the eligibility answer agree with the model's OWN grade? A model
        # routing through the intermediate should be self-consistent even where
        # its grade is wrong; disagreement means the two turns are computed by
        # different paths.
        r["self_consistent"] = (
            r["eligibility"]["says"] == r["grading"]["pred_eligible_from_grade"])
        # The pillar-1 dissociation: right answer, wrong intermediate.
        ec, gc = r["eligibility_correct"], r["grade_correct"]
        r["dissociation"] = (
            "incomplete" if ec is None or gc is None
            else "both_right" if ec and gc
            else "right_answer_wrong_grade" if ec and not gc
            else "wrong_answer_right_grade" if gc
            else "both_wrong")

    print("\n=== per-stimulus ===")
    for r in rows:
        e, g = r["eligibility"], r["grading"]
        ef = "ok " if r["eligibility_correct"] else "XX "
        gf = ("amb" if r["ambiguous"] else "ok ") if r["grade_correct"] else \
             ("amb" if r["ambiguous"] else "XX ")
        tags = "".join([" D" if r["decisive"] else "  ", "A" if r["ambiguous"] else " "])
        print(f"  {r['id']} set={str(r['set_id'] or '-'):<4} [{r['lexical_distance']:<8}]"
              f"{tags} exp: elig={str(r['expected_eligible']):<3} grade={r['expected_grade']}")
        print(f"    {ef}elig  says={e['says']:<3} p(Yes)={e['p_yes']:.3f} "
              f"p(No)={e['p_no']:.3f} dlogit={e['logit_diff']:+.2f}")
        print(f"    {gf}grade pred={g['pred_grade']} "
              f"dist=[{' '.join(f'{k}:{v:.2f}' for k, v in g['grade_dist'].items())}] "
              f"mass={g['grade_token_mass']:.3f}"
              f"{'' if r['self_consistent'] else '  <- answer disagrees with own grade'}")
        print(f"       {r['patient'][:92]}")

    scored = [r for r in rows if not r["ambiguous"]]
    elig_n = sum(1 for r in rows if r["eligibility_correct"] is not None)
    elig_ok = sum(1 for r in rows if r["eligibility_correct"])
    grade_n = sum(1 for r in scored if r["grade_correct"] is not None)
    grade_ok = sum(1 for r in scored if r["grade_correct"])
    amb_n = sum(1 for r in rows if r["ambiguous"] and r["grade_correct"] is not None)
    amb_ok = sum(1 for r in rows if r["ambiguous"] and r["grade_correct"])
    self_ok = sum(1 for r in rows if r["self_consistent"])

    print("\n=== overall ===")
    print(f"  eligibility           {pct(elig_ok, elig_n)}   (all rows)")
    print(f"  grading               {pct(grade_ok, grade_n)}   (unambiguous rows only)")
    print(f"  grading (ambiguous)   {pct(amb_ok, amb_n)}   reported, not scored")
    print(f"  answer/grade agree    {pct(self_ok, len(rows))}")

    by_grade = breakdown(rows, lambda r: r["expected_grade"])
    by_lex = breakdown(rows, lambda r: r["lexical_distance"])
    by_dec = breakdown(rows, lambda r: "decisive" if r["decisive"] else "non-decisive")
    print_breakdown("=== accuracy by expected grade ===", by_grade, "grade")
    print_breakdown("=== accuracy by lexical distance ===", by_lex, "distance",
                    order=lambda k: LEX_ORDER.get(k, 9))
    print_breakdown("=== accuracy by decisiveness ===", by_dec, "kind")

    sets = paraphrase_sets(rows)
    if sets:
        print("\n=== paraphrase generalisation (same patient, by set_id) ===")
        for sid in sorted(sets):
            s = sets[sid]
            flag = "ok " if s["grade_consistent"] else "XX "
            members = "  ".join(f"{m['lexical_distance']}={m['pred_grade']}"
                                for m in s["members"])
            print(f"  {flag}{sid} truth={s['expected_grade']}  {members}"
                  f"  (drift={s['grade_drift']})")
        n_ok = sum(s["grade_consistent"] for s in sets.values())
        print(f"  grade held constant across {n_ok}/{len(sets)} sets")

    diss: dict = {}
    for r in rows:
        diss[r["dissociation"]] = diss.get(r["dissociation"], 0) + 1
    print("\n=== eligibility x grading (the pillar-1 read) ===")
    for k in ("both_right", "right_answer_wrong_grade",
              "wrong_answer_right_grade", "both_wrong", "incomplete"):
        if k in diss:
            print(f"  {k:<26} {diss[k]}")
    if diss.get("right_answer_wrong_grade"):
        print("  -> answers the eligibility question without recovering the grade "
              "it is defined over. That gap is the pillar-1 signal.")

    results = {
        "model": args.model,
        "stimuli_file": str(stim_path),
        "n_rows": len(rows),
        "scoring": {
            "eligibility": "all rows",
            "grading": "ambiguous rows excluded from headline, reported separately",
            "primary_tokens": "canonical leading-space only; *_agg is evaluation-only",
        },
        "token_ids": {
            "canonical": {"yes": tok["yes"], "no": tok["no"],
                          "grades": {str(g): tok["grade"][g] for g in GRADES}},
            "aggregated_eval_only": {
                "yes": tok["yes_agg"], "no": tok["no_agg"],
                "grades": {str(g): tok["grade_agg"][g] for g in GRADES}},
        },
        "prompts": {
            "eligibility": build_body(
                {"criterion": "Inclusion: <inclusion_rule>.", "patient": "<patient_detail>"}),
            "grading": f"{GRADE_INSTRUCTION}\nPatient: <patient_detail>\n{GRADE_QUESTION}",
        },
        "warnings": warnings,
        "summary": {
            "eligibility_correct": elig_ok, "eligibility_n": elig_n,
            "grade_correct": grade_ok, "grade_n": grade_n,
            "grade_ambiguous_correct": amb_ok, "grade_ambiguous_n": amb_n,
            "self_consistent": self_ok,
            "by_expected_grade": by_grade,
            "by_lexical_distance": by_lex,
            "by_decisiveness": by_dec,
            "paraphrase_sets": sets,
            "dissociation": diss,
        },
        "rows": rows,
    }
    out = Path(args.out) if args.out else Path(OUT_TEMPLATE.format(slug=model_slug))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    if args.write_csv:
        # Fill the blank model_inclusion / model_ecog columns the stimulus file
        # reserves for results, leaving every hand-written column untouched.
        with stim_path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            src = list(reader)
        by_id = {r["id"]: r for r in rows}
        key = next((f for f in fields if norm_key(f) == "id"), None)
        for raw in src:
            r = by_id.get((raw.get(key) or "").strip()) if key else None
            if r and "eligibility" in r:
                raw["model_inclusion"] = r["eligibility"]["says"]
                raw["model_ecog"] = str(r["grading"]["pred_grade"])
        csv_out = (Path(f"data/ecog_v0_scored_{model_slug}.csv")
                   if args.write_csv == "AUTO" else Path(args.write_csv))
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(src)
        print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
