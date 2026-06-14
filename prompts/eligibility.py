"""
prompts/eligibility.py

Clinical trial eligibility prompts.

Two sets live here:

1. ELIGIBILITY_PROMPTS — LEGACY single-token completion prompts from the PT /
   MedGemma path. Kept only for backward compatibility (imported into
   prompts.ALL_PROMPTS). Do NOT use for the active path: they telegraph the
   answer and use inconsistent targets (eligible / excluded / yes).

2. ELIGIBILITY_PAIRS — matched contrastive pairs for the ACTIVE path
   (Gemma-3-4B-IT + circuit-tracer; chosen 2026-06-13). Each pair shares a
   skeleton and flips ONE deciding fact across the eligibility threshold, so the
   pair difference isolates the comparison/decision circuit and scaffold cancels.
   Target is the Yes-vs-No contrast: logit(" Yes") - logit(" No").

   Render with to_gemma_chat() so the IT chat template is applied consistently.
   Rule (CLAUDE.md): IT model -> IT transcoders -> chat template. Feeding raw
   text to IT transcoders reintroduces the Phase 6 OOD failure.

   Convention: "Yes" == eligible across every pair (for exclusion criteria, the
   eligible/Yes member is the one WITHOUT the excluded feature). Keeping Yes=eligible
   uniform makes the contrastive readout comparable across pairs.
"""

from __future__ import annotations

from typing import TypedDict


class TrialPrompt(TypedDict):
    id: str
    prompt: str
    target_token: str
    domain_tags: list[str]


class ContrastivePrompt(TypedDict):
    id: str                 # unique, e.g. "elig_age_pos"
    pair_id: str            # links the matched pair, e.g. "age"
    criterion: str          # the inclusion/exclusion line shown to the model
    patient: str            # the patient description (the only thing that flips)
    expected: str           # ground truth: "Yes" (eligible) or "No" (ineligible)
    requires: str           # the inference the decision actually demands
    domain_tags: list[str]


# Contrastive target tokens. Verified single clean tokens in the Gemma-3
# tokenizer (2026-06-13, scripts/stage0_tokenizer_check.py); chat template
# verified byte-identical to apply_chat_template the same day.
#   " Yes" -> 8438 (▁Yes) | "Yes" -> 10784 | " No" -> 2301 (▁No) | "No" -> 3771
# The space-prefixed and bare variants are DIFFERENT tokens, and which one the
# model emits at the generation site (after "<start_of_turn>model\n") is only
# knowable from a forward pass. So the contrastive readout should average over
# BOTH variants per side rather than guess:
POS_TOKEN_IDS = [8438, 10784]   # " Yes", "Yes"
NEG_TOKEN_IDS = [2301, 3771]    # " No", "No"
POS_TOKEN = " Yes"              # convenience single-token (prefer the *_IDS sets)
NEG_TOKEN = " No"

INSTRUCTION = "You are screening a patient for a clinical trial."
QUESTION = "Is the patient eligible for the trial? Answer Yes or No."


def build_body(p: ContrastivePrompt) -> str:
    """The user-turn content, before chat-template wrapping."""
    return (
        f"{INSTRUCTION}\n"
        f"{p['criterion']}\n"
        f"Patient: {p['patient']}\n"
        f"{QUESTION}"
    )


def to_chat(tokenizer, p: ContrastivePrompt) -> str:
    """Canonical Gemma-3 IT chat string via the tokenizer's own template.

    Source of truth — prefer this whenever a tokenizer is available. The
    returned string already includes <bos>, so tokenize it with
    add_special_tokens=False to avoid a doubled BOS.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": build_body(p)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def to_gemma_chat(p: ContrastivePrompt) -> str:
    """Tokenizer-free Gemma-3 IT chat string. Verified byte-identical to
    tokenizer.apply_chat_template(add_generation_prompt=True) on
    google/gemma-3-4b-it (2026-06-13). Includes the leading <bos>, so tokenize
    with add_special_tokens=False. Prefer to_chat() when a tokenizer is loaded.
    """
    body = build_body(p)
    return f"<bos><start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"


# --- Active path: matched contrastive eligibility pairs ----------------------
# A ladder of decision primitives, surface -> conceptual. Each control isolates
# a DIFFERENT operation rather than re-measuring one:
#   Pair 1    — numeric magnitude / range comparison
#   Pair 2    — ordinal set membership
#   Pair 3    — lexical / categorical match
#   Pairs 4-5 — world-knowledge retrieval (pembrolizumab is anti-PD-1;
#               stage IV = metastatic), where no string-match with the criterion
#               is possible and a genuine cross-layer concept->decision circuit
#               would show up.
# The question the ladder asks: at which rung does the decision computation stop
# being surface and start being conceptual?

ELIGIBILITY_PAIRS: list[ContrastivePrompt] = [
    # Pair 1 — numeric age threshold
    {
        "id": "elig_age_pos",
        "pair_id": "age",
        "criterion": "Inclusion: age 18-75 years.",
        "patient": "73-year-old with stage IIIA non-small cell lung cancer.",
        "expected": "Yes",
        "requires": "numeric threshold comparison",
        "domain_tags": ["eligibility", "age", "control"],
    },
    {
        "id": "elig_age_neg",
        "pair_id": "age",
        "criterion": "Inclusion: age 18-75 years.",
        "patient": "78-year-old with stage IIIA non-small cell lung cancer.",
        "expected": "No",
        "requires": "numeric threshold comparison",
        "domain_tags": ["eligibility", "age", "control"],
    },
    # Pair 2 — ordinal performance status
    {
        "id": "elig_ecog_pos",
        "pair_id": "ecog",
        "criterion": "Inclusion: ECOG performance status 0 or 1.",
        "patient": "62-year-old with colorectal cancer, ECOG performance status 1.",
        "expected": "Yes",
        "requires": "ordinal performance-status comparison",
        "domain_tags": ["eligibility", "performance_status", "control"],
    },
    {
        "id": "elig_ecog_neg",
        "pair_id": "ecog",
        "criterion": "Inclusion: ECOG performance status 0 or 1.",
        "patient": "62-year-old with colorectal cancer, ECOG performance status 2.",
        "expected": "No",
        "requires": "ordinal performance-status comparison",
        "domain_tags": ["eligibility", "performance_status", "control"],
    },
    # Pair 3 — lexical / categorical match (no number, no world knowledge).
    # Pure surface: the criterion names a histology; the patient either contains
    # that token or names a different one. Isolates the lexical-match circuit as
    # the baseline the knowledge pairs (4-5) are compared against.
    {
        "id": "elig_histology_pos",
        "pair_id": "histology",
        "criterion": "Inclusion: histologically confirmed adenocarcinoma.",
        "patient": "61-year-old with lung adenocarcinoma.",
        "expected": "Yes",
        "requires": "lexical category match (named histology matches the criterion)",
        "domain_tags": ["eligibility", "histology", "control"],
    },
    {
        "id": "elig_histology_neg",
        "pair_id": "histology",
        "criterion": "Inclusion: histologically confirmed adenocarcinoma.",
        "patient": "61-year-old with squamous cell lung carcinoma.",
        "expected": "No",
        "requires": "lexical category match (squamous cell carcinoma is not adenocarcinoma)",
        "domain_tags": ["eligibility", "histology", "control"],
    },
    # Pair 4 — drug -> class knowledge (no verbatim match with the criterion).
    # Both members name exactly one real drug. In the Gemma-3 tokenizer they
    # share only the first subword 'p' and diverge at subword 2 (pemetrexed=
    # p|emet|rex|ed, pembrolizumab=p|emb|rol|izumab), so the deciding signal
    # cannot be the first token — it must come from post-token-1 discrimination
    # + drug->class retrieval. (Verified scripts/stage0_tokenizer_check.py.)
    {
        "id": "elig_priortx_pos",
        "pair_id": "prior_tx",
        "criterion": "Exclusion: prior treatment with a PD-1 or PD-L1 inhibitor.",
        "patient": "60-year-old with NSCLC who previously received pemetrexed.",
        "expected": "Yes",
        "requires": "drug-to-class knowledge (pemetrexed is chemotherapy, not a checkpoint inhibitor)",
        "domain_tags": ["eligibility", "prior_therapy", "knowledge"],
    },
    {
        "id": "elig_priortx_neg",
        "pair_id": "prior_tx",
        "criterion": "Exclusion: prior treatment with a PD-1 or PD-L1 inhibitor.",
        "patient": "60-year-old with NSCLC who previously received pembrolizumab.",
        "expected": "No",
        "requires": "drug-to-class knowledge (pembrolizumab is an anti-PD-1 inhibitor)",
        "domain_tags": ["eligibility", "prior_therapy", "knowledge"],
    },
    # Pair 5 — staging -> metastatic knowledge. Criterion deliberately omits the
    # "(stage IV)" gloss so the model cannot string-match the stage; eligibility
    # hinges on knowing III is M0 and IV is M1. Single roman-numeral flip.
    {
        "id": "elig_stage_pos",
        "pair_id": "stage",
        "criterion": "Exclusion: metastatic disease.",
        "patient": "57-year-old with stage III non-small cell lung cancer.",
        "expected": "Yes",
        "requires": "staging knowledge (stage III is not metastatic)",
        "domain_tags": ["eligibility", "staging", "knowledge"],
    },
    {
        "id": "elig_stage_neg",
        "pair_id": "stage",
        "criterion": "Exclusion: metastatic disease.",
        "patient": "57-year-old with stage IV non-small cell lung cancer.",
        "expected": "No",
        "requires": "staging knowledge (stage IV is metastatic)",
        "domain_tags": ["eligibility", "staging", "knowledge"],
    },
]


# --- Legacy: single-token PT-path prompts (do not use on the active path) -----
ELIGIBILITY_PROMPTS: list[TrialPrompt] = [
    {
        "id": "eligible_inclusion",
        "prompt": (
            "Patient: 58-year-old with stage IIIB non-small cell lung cancer, "
            "ECOG PS 1, no prior systemic therapy. "
            "Trial requires: NSCLC diagnosis, ECOG PS 0-1, no prior chemotherapy. "
            "Eligibility assessment:"
        ),
        "target_token": " eligible",
        "domain_tags": ["eligibility"],
    },
    {
        "id": "ineligible_ecog",
        "prompt": (
            "Patient: 72-year-old with metastatic breast cancer, ECOG PS 3, "
            "hepatic metastases. "
            "Trial requires: ECOG PS 0-1, adequate organ function. "
            "This patient is"
        ),
        "target_token": " excluded",
        "domain_tags": ["eligibility", "performance_status"],
    },
    {
        "id": "eligible_age",
        "prompt": (
            "Patient: 45-year-old female, renal cell carcinoma, clear cell histology, "
            "Karnofsky 90%, no prior targeted therapy. "
            "Trial eligibility criteria met:"
        ),
        "target_token": " yes",
        "domain_tags": ["eligibility"],
    },
    {
        "id": "ineligible_prior_therapy",
        "prompt": (
            "Patient received prior anti-PD1 therapy for 8 months before progression. "
            "The trial excludes patients with prior checkpoint inhibitor exposure. "
            "Enrollment decision:"
        ),
        "target_token": " excluded",
        "domain_tags": ["eligibility", "prior_therapy"],
    },
]
