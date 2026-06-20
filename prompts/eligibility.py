"""
prompts/eligibility.py

Clinical trial eligibility prompts.

Two sets live here:

1. ELIGIBILITY_PROMPTS — LEGACY single-token completion prompts from the PT /
   MedGemma path. Kept only for backward compatibility (imported into
   prompts.ALL_PROMPTS). Do NOT use for the active path: they telegraph the
   answer and use inconsistent targets (eligible / excluded / yes).

2. The ACTIVE-PATH contrastive sets (Gemma-3-4B-IT + circuit-tracer). Each item
   shares a skeleton and flips ONE deciding fact across the eligibility
   threshold, so the pair difference isolates the decision circuit and scaffold
   cancels. Target is the Yes-vs-No contrast: logit(" Yes") - logit(" No").

   Render with to_chat() / to_gemma_chat() so the IT chat template is applied
   consistently. Rule (CLAUDE.md): IT model -> IT transcoders -> chat template.
   Feeding raw text to IT transcoders reintroduces the Phase 6 OOD failure.

   Convention: "Yes" == eligible across every pair (for exclusion criteria, the
   eligible/Yes member is the one WITHOUT the excluded feature, OR the one the
   stated exception clears). Keeping Yes=eligible uniform makes the contrastive
   readout comparable across pairs.

   --- Redesign 2026-06-19: corpus-grounded, structure-spanning ----------------
   The Round-1 set (5 inclusion/exclusion pairs) was confounded: all 3 "clean"
   pairs were INCLUSION criteria and both "failing" pairs were EXCLUSION, so the
   over-exclusion finding could not separate "knowledge gap" from
   "exclusion-phrasing -> No bias". It also tested only the simplest 1-clause
   form. A profile of 2,545 real ClinicalTrials.gov eligibility sections (see
   scripts/profile_criteria.py) shows our prompts were off-distribution on every
   structural axis real criteria use: numeric+unit thresholds (76% of real
   protocols), temporal qualifiers (45%), EXCEPTIONS / carve-outs (44%),
   conjunction/aggregation (38%), named instruments (21%), relative thresholds
   x ULN (17%). The carve-out operation — "apply a rule/knowledge to CLEAR a
   patient against a general exclusion" — is the naturalistic twin of the
   over-exclusion finding AND appears in ~44% of real protocols.

   Three sets result:
     * ELIGIBILITY_PAIRS           — CORE graph tier. A complete 2x2 factorial
       {inclusion,exclusion} x {surface,knowledge} (kills the Round-1 confound)
       + two exception/carve-out pairs (the headline tier). All short (<~150
       tokens) so attribution stays tractable and readable.
     * ELIGIBILITY_PAIRS_EXTENDED  — extra graph-tier controls (ordinal, relative
       threshold, temporal window). Graph if budget allows.
     * ELIGIBILITY_BEHAVIORAL      — forward-pass ONLY (too long / not cleanly
       isolable to attribute): conjunction-aggregation over many sub-conditions,
       and full realistic multi-criterion blocks. Length is a non-issue for a
       forward pass; we deliberately do NOT graph these (see CLAUDE.md / the
       length-vs-attribution discussion: graph short pairs, sweep everything).

   Each contrastive item now carries four analysis axes so the sweep/compare
   scripts can group by cell:
     tier      "graph" | "behavioral"
     structure "fragment" | "relative_threshold" | "temporal" | "exception"
               | "conjunction"
     phrasing  "inclusion" | "exclusion"
     inference "surface" | "knowledge"
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
    tier: str               # "graph" | "behavioral"
    structure: str          # fragment|relative_threshold|temporal|exception|conjunction
    phrasing: str           # "inclusion" | "exclusion"
    inference: str          # "surface" | "knowledge"
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


# ============================================================================ #
# ACTIVE PATH — CORE graph tier
# ============================================================================ #
# A complete 2x2 factorial crossing criterion PHRASING (inclusion vs exclusion)
# with required INFERENCE (surface string-match vs world knowledge), each with a
# Yes (eligible) and No (ineligible) member, PLUS two exception/carve-out pairs.
#
# Why the factorial: Round 1 only had inclusion-surface and exclusion-knowledge
# cells, so "fails on knowledge" and "fails on exclusion phrasing" were perfectly
# entangled. The two new cells break that:
#   - exclusion x SURFACE  (prior_chemo): if the model still over-excludes the
#     treatment-naive arm here — no world knowledge required — the bug is
#     exclusion-PHRASING bias, not a knowledge gap.
#   - inclusion x KNOWLEDGE (her2): if knowledge failures appear under inclusion
#     phrasing too, the failure is about knowledge, not exclusion.
#
# The exception pairs are the headline tier (carve-outs are ~44% of real
# protocols). They flip ONLY the clause that should CLEAR the patient — the exact
# "apply a rule to clear" operation the over-exclusion finding is about:
#   - gilbert    : STATED exception (no world knowledge — model just has to apply
#                  the carve-out written in the criterion).
#   - malignancy : KNOWLEDGE exception (model must know basal cell carcinoma is a
#                  non-melanoma skin cancer to apply the stated carve-out).
# Contrast of those two isolates "fails to apply a stated exception" from "lacks
# the knowledge to match the exception".

ELIGIBILITY_PAIRS: list[ContrastivePrompt] = [
    # --- cell: inclusion x surface (numeric range) ------------------------- #
    {
        "id": "elig_age_pos",
        "pair_id": "age",
        "criterion": "Inclusion: age 18 to 75 years.",
        "patient": "73-year-old with stage IIIA non-small cell lung cancer.",
        "expected": "Yes",
        "requires": "numeric range comparison (73 is within 18-75)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "age", "control"],
    },
    {
        "id": "elig_age_neg",
        "pair_id": "age",
        "criterion": "Inclusion: age 18 to 75 years.",
        "patient": "78-year-old with stage IIIA non-small cell lung cancer.",
        "expected": "No",
        "requires": "numeric range comparison (78 exceeds 75)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "age", "control"],
    },
    # --- cell: inclusion x surface (lexical category match) ---------------- #
    # Pure surface: criterion names a histology; patient contains that token or a
    # different one. No number, no world knowledge — the lexical-match baseline.
    {
        "id": "elig_histology_pos",
        "pair_id": "histology",
        "criterion": "Inclusion: histologically confirmed adenocarcinoma.",
        "patient": "61-year-old with lung adenocarcinoma.",
        "expected": "Yes",
        "requires": "lexical category match (named histology matches the criterion)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "histology", "control"],
    },
    {
        "id": "elig_histology_neg",
        "pair_id": "histology",
        "criterion": "Inclusion: histologically confirmed adenocarcinoma.",
        "patient": "61-year-old with squamous cell lung carcinoma.",
        "expected": "No",
        "requires": "lexical category mismatch (squamous cell is not adenocarcinoma)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "histology", "control"],
    },
    # --- cell: inclusion x KNOWLEDGE (new — confound-breaker) -------------- #
    # The criterion names a biomarker by its clinical name; the patient is
    # described by an equivalent that requires knowledge to match:
    #   pos: ERBB2 amplification  == HER2-positive   (gene name -> protein status)
    #   neg: triple-negative      => HER2-negative    (definition of TNBC)
    # No verbatim "HER2" string in either patient line, so a lexical match is
    # impossible; the decision needs a concept->status retrieval under INCLUSION
    # phrasing (the mirror of the exclusion-knowledge pairs below).
    {
        "id": "elig_her2_pos",
        "pair_id": "her2",
        "criterion": "Inclusion: HER2-positive breast cancer.",
        "patient": "58-year-old with breast cancer showing ERBB2 gene amplification.",
        "expected": "Yes",
        "requires": "knowledge (ERBB2 amplification defines HER2-positive disease)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "biomarker", "knowledge"],
    },
    {
        "id": "elig_her2_neg",
        "pair_id": "her2",
        "criterion": "Inclusion: HER2-positive breast cancer.",
        "patient": "58-year-old with triple-negative breast cancer.",
        "expected": "No",
        "requires": "knowledge (triple-negative breast cancer is HER2-negative)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "biomarker", "knowledge"],
    },
    # --- cell: exclusion x SURFACE (new — the key confound-breaker) -------- #
    # Same exclusion axis as prior_tx, but NO world knowledge: the patient line
    # states presence/absence of the excluded thing in plain words. If the model
    # over-excludes the treatment-naive (Yes) arm here, the over-exclusion is
    # exclusion-PHRASING bias, independent of any knowledge gap.
    {
        "id": "elig_priorchemo_pos",
        "pair_id": "prior_chemo",
        "criterion": "Exclusion: prior chemotherapy.",
        "patient": "60-year-old with NSCLC who is treatment-naive and has received no prior chemotherapy.",
        "expected": "Yes",
        "requires": "surface negation (patient explicitly has had no prior chemotherapy)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "exclusion", "inference": "surface",
        "domain_tags": ["eligibility", "prior_therapy", "control"],
    },
    {
        "id": "elig_priorchemo_neg",
        "pair_id": "prior_chemo",
        "criterion": "Exclusion: prior chemotherapy.",
        "patient": "60-year-old with NSCLC who previously received chemotherapy.",
        "expected": "No",
        "requires": "surface match (patient explicitly had prior chemotherapy)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "exclusion", "inference": "surface",
        "domain_tags": ["eligibility", "prior_therapy", "control"],
    },
    # --- cell: exclusion x KNOWLEDGE (drug -> class) ----------------------- #
    # Both members name exactly one real drug. In the Gemma-3 tokenizer they share
    # only the first subword 'p' and diverge at subword 2 (pemetrexed=p|emet|rex|ed,
    # pembrolizumab=p|emb|rol|izumab), so the deciding signal cannot be the first
    # token — it must come from post-token-1 discrimination + drug->class retrieval.
    {
        "id": "elig_priortx_pos",
        "pair_id": "prior_tx",
        "criterion": "Exclusion: prior treatment with a PD-1 or PD-L1 inhibitor.",
        "patient": "60-year-old with NSCLC who previously received pemetrexed.",
        "expected": "Yes",
        "requires": "drug-to-class knowledge (pemetrexed is chemotherapy, not a checkpoint inhibitor)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "exclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "prior_therapy", "knowledge"],
    },
    {
        "id": "elig_priortx_neg",
        "pair_id": "prior_tx",
        "criterion": "Exclusion: prior treatment with a PD-1 or PD-L1 inhibitor.",
        "patient": "60-year-old with NSCLC who previously received pembrolizumab.",
        "expected": "No",
        "requires": "drug-to-class knowledge (pembrolizumab is an anti-PD-1 inhibitor)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "exclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "prior_therapy", "knowledge"],
    },
    # --- cell: exclusion x KNOWLEDGE (staging -> metastatic) --------------- #
    # Criterion omits the "(stage IV)" gloss so the model cannot string-match the
    # stage; eligibility hinges on knowing III is M0 and IV is M1. Roman-numeral flip.
    {
        "id": "elig_stage_pos",
        "pair_id": "stage",
        "criterion": "Exclusion: metastatic disease.",
        "patient": "57-year-old with stage III non-small cell lung cancer.",
        "expected": "Yes",
        "requires": "staging knowledge (stage III is not metastatic)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "exclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "staging", "knowledge"],
    },
    {
        "id": "elig_stage_neg",
        "pair_id": "stage",
        "criterion": "Exclusion: metastatic disease.",
        "patient": "57-year-old with stage IV non-small cell lung cancer.",
        "expected": "No",
        "requires": "staging knowledge (stage IV is metastatic)",
        "tier": "graph", "structure": "fragment",
        "phrasing": "exclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "staging", "knowledge"],
    },
    # --- HEADLINE: exception / carve-out, STATED (no world knowledge) ------ #
    # Anchored on the real Gilbert-syndrome bilirubin carve-out (corpus). The
    # exception is written in the criterion; the model only has to APPLY it to
    # clear an otherwise-excluded value. The pair flips ONLY the Gilbert clause in
    # the patient line, everything else (the 2.0 x ULN value) held fixed. This is
    # the cleanest over-exclusion test: failure here is failure to apply a
    # carve-out the prompt already gave it.
    {
        "id": "elig_gilbert_pos",
        "pair_id": "gilbert",
        "criterion": "Exclusion: total bilirubin greater than 1.5 times the upper limit of normal, unless the elevation is attributable to Gilbert syndrome.",
        "patient": "60-year-old with NSCLC; total bilirubin 2.0 times the upper limit of normal, attributable to Gilbert syndrome.",
        "expected": "Yes",
        "requires": "apply a STATED exception to clear an otherwise-excluded value",
        "tier": "graph", "structure": "exception",
        "phrasing": "exclusion", "inference": "surface",
        "domain_tags": ["eligibility", "organ_function", "exception"],
    },
    {
        "id": "elig_gilbert_neg",
        "pair_id": "gilbert",
        "criterion": "Exclusion: total bilirubin greater than 1.5 times the upper limit of normal, unless the elevation is attributable to Gilbert syndrome.",
        "patient": "60-year-old with NSCLC; total bilirubin 2.0 times the upper limit of normal.",
        "expected": "No",
        "requires": "exclusion applies (elevated bilirubin, exception does not hold)",
        "tier": "graph", "structure": "exception",
        "phrasing": "exclusion", "inference": "surface",
        "domain_tags": ["eligibility", "organ_function", "exception"],
    },
    # --- HEADLINE: exception / carve-out, KNOWLEDGE ------------------------ #
    # Real corpus carve-out: "history of malignancy ... except non-melanoma skin
    # cancer within the past 5 years". To CLEAR the pos arm the model must know
    # basal cell carcinoma IS a non-melanoma skin cancer (knowledge) AND apply the
    # stated exception. Contrast with gilbert (stated, no knowledge) isolates
    # "can't apply a stated exception" from "lacks the knowledge to match it".
    {
        "id": "elig_malignancy_pos",
        "pair_id": "malignancy",
        "criterion": "Exclusion: history of malignancy within the past 5 years, except adequately treated non-melanoma skin cancer.",
        "patient": "62-year-old with NSCLC and a history of basal cell carcinoma treated 2 years ago.",
        "expected": "Yes",
        "requires": "knowledge (basal cell carcinoma is a non-melanoma skin cancer) + apply exception",
        "tier": "graph", "structure": "exception",
        "phrasing": "exclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "malignancy_history", "exception"],
    },
    {
        "id": "elig_malignancy_neg",
        "pair_id": "malignancy",
        "criterion": "Exclusion: history of malignancy within the past 5 years, except adequately treated non-melanoma skin cancer.",
        "patient": "62-year-old with NSCLC and a history of colon cancer treated 2 years ago.",
        "expected": "No",
        "requires": "knowledge (colon cancer is not exempted) + exclusion applies",
        "tier": "graph", "structure": "exception",
        "phrasing": "exclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "malignancy_history", "exception"],
    },
]


# ============================================================================ #
# ACTIVE PATH — EXTENDED graph tier (extra controls; graph if budget allows)
# ============================================================================ #
# More structural primitives from the real distribution. Short enough to graph,
# but secondary to the core factorial — run after ELIGIBILITY_PAIRS.

ELIGIBILITY_PAIRS_EXTENDED: list[ContrastivePrompt] = [
    # ordinal set membership (the Round-1 ECOG control) ------------------- #
    {
        "id": "elig_ecog_pos",
        "pair_id": "ecog",
        "criterion": "Inclusion: ECOG performance status 0 or 1.",
        "patient": "62-year-old with colorectal cancer, ECOG performance status 1.",
        "expected": "Yes",
        "requires": "ordinal set membership (1 is in {0,1})",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "performance_status", "control"],
    },
    {
        "id": "elig_ecog_neg",
        "pair_id": "ecog",
        "criterion": "Inclusion: ECOG performance status 0 or 1.",
        "patient": "62-year-old with colorectal cancer, ECOG performance status 2.",
        "expected": "No",
        "requires": "ordinal set membership (2 is not in {0,1})",
        "tier": "graph", "structure": "fragment",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "performance_status", "control"],
    },
    # relative threshold (x ULN — 17% of real protocols) ----------------- #
    # Anchored on the real "Serum creatinine <= 1.5 x ULN" criterion. Tests the
    # relative-comparison primitive (value expressed as a multiple of a reference
    # not given numerically in the prompt).
    {
        "id": "elig_creatinine_pos",
        "pair_id": "creatinine",
        "criterion": "Inclusion: serum creatinine no greater than 1.5 times the upper limit of normal.",
        "patient": "64-year-old with NSCLC; serum creatinine 1.1 times the upper limit of normal.",
        "expected": "Yes",
        "requires": "relative-threshold comparison (1.1x <= 1.5x)",
        "tier": "graph", "structure": "relative_threshold",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "organ_function", "control"],
    },
    {
        "id": "elig_creatinine_neg",
        "pair_id": "creatinine",
        "criterion": "Inclusion: serum creatinine no greater than 1.5 times the upper limit of normal.",
        "patient": "64-year-old with NSCLC; serum creatinine 2.2 times the upper limit of normal.",
        "expected": "No",
        "requires": "relative-threshold comparison (2.2x > 1.5x)",
        "tier": "graph", "structure": "relative_threshold",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "organ_function", "control"],
    },
    # temporal window (45% of real protocols) ---------------------------- #
    # "major surgery within the prior 4 weeks" — eligibility hinges on whether the
    # patient's event falls inside or outside the stated window.
    {
        "id": "elig_temporal_pos",
        "pair_id": "temporal",
        "criterion": "Exclusion: major surgery within the prior 4 weeks.",
        "patient": "59-year-old with NSCLC who underwent major surgery 3 months ago.",
        "expected": "Yes",
        "requires": "temporal window comparison (3 months is outside 4 weeks)",
        "tier": "graph", "structure": "temporal",
        "phrasing": "exclusion", "inference": "surface",
        "domain_tags": ["eligibility", "washout", "control"],
    },
    {
        "id": "elig_temporal_neg",
        "pair_id": "temporal",
        "criterion": "Exclusion: major surgery within the prior 4 weeks.",
        "patient": "59-year-old with NSCLC who underwent major surgery 10 days ago.",
        "expected": "No",
        "requires": "temporal window comparison (10 days is inside 4 weeks)",
        "tier": "graph", "structure": "temporal",
        "phrasing": "exclusion", "inference": "surface",
        "domain_tags": ["eligibility", "washout", "control"],
    },
]


# ============================================================================ #
# ACTIVE PATH — BEHAVIORAL tier (forward-pass ONLY; do NOT graph)
# ============================================================================ #
# These are long / not cleanly isolable, so attribution would be an expensive
# hairball with diluted completeness (see the length-vs-attribution discussion).
# But a forward pass costs nothing extra, so they belong in the behavioral sweep:
# they test whether over-exclusion (and correct reading) survive REALISTIC,
# complex criteria, which is the ecological-validity check the graphs can't give.
#
# Convention preserved (Yes == eligible). The "conjunction" pairs flip ONE
# sub-condition inside an all-else-passing block, so the ground truth is clean
# even though the prompt is long.

ELIGIBILITY_BEHAVIORAL: list[ContrastivePrompt] = [
    # conjunction / aggregation over many sub-conditions (38% of protocols) #
    # Anchored on a real "adequate organ function defined as: ..." block. All
    # sub-conditions must hold; the neg arm flips exactly one (platelets) below
    # threshold. Tests AND-aggregation: does the model catch a single failing
    # value buried in an otherwise-passing list?
    {
        "id": "elig_organfx_pos",
        "pair_id": "organ_fx",
        "criterion": (
            "Inclusion: adequate organ function, defined as absolute neutrophil "
            "count at least 1.5 x 10^9/L, platelet count at least 100 x 10^9/L, "
            "hemoglobin at least 9.0 g/dL, serum creatinine no greater than 1.5 "
            "times the upper limit of normal, and AST and ALT no greater than 3.0 "
            "times the upper limit of normal."
        ),
        "patient": (
            "63-year-old with NSCLC; neutrophils 3.0 x 10^9/L, platelets "
            "180 x 10^9/L, hemoglobin 11.5 g/dL, creatinine 1.0 x ULN, AST and "
            "ALT 1.5 x ULN."
        ),
        "expected": "Yes",
        "requires": "conjunction over 5 sub-conditions (all pass)",
        "tier": "behavioral", "structure": "conjunction",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "organ_function", "aggregation"],
    },
    {
        "id": "elig_organfx_neg",
        "pair_id": "organ_fx",
        "criterion": (
            "Inclusion: adequate organ function, defined as absolute neutrophil "
            "count at least 1.5 x 10^9/L, platelet count at least 100 x 10^9/L, "
            "hemoglobin at least 9.0 g/dL, serum creatinine no greater than 1.5 "
            "times the upper limit of normal, and AST and ALT no greater than 3.0 "
            "times the upper limit of normal."
        ),
        "patient": (
            "63-year-old with NSCLC; neutrophils 3.0 x 10^9/L, platelets "
            "70 x 10^9/L, hemoglobin 11.5 g/dL, creatinine 1.0 x ULN, AST and "
            "ALT 1.5 x ULN."
        ),
        "expected": "No",
        "requires": "conjunction over 5 sub-conditions (one fails: platelets 70 < 100)",
        "tier": "behavioral", "structure": "conjunction",
        "phrasing": "inclusion", "inference": "surface",
        "domain_tags": ["eligibility", "organ_function", "aggregation"],
    },
    # threshold + nested carve-out, behavioral (the liver-mets real example) #
    # Real corpus: AST/ALT <= 3 x ULN, but <= 5 x ULN if liver metastases are
    # present. The pos arm is cleared ONLY by the nested carve-out (4x ULN would
    # fail the base bound but passes the liver-mets bound). Longer + two-step, so
    # behavioral rather than graph.
    {
        "id": "elig_livermets_pos",
        "pair_id": "liver_mets",
        "criterion": (
            "Inclusion: AST and ALT no greater than 3.0 times the upper limit of "
            "normal, or no greater than 5.0 times the upper limit of normal in the "
            "presence of liver metastases."
        ),
        "patient": "61-year-old with NSCLC and liver metastases; AST and ALT 4.0 times the upper limit of normal.",
        "expected": "Yes",
        "requires": "apply the nested liver-metastases carve-out (4x passes the 5x bound)",
        "tier": "behavioral", "structure": "exception",
        "phrasing": "inclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "organ_function", "exception"],
    },
    {
        "id": "elig_livermets_neg",
        "pair_id": "liver_mets",
        "criterion": (
            "Inclusion: AST and ALT no greater than 3.0 times the upper limit of "
            "normal, or no greater than 5.0 times the upper limit of normal in the "
            "presence of liver metastases."
        ),
        "patient": "61-year-old with NSCLC and no liver metastases; AST and ALT 4.0 times the upper limit of normal.",
        "expected": "No",
        "requires": "carve-out does not apply (4x exceeds the 3x base bound)",
        "tier": "behavioral", "structure": "exception",
        "phrasing": "inclusion", "inference": "knowledge",
        "domain_tags": ["eligibility", "organ_function", "exception"],
    },
]


# Everything the behavioral sweep should run (graphs use ELIGIBILITY_PAIRS only).
ELIGIBILITY_ALL: list[ContrastivePrompt] = (
    ELIGIBILITY_PAIRS + ELIGIBILITY_PAIRS_EXTENDED + ELIGIBILITY_BEHAVIORAL
)


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
