"""
prompts/medical_knowledge.py

Medical-factual cloze prompts — the medical equivalent of "The capital of
France is" → " Paris".  Base model (`medgemma-4b-pt`) should predict the target
token confidently since these are factual completions heavily represented in
pretraining (textbook knowledge, drug references, clinical guidelines).

Used for Track B (classic CLT tracing): validates the MedGemma CLT pipeline
produces interpretable medical-knowledge features.  No scaffold engineering
needed — the base model genuinely commits to factual completions.

Target tokens are space-prefixed to match sentencepiece conventions.
All targets verified to be single-token under the Gemma tokenizer.
"""

from __future__ import annotations

from typing import TypedDict


class TrialPrompt(TypedDict):
    id: str
    text: str
    target_token: str
    domain_tags: list[str]


MEDICAL_KNOWLEDGE_PROMPTS: list[TrialPrompt] = [

    # --- Pharmacology: drug → mechanism / target ---

    {
        "id": "pharm_imatinib",
        "text": "The standard first-line treatment for chronic myeloid leukemia is",
        "target_token": " imatinib",
        "domain_tags": ["pharmacology", "CML", "TKI"],
    },
    {
        "id": "pharm_trastuzumab",
        "text": "HER2 amplification in breast cancer is targeted by the monoclonal antibody",
        "target_token": " trastuzumab",
        "domain_tags": ["pharmacology", "HER2", "mAb", "oncology"],
    },
    {
        "id": "pharm_protamine",
        "text": "The antidote for heparin overdose is",
        "target_token": " protamine",
        "domain_tags": ["pharmacology", "anticoagulation", "antidote"],
    },
    {
        "id": "pharm_warfarin_moa",
        "text": "Warfarin acts by inhibiting vitamin",
        "target_token": " K",
        "domain_tags": ["pharmacology", "anticoagulation", "mechanism"],
    },
    {
        "id": "pharm_metformin",
        "text": "The first-line oral agent for type 2 diabetes is",
        "target_token": " metformin",
        "domain_tags": ["pharmacology", "diabetes", "endocrine"],
    },

    # --- Pathophysiology: disease → mechanism / consequence ---

    {
        "id": "path_insulin_organ",
        "text": "Insulin is produced by the beta cells of the",
        "target_token": " pancreas",
        "domain_tags": ["pathophysiology", "endocrine", "anatomy"],
    },
    {
        "id": "path_mi_artery",
        "text": "The left anterior descending artery supplies the anterior wall of the",
        "target_token": " heart",
        "domain_tags": ["pathophysiology", "cardiology", "anatomy"],
    },
    {
        "id": "path_dvt_complication",
        "text": "The most dangerous complication of a deep vein thrombosis is pulmonary",
        "target_token": " embolism",
        "domain_tags": ["pathophysiology", "hematology", "VTE"],
    },

    # --- Diagnostics: finding → diagnosis ---

    {
        "id": "dx_reed_sternberg",
        "text": "Reed-Sternberg cells are the hallmark of",
        "target_token": " Hodgkin",
        "domain_tags": ["diagnostics", "hematology", "pathology"],
    },
    {
        "id": "dx_auer_rods",
        "text": "Auer rods on peripheral smear are pathognomonic for acute",
        "target_token": " myeloid",
        "domain_tags": ["diagnostics", "hematology", "AML"],
    },
    {
        "id": "dx_troponin",
        "text": "The most sensitive biomarker for myocardial infarction is",
        "target_token": " troponin",
        "domain_tags": ["diagnostics", "cardiology", "biomarker"],
    },

    # --- Clinical guidelines: standard of care ---

    {
        "id": "guide_colonoscopy_age",
        "text": "Current guidelines recommend initiating colorectal cancer screening at age",
        "target_token": " 45",
        "domain_tags": ["guidelines", "screening", "GI", "oncology"],
    },
    {
        "id": "guide_hypertension_threshold",
        "text": "Stage 2 hypertension is defined as systolic blood pressure at or above",
        "target_token": " 140",
        "domain_tags": ["guidelines", "cardiology", "hypertension"],
    },

    # --- Anatomy / physiology ---

    {
        "id": "anat_csf_production",
        "text": "Cerebrospinal fluid is produced primarily by the choroid",
        "target_token": " plexus",
        "domain_tags": ["anatomy", "neurology", "physiology"],
    },
    {
        "id": "anat_bile_storage",
        "text": "Bile is stored and concentrated in the",
        "target_token": " gallbladder",
        "domain_tags": ["anatomy", "GI", "hepatobiliary"],
    },
]
