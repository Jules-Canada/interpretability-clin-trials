"""
prompts/eligibility.py

Clinical trial eligibility prompts — NSCLC, performance status, prior therapy.
"""

from __future__ import annotations

from typing import TypedDict


class TrialPrompt(TypedDict):
    id: str
    prompt: str
    target_token: str
    domain_tags: list[str]


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
