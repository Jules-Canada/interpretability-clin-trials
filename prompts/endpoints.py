"""
prompts/endpoints.py

Clinical trial endpoint inference prompts — PFS, OS, ORR definitions.
"""

from __future__ import annotations

from typing import TypedDict


class TrialPrompt(TypedDict):
    id: str
    prompt: str
    target_token: str
    domain_tags: list[str]


ENDPOINT_PROMPTS: list[TrialPrompt] = [
    {
        "id": "partial_response",
        "prompt": (
            "CT scan at week 8: target lesion sum decreased from 42mm to 28mm (33% reduction). "
            "RECIST 1.1 assessment: this response is classified as"
        ),
        "target_token": " partial",
        "domain_tags": ["endpoints", "response"],
    },
    {
        "id": "progressive_disease",
        "prompt": (
            "Week 12 imaging: new hepatic metastases identified, target lesions increased 25%. "
            "RECIST 1.1 classification: this represents"
        ),
        "target_token": " progressive",
        "domain_tags": ["endpoints", "progression"],
    },
    {
        "id": "endpoint_survival",
        "prompt": (
            "Primary endpoint: overall survival defined as time from randomization to death "
            "from any cause. Median OS in the treatment arm was 18.2 months. "
            "The survival benefit was"
        ),
        "target_token": " significant",
        "domain_tags": ["endpoints", "OS"],
    },
    {
        "id": "trial_met_endpoint",
        "prompt": (
            "Progression-free survival: median PFS 11.3 months (treatment) vs 5.7 months "
            "(control), HR 0.48, p<0.001. The trial"
        ),
        "target_token": " met",
        "domain_tags": ["endpoints", "PFS"],
    },
]
