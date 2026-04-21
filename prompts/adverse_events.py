"""
prompts/adverse_events.py

Clinical trial adverse event attribution prompts — hematologic toxicity,
hepatotoxicity, cardiotoxicity, treatment discontinuation.
"""

from __future__ import annotations

from typing import TypedDict


class TrialPrompt(TypedDict):
    id: str
    prompt: str
    target_token: str
    domain_tags: list[str]


ADVERSE_EVENT_PROMPTS: list[TrialPrompt] = [
    {
        "id": "dose_reduction_neutropenia",
        "prompt": (
            "Day 14 labs: ANC 0.8 × 10⁹/L (Grade 3 neutropenia). "
            "Per protocol, Grade 3 hematologic toxicity requires dose reduction. "
            "Recommended action: dose"
        ),
        "target_token": " reduction",
        "domain_tags": ["adverse_events", "hematologic"],
    },
    {
        "id": "hepatotoxicity_toxic",
        "prompt": (
            "ALT 8× ULN, total bilirubin 2× ULN on day 21 assessment. "
            "Drug-induced liver injury grading: this pattern is"
        ),
        "target_token": " toxic",
        "domain_tags": ["adverse_events", "hepatotoxicity"],
    },
    {
        "id": "treatment_discontinued",
        "prompt": (
            "Patient developed Grade 4 pneumonitis after cycle 3. "
            "Per CTCAE guidelines, Grade 4 immune-related pneumonitis requires "
            "permanent treatment"
        ),
        "target_token": " discontinuation",
        "domain_tags": ["adverse_events"],
    },
    {
        "id": "ae_grade_severe",
        "prompt": (
            "LVEF declined from 65% to 42% after 4 cycles of anthracycline-based therapy. "
            "Cardiotoxicity classification: this finding is"
        ),
        "target_token": " severe",
        "domain_tags": ["adverse_events", "cardiotoxicity"],
    },
]
