from prompts.eligibility import (
    ELIGIBILITY_PAIRS,
    ELIGIBILITY_PROMPTS,
    ContrastivePrompt,
    TrialPrompt,
    to_chat,
    to_gemma_chat,
)
from prompts.adverse_events import ADVERSE_EVENT_PROMPTS
from prompts.endpoints import ENDPOINT_PROMPTS

ALL_PROMPTS: list[TrialPrompt] = (
    ELIGIBILITY_PROMPTS + ADVERSE_EVENT_PROMPTS + ENDPOINT_PROMPTS
)

__all__ = [
    "TrialPrompt",
    "ContrastivePrompt",
    "ELIGIBILITY_PROMPTS",
    "ELIGIBILITY_PAIRS",
    "to_chat",
    "to_gemma_chat",
    "ADVERSE_EVENT_PROMPTS",
    "ENDPOINT_PROMPTS",
    "ALL_PROMPTS",
]
