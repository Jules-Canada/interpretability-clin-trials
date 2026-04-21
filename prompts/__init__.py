from prompts.eligibility import ELIGIBILITY_PROMPTS, TrialPrompt
from prompts.adverse_events import ADVERSE_EVENT_PROMPTS
from prompts.endpoints import ENDPOINT_PROMPTS

ALL_PROMPTS: list[TrialPrompt] = (
    ELIGIBILITY_PROMPTS + ADVERSE_EVENT_PROMPTS + ENDPOINT_PROMPTS
)

__all__ = [
    "TrialPrompt",
    "ELIGIBILITY_PROMPTS",
    "ADVERSE_EVENT_PROMPTS",
    "ENDPOINT_PROMPTS",
    "ALL_PROMPTS",
]
