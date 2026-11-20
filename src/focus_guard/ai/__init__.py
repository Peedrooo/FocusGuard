"""AI subsystem: Gemini client, Semantic Judge, Coach Agent, and Voice Pipeline."""

from focus_guard.ai.coach_agent import FirmCoachAgent
from focus_guard.ai.gemini_client import (
    BudgetExceededError,
    GeminiClient,
    GeminiNotConfiguredError,
    extract_response_text,
    parse_json_object,
)
from focus_guard.ai.semantic_judge import GeminiSemanticJudge
from focus_guard.ai.voice_pipeline import EdgeAndOfflineTTS, SoundDeviceMicRecorder

__all__ = [
    "BudgetExceededError",
    "EdgeAndOfflineTTS",
    "FirmCoachAgent",
    "GeminiClient",
    "GeminiNotConfiguredError",
    "GeminiSemanticJudge",
    "SoundDeviceMicRecorder",
    "extract_response_text",
    "parse_json_object",
]
