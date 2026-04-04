"""LangGraph state definition for the CommCopilot pipeline."""

from typing import Optional
import operator
from typing_extensions import Annotated, TypedDict


class PipelineState(TypedDict):
    # Session info
    session_id: str
    scenario: str  # "office_hours" or "admin_office"

    # Input from STT / hesitation trigger
    transcript: str
    session_history: Annotated[list[str], operator.add]
    hesitation_trigger: str  # "pause" or "filler"

    # N2: Embedding (fire-and-forget)
    embedding_stored: bool

    # N3: Context Inference
    context_role: str
    context_tone: str
    context_formality: str
    context_intent: str
    context_confidence: float

    # N5: Retrieval
    relevant_history: list[str]

    # N6: Phrase Generation
    phrases: list[str]

    # N7: Safety Filter
    safe_phrases: list[str]

    # Error handling
    error: Optional[str]
