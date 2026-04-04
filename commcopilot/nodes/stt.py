"""N1: Speech-to-Text node. Updates transcript in pipeline state.

Note: Watson STT streaming runs separately via different file (watson_stt.py).
This node is called when a new transcript segment arrives to update the state.
"""

import logging
from commcopilot.state import PipelineState

logger = logging.getLogger(__name__)


def update_transcript(state: PipelineState) -> dict:
    """Update session history with the latest transcript segment."""
    transcript = state.get("transcript", "")
    if not transcript.strip():
        return {"session_history": []}

    logger.info("Transcript updated: %s", transcript[:80])
    return {"session_history": [transcript]}
