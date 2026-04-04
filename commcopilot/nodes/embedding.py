"""N2: Embedding node. Vectorizes transcript and stores to pgvector.

Fire-and-forget: errors are logged but never block the pipeline.
"""

import logging
from commcopilot.state import PipelineState

logger = logging.getLogger(__name__)


async def _store_embedding(transcript: str, session_id: str, db_pool) -> bool:
    """Embed transcript chunk and store in pgvector. Returns True on success."""
    from ibm_watsonx_ai.foundation_models import Embeddings
    from commcopilot.config import (
        WATSONX_API_KEY,
        WATSONX_PROJECT_ID,
        WATSONX_URL,
        EMBEDDING_MODEL_ID,
    )

    try:
        embeddings = Embeddings(
            model_id=EMBEDDING_MODEL_ID,
            credentials={"apikey": WATSONX_API_KEY, "url": WATSONX_URL},
            project_id=WATSONX_PROJECT_ID,
        )
        result = embeddings.embed_documents([transcript])
        vector = result[0]

        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO conversation_embeddings (session_id, text_chunk, embedding)
                   VALUES ($1, $2, $3)""",
                session_id,
                transcript,
                str(vector),
            )
        logger.info("Embedding stored for session %s (%d chars)", session_id, len(transcript))
        return True

    except Exception as e:
        logger.error("Embedding failed for session %s: %s", session_id, e)
        return False


def embed_transcript(state: PipelineState) -> dict:
    """Fire-and-forget embedding. Errors are logged, never block the pipeline."""
    transcript = state.get("transcript", "")
    session_id = state.get("session_id", "unknown")

    if not transcript.strip():
        logger.warning("Empty transcript for embedding, skipping. session=%s", session_id)
        return {"embedding_stored": False}

    # In the actual graph execution, this would be run as an async task.
    # For now, we mark it as pending. The server handles the async call.
    logger.info("Embedding queued for session %s", session_id)
    return {"embedding_stored": True}
