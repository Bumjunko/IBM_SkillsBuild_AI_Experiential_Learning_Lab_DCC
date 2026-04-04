"""N5: Retrieval node. Vector similarity search on past conversation embeddings.

Only runs when N3 context confidence < CONFIDENCE_THRESHOLD.
Returns empty list gracefully on any failure.
"""

import logging
from commcopilot.state import PipelineState

logger = logging.getLogger(__name__)


async def _search_similar(query_text: str, session_id: str, db_pool, limit: int = 3) -> list[str]:
    """Search pgvector for similar past conversation chunks."""
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
        query_vector = embeddings.embed_documents([query_text])[0]

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT text_chunk, 1 - (embedding <=> $1::vector) AS similarity
                   FROM conversation_embeddings
                   WHERE session_id != $2
                   ORDER BY embedding <=> $1::vector
                   LIMIT $3""",
                str(query_vector),
                session_id,
                limit,
            )
        return [row["text_chunk"] for row in rows]

    except Exception as e:
        logger.error("Retrieval search failed: %s", e)
        return []


def retrieve_history(state: PipelineState) -> dict:
    """Retrieve relevant past conversation history via vector search.

    Returns empty list on failure (graceful degradation).
    """
    transcript = state.get("transcript", "")

    if not transcript.strip():
        logger.info("Empty transcript for retrieval, returning empty history")
        return {"relevant_history": []}

    # Actual async search is handled by the server when invoking the graph.
    # This synchronous version returns empty for now.
    logger.info("Retrieval requested for transcript: %s", transcript[:60])
    return {"relevant_history": []}
