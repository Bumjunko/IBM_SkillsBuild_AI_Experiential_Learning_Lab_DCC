"""N7: Safety Filter node. Filters unsafe phrases, re-generates if needed."""

import json
import logging
from commcopilot.state import PipelineState
from commcopilot.config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    AGENT_MODEL_ID,
    FALLBACK_PHRASES,
    MAX_SAFETY_RETRIES,
    MIN_SAFE_PHRASES,
)
from commcopilot.prompts import safety_filter_prompt

logger = logging.getLogger(__name__)


def _call_safety_filter(phrases: list[str]) -> list[str]:
    """Call Granite to filter phrases. Returns list of safe phrases."""
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai import Credentials

    credentials = Credentials(api_key=WATSONX_API_KEY, url=WATSONX_URL)
    model = ModelInference(
        model_id=AGENT_MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params={"max_new_tokens": 300, "temperature": 0.1},
    )

    prompt = safety_filter_prompt(phrases)
    response = model.generate_text(prompt=prompt)
    parsed = json.loads(response.strip())
    safe = parsed.get("safe", [])

    if parsed.get("rejected"):
        for item in parsed["rejected"]:
            logger.warning("Phrase rejected: %s", item)

    return [str(p) for p in safe]


def filter_phrases(state: PipelineState) -> dict:
    """Filter phrases for safety. Re-generates once if too few pass."""
    phrases = state.get("phrases", [])

    if not phrases:
        logger.warning("No phrases to filter, using fallback")
        return {"safe_phrases": list(FALLBACK_PHRASES), "error": "no_phrases_to_filter"}

    try:
        safe = _call_safety_filter(phrases)

        if len(safe) < MIN_SAFE_PHRASES:
            logger.warning("Only %d safe phrases, retrying", len(safe))
            for _ in range(MAX_SAFETY_RETRIES):
                safe = _call_safety_filter(phrases)
                if len(safe) >= MIN_SAFE_PHRASES:
                    break

        if len(safe) < MIN_SAFE_PHRASES:
            logger.warning("Too few safe phrases after retries, using fallback")
            return {"safe_phrases": list(FALLBACK_PHRASES), "error": None}

        logger.info("Safety filter passed %d/%d phrases", len(safe), len(phrases))
        return {"safe_phrases": safe, "error": None}

    except json.JSONDecodeError as e:
        logger.error("Safety filter JSON parse error: %s. Passing all phrases.", e)
        return {"safe_phrases": phrases, "error": f"safety_json_error: {e}"}
    except Exception as e:
        logger.error("Safety filter failed: %s. Passing all phrases (fail-open).", e)
        return {"safe_phrases": phrases, "error": f"safety_error: {e}"}
