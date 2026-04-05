"""N6: Phrase Generation node. Generates 3 contextual phrase suggestions via Granite."""

import json
import logging
from commcopilot.state import PipelineState
from commcopilot.config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    AGENT_MODEL_ID,
    FALLBACK_PHRASES,
    SCENARIOS,
)
from commcopilot.prompts import phrase_generation_prompt

logger = logging.getLogger(__name__)


def generate_phrases(state: PipelineState) -> dict:
    """Generate 3 contextual phrase suggestions using Granite."""
    transcript = state.get("transcript", "")
    context_role = state.get("context_role", "")
    context_tone = state.get("context_tone", "")
    context_intent = state.get("context_intent", "")
    relevant_history = state.get("relevant_history", [])
    scenario_key = state.get("scenario", "")
    scenario = SCENARIOS.get(scenario_key)

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials

        credentials = Credentials(api_key=WATSONX_API_KEY, url=WATSONX_URL)
        model = ModelInference(
            model_id=AGENT_MODEL_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={"max_new_tokens": 200, "temperature": 0.7},
        )

        prompt = phrase_generation_prompt(
            transcript=transcript,
            context_role=context_role,
            context_tone=context_tone,
            context_intent=context_intent,
            relevant_history=relevant_history,
            scenario_context=scenario["system_context"],
        )
        response = model.generate_text(prompt=prompt)
        phrases = json.loads(response.strip())

        if not isinstance(phrases, list) or len(phrases) == 0:
            raise ValueError(f"Expected list of phrases, got: {type(phrases)}")

        phrases = [str(p) for p in phrases[:3]]
        logger.info("Generated %d phrases", len(phrases))
        return {"phrases": phrases, "error": None}

    except json.JSONDecodeError as e:
        logger.error("Phrase generation JSON parse error: %s", e)
        return {"phrases": list(FALLBACK_PHRASES), "error": f"generation_json_error: {e}"}
    except Exception as e:
        logger.error("Phrase generation failed: %s", e)
        return {"phrases": list(FALLBACK_PHRASES), "error": f"generation_error: {e}"}
