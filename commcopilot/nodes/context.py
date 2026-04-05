"""N3: Context Inference node. Infers role, tone, formality, intent, confidence."""

import json
import logging
from commcopilot.state import PipelineState
from commcopilot.config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    AGENT_MODEL_ID,
    SCENARIOS,
)
from commcopilot.prompts import context_inference_prompt

logger = logging.getLogger(__name__)


def infer_context(state: PipelineState) -> dict:
    """Run Granite to infer conversation context from transcript."""
    transcript = state.get("transcript", "")
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
            params={"max_new_tokens": 300, "temperature": 0.3},
        )

        prompt = context_inference_prompt(transcript, scenario["system_context"])
        response = model.generate_text(prompt=prompt)

        parsed = json.loads(response.strip())
        logger.info("Context inferred: role=%s, confidence=%.2f", parsed.get("role"), parsed.get("confidence", 0))

        return {
            "context_role": parsed.get("role"),
            "context_tone": parsed.get("tone"),
            "context_formality": parsed.get("formality"),
            "context_intent": parsed.get("intent"),
            "context_confidence": float(parsed.get("confidence")),
            "error": None,
        }

    except json.JSONDecodeError as e:
        logger.error("Context inference JSON parse error: %s", e)
        return {
            "context_role": scenario["default_role"],
            "context_tone": scenario["default_tone"],
            "context_formality": "medium",
            "context_intent": "respond_to_question",
            "context_confidence": 0.3,
            "error": f"context_json_error: {e}",
        }
    except Exception as e:
        logger.error("Context inference failed: %s", e)
        return {
            "context_role": scenario["default_role"],
            "context_tone": scenario["default_tone"],
            "context_formality": "medium",
            "context_intent": "respond_to_question",
            "context_confidence": 0.3,
            "error": f"context_error: {e}",
        }
