"""FastAPI application with WebSocket endpoint for CommCopilot."""

import json
import re
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from commcopilot.config import FILLER_WORDS, SCENARIOS, FALLBACK_PHRASES
from commcopilot.graph import pipeline
from commcopilot.state import PipelineState
from commcopilot import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Filler word regex (built from config)
_filler_pattern = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b",
    re.IGNORECASE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    try:
        await db.init_schema()
        logger.info("Database schema ready")
    except Exception as e:
        logger.warning("Database not available: %s. Running without persistence.", e)
    yield
    await db.close_pool()


app = FastAPI(title="CommCopilot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


@app.get("/api/scenarios")
async def get_scenarios():
    return {key: {"name": val["name"]} for key, val in SCENARIOS.items()}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    scenario = "office_hours"
    transcript_buffer = ""
    phrases_used: list[str] = []

    logger.info("Session started: %s", session_id)

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "scenario":
                scenario = msg.get("scenario", "office_hours")
                logger.info("Scenario set: %s (session %s)", scenario, session_id)

            elif msg_type == "transcript":
                text = msg.get("text", "")
                transcript_buffer = text
                # Check for filler words (server-side hesitation detection)
                if _filler_pattern.search(text):
                    logger.info("Filler word detected in transcript, triggering pipeline")
                    result = await _run_pipeline(session_id, scenario, transcript_buffer, "filler")
                    await ws.send_text(json.dumps({
                        "type": "phrases",
                        "phrases": result.get("safe_phrases", list(FALLBACK_PHRASES)),
                    }))

            elif msg_type == "pause":
                duration_ms = msg.get("duration_ms", 0)
                logger.info("Pause detected: %dms (session %s)", duration_ms, session_id)
                result = await _run_pipeline(session_id, scenario, transcript_buffer, "pause")
                await ws.send_text(json.dumps({
                    "type": "phrases",
                    "phrases": result.get("safe_phrases", list(FALLBACK_PHRASES)),
                }))

            elif msg_type == "phrase_selected":
                phrase = msg.get("phrase", "")
                phrases_used.append(phrase)
                logger.info("Phrase selected: %s", phrase)

            elif msg_type == "end_session":
                logger.info("Session ending: %s", session_id)
                recap = f"Session completed. You used {len(phrases_used)} suggested phrases."
                await ws.send_text(json.dumps({
                    "type": "recap",
                    "recap": recap,
                    "phrases_used": phrases_used,
                }))

            elif msg_type == "audio":
                # Audio data would be forwarded to Watson STT here.
                # For now, we expect transcripts to come from the browser directly.
                pass

    except WebSocketDisconnect:
        logger.info("Session disconnected: %s", session_id)
    except Exception as e:
        logger.error("WebSocket error (session %s): %s", session_id, e)


async def _run_pipeline(session_id: str, scenario: str, transcript: str, trigger: str) -> dict:
    """Run the LangGraph pipeline and return the final state."""
    initial_state: PipelineState = {
        "session_id": session_id,
        "scenario": scenario,
        "transcript": transcript,
        "session_history": [],
        "hesitation_trigger": trigger,
        "embedding_stored": False,
        "context_role": "",
        "context_tone": "",
        "context_formality": "",
        "context_intent": "",
        "context_confidence": 0.0,
        "relevant_history": [],
        "phrases": [],
        "safe_phrases": [],
        "error": None,
    }

    try:
        result = await pipeline.ainvoke(initial_state)
        return result
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        return {"safe_phrases": list(FALLBACK_PHRASES), "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
