import os
from dotenv import load_dotenv

load_dotenv()

# IBM watsonx.ai
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "")
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "")

# IBM Watson STT
WATSON_STT_API_KEY = os.getenv("WATSON_STT_API_KEY", "")
WATSON_STT_URL = os.getenv("WATSON_STT_URL", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Pipeline thresholds
CONFIDENCE_THRESHOLD = 0.6
HESITATION_PAUSE_MS = 1500
PHRASE_AUTO_DISMISS_S = 5
LATENCY_BUDGET_S = 4.0

# Safety
MAX_SAFETY_RETRIES = 1
MIN_SAFE_PHRASES = 2

FALLBACK_PHRASES = [
    "Could you repeat that, please?",
    "Let me think about that for a moment.",
    "I understand. Thank you.",
]

# Filler word patterns (used server-side on STT transcript)
FILLER_WORDS = ["um", "uh", "uh", "like", "you know", "er", "ah"]

# Scenarios
SCENARIOS = {
    "office_hours": {
        "name": "Office Hours with Professor",
        "default_role": "professor",
        "default_tone": "formal",
        "system_context": (
            "The student is in a professor's office during office hours. "
            "The conversation is academic and formal. The professor discusses "
            "deadlines, assignments, grades, or course material."
        ),
    },
    "admin_office": {
        "name": "Admin Office Interaction",
        "default_role": "admin_staff",
        "default_tone": "semi-formal",
        "system_context": (
            "The student is at a university administrative office. "
            "The conversation is semi-formal and procedural, covering topics like "
            "enrollment, housing, financial aid, or document requests."
        ),
    },
}
