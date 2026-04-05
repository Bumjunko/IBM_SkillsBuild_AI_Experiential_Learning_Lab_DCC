# CommCopilot

Real-time AI conversation assistant for international students. CommCopilot listens to live English conversations, detects hesitations (pauses and filler words), and suggests contextually appropriate phrases or words the student can use to continue the conversation naturally.

Built as part of the **IBM SkillsBuild AI Experiential Learning Lab**.

## How It Works

```
Microphone → Speech-to-Text → Hesitation Detection → LangGraph Pipeline → Phrase Suggestions
```

1. **Speech-to-Text (N1)** — IBM Watson STT transcribes the live conversation via WebSocket streaming
2. **Embedding (N2)** — Transcript chunks are embedded (IBM watsonx.ai) and stored in PostgreSQL + pgvector for retrieval
3. **Context Inference (N3)** — IBM Granite analyzes the transcript to determine speaker roles, tone, formality, and student intent
4. **Retrieval (N5)** — If context confidence is low, relevant past conversation history is retrieved via vector similarity search
5. **Phrase Generation (N6)** — Granite generates 3 short, natural phrases the student could say next
6. **Safety Filter (N7)** — All suggested phrases are screened for appropriateness before display

The pipeline is orchestrated as a **LangGraph StateGraph** with parallel fan-out (embedding + context run simultaneously) and conditional routing (retrieval is skipped when confidence is high). (Can be changed to IBM orchastration)

## Scenarios? 

| Scenario | Description | Tone |
|---|---|---|
| **Office Hours** | Meeting with a professor about assignments, grades, or course material | Formal |
| **Admin Office** | University admin interactions — enrollment, housing, financial aid | Semi-formal |

## Tech Stack

- **Backend**: FastAPI + WebSocket
- **AI Pipeline**: LangGraph
- **LLM**: IBM watsonx.ai
- **Speech-to-Text**: IBM Watson STT
- **Database**: PostgreSQL + pgvector (IBM Cloud)
- **Frontend**: Vanilla HTML/CSS/JS

## Project Structure

```
├── commcopilot/           # Core pipeline
│   ├── config.py          # Environment variables, thresholds, scenarios
│   ├── state.py           # LangGraph PipelineState definition
│   ├── graph.py           # StateGraph build & compile
│   ├── prompts.py         # Granite prompt templates
│   ├── db.py              # PostgreSQL + pgvector connection
│   └── nodes/             # Pipeline nodes
│       ├── stt.py         # N1: Transcript update
│       ├── embedding.py   # N2: Embed & store
│       ├── context.py     # N3: Context inference
│       ├── retrieval.py   # N5: History retrieval
│       ├── generation.py  # N6: Phrase generation
│       └── safety.py      # N7: Safety filter
├── server/
│   ├── app.py             # FastAPI app with WebSocket endpoint
│   └── watson_stt.py      # Watson STT streaming client
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                 # pytest test suite
│   ├── conftest.py
│   └── test_nodes/
├── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- IBM Cloud account (watsonx.ai + Watson STT)

### Installation

```bash
git clone <repo-url>
cd IBM_SkillsBuild_AI_Experiential_Learning_Lab_DCC
python -m venv .venv
source .venv/bin/activate  # Windows: source .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# IBM watsonx.ai
WATSONX_API_KEY=
WATSONX_PROJECT_ID=
WATSONX_URL=
AGENT_MODEL_ID=
EMBEDDING_MODEL_ID=

# IBM Watson STT
WATSON_STT_API_KEY=
WATSON_STT_URL=

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/commcopilot
```

### Run

```bash
python -m server.app
```

The app starts at `http://localhost:8000`.

### Tests

```bash
pytest
```
