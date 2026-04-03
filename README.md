# IBM SkillsBuild AI Experiential Learning Lab - DCC

A LangGraph ReAct agent project powered by IBM watsonx.ai, featuring persistent conversation memory via PostgreSQL.

## Project Structure

```
IBM_SkillsBuild_AI_Experiential_Learning_Lab_DCC/
├── langgraph-react-with-database-memory/   # watsonx.ai agent template
│   ├── src/                                # Agent source code
│   ├── schema/                             # Request/response schemas
│   ├── tests/                              # Tests
│   ├── examples/                           # Example scripts
│   ├── ai_service.py                       # AI service main logic
│   ├── config.toml                         # Deployment configuration
│   └── .env                                # Environment variables (API keys, etc.)
├── .gitignore
└── README.md
```

## Tech Stack

- **Python 3.12**
- **LangGraph** — ReAct agent framework
- **IBM watsonx.ai** — LLM model serving (Granite, etc.)
- **PostgreSQL** — Conversation memory storage
- **Poetry** — Package management

## Setup

### 1. Python Environment

Install Python 3.12 using pyenv and create a virtual environment.

#### Windows

```bash
# Install pyenv-win
pip install pyenv-win --target "$HOME/.pyenv"

# Add to PATH (via System Environment Variables):
#   PYENV = %USERPROFILE%\.pyenv\pyenv-win
#   PATH += %USERPROFILE%\.pyenv\pyenv-win\bin
#   PATH += %USERPROFILE%\.pyenv\pyenv-win\shims

# Install Python 3.12 and set up venv
pyenv install 3.12.10
pyenv local 3.12.10
python -m venv .venv
source .venv/Scripts/activate   # Git Bash
# or
.venv\Scripts\activate          # Command Prompt / PowerShell
```

#### macOS

```bash
# Install pyenv via Homebrew
brew install pyenv

# Add to shell profile (~/.zshrc or ~/.bash_profile)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Install Python 3.12 and set up venv
pyenv install 3.12.10
pyenv local 3.12.10
python -m venv .venv
source .venv/bin/activate
```

### 2. Install CLI

```bash
pip install -U ibm-watsonx-ai-cli
```

### 3. Install Dependencies

```bash
cd langgraph-react-with-database-memory
pip install poetry
poetry install --with dev
```

### 4. Environment Variables

Fill in your IBM Cloud credentials in `langgraph-react-with-database-memory/.env`.

```env
WATSONX_APIKEY=<your IBM Cloud API key>
WATSONX_URL=https://<REGION>.ml.cloud.ibm.com
WATSONX_SPACE_ID=<your deployment space ID>
```

### 5. Deployment Configuration

Configure the model and database in `langgraph-react-with-database-memory/config.toml`.

```toml
[deployment.online.parameters]
model_id = "ibm/granite-4-h-small"
url = "https://<REGION>.ml.cloud.ibm.com"
postgres_db_connection_id = "<your Postgres connection ID>"
```

## Usage

### Run Locally

```bash
cd langgraph-react-with-database-memory
watsonx-ai template invoke "Hello!"
```

### Deploy to IBM Cloud

```bash
cd langgraph-react-with-database-memory
watsonx-ai service new
```

### Query Deployed Service

```bash
watsonx-ai service invoke "<your question>"
```

## Testing

```bash
cd langgraph-react-with-database-memory
poetry run pytest -r 'fEsxX' tests/
```

## References

- [IBM watsonx.ai Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ai-services-templates.html?context=wx&audience=wdp)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Original Template Repository](https://github.com/IBM/watsonx-developer-hub/tree/main/agents/community/langgraph-react-with-database-memory)
