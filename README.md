# AI Multi-Agent Orchestrator

A production-grade multi-agent AI system built with **FastAPI**, **LangGraph**, and **Gemini**.

Submit any task in plain text. The system automatically routes it to the right AI agents, executes them in sequence, and returns structured JSON — with long-term memory across sessions.

---

## Architecture

```
POST /api/v1/task
       │
       ▼
  ┌─────────────┐
  │ Router Agent│  Classifies task, picks agents + strategy
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │              LangGraph StateGraph           │
  │                                             │
  │  [retrieval] → [agent1] → [agent2] → ...   │
  │                                             │
  │  Each agent receives the previous agent's   │
  │  output as context (pipeline pattern)       │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Aggregator │  Merges outputs, picks primary result
              └──────┬──────┘
                     │
                     ├──► ChromaDB  (saves to long-term memory)
                     │
                     ▼
              Structured JSON response
```

### Agents

| Agent | Purpose | Temperature |
|---|---|---|
| **Router** | Classifies task, selects agents + strategy | 0.1 |
| **Research** | Analyzes topics, extracts key facts + insights | 0.3 |
| **Coding** | Writes, debugs, reviews, explains code | 0.1 |
| **Summarization** | Condenses content, extracts action items | 0.2 |
| **Outreach** | Writes cold emails, follow-ups, proposals | 0.7 |
| **Retrieval** | Semantic search over past task memory | — |

### Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph StateGraph |
| LLM | Gemini 2.5 Flash (Google AI) |
| Embeddings | Gemini Embedding-001 (3072-dim) |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Validation | Pydantic v2 |
| Logging | structlog (structured JSON logs) |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/ai-orchestrator
cd ai-orchestrator
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
GEMINI_API_KEY=your_key_here   # https://aistudio.google.com/apikey
```

### 3. Start the server

```bash
python run_api.py
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API

### POST /api/v1/task

Submit any task. The orchestrator handles routing automatically.

**Request:**
```json
{
  "message": "Research what LangGraph is and write a cold email to their sales team",
  "priority": "high"
}
```

**Response:**
```json
{
  "task_id": "2ca1779c-b483-4e1b-b34c-553da95d2f55",
  "status": "complete",
  "agents_used": ["research", "outreach"],
  "execution_strategy": "sequential",
  "primary_output": "Hi [Recipient Name], ...",
  "all_outputs": {
    "research": { "topic": "LangGraph", "summary": "...", "key_facts": [...] },
    "outreach": { "email_subject": "...", "email_body": "..." }
  },
  "performance": {
    "total_tokens": 2841,
    "total_execution_ms": 9200
  },
  "routing": {
    "task_type": "research_and_outreach",
    "confidence": 0.95,
    "primary_agent": "outreach",
    "reasoning": "Task requires research followed by email writing"
  }
}
```

**Example tasks:**

```bash
# Single agent — coding
curl -X POST http://localhost:8000/api/v1/task \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a Python function to check if a number is prime"}'

# Multi-agent pipeline — research + outreach
curl -X POST http://localhost:8000/api/v1/task \
  -H "Content-Type: application/json" \
  -d '{"message": "Research Anthropic and write a cold email to their Head of Product"}'

# With context
curl -X POST http://localhost:8000/api/v1/task \
  -H "Content-Type: application/json" \
  -d '{"message": "Write unit tests", "priority": "high", "context": {"framework": "pytest"}}'
```

### GET /api/v1/health

System status check.

```json
{
  "status": "healthy",
  "agents_available": ["coding", "outreach", "research", "retrieval", "summarization"],
  "memory_count": 12
}
```

### GET /api/v1/memories

List past tasks stored in vector memory.

```bash
GET /api/v1/memories?limit=10
```

### POST /api/v1/memories/search

Semantic search over past tasks using Gemini embeddings.

```json
{ "query": "Python sorting algorithms", "n_results": 5 }
```

---

## Project Structure

```
ai-orchestrator/
├── agents/
│   ├── base_agent.py          # shared Gemini client, retry logic, timers
│   ├── router_agent.py        # task classification
│   ├── coding_agent.py
│   ├── research_agent.py
│   ├── summarization_agent.py
│   ├── outreach_agent.py
│   └── retrieval_agent.py     # ChromaDB vector search (no LLM)
├── orchestrator/
│   ├── graph.py               # LangGraph StateGraph definition
│   ├── nodes.py               # node functions (router, agents, aggregator)
│   ├── edges.py               # conditional routing logic
│   └── state.py               # shared state schema
├── api/
│   ├── main.py                # FastAPI app, middleware, exception handlers
│   ├── middleware.py          # RequestID + RequestLogging middleware
│   ├── schemas.py             # Pydantic request/response models
│   └── routes/
│       ├── task.py            # POST /task
│       ├── health.py          # GET /health
│       └── memory.py          # GET/POST /memories
├── core/
│   ├── config.py              # settings (loaded from .env)
│   ├── memory.py              # ChromaDB + Gemini embeddings singleton
│   └── logger.py              # structlog setup
├── tests/
│   ├── test_agents.py         # individual agent tests
│   ├── test_orchestrator.py   # full pipeline tests
│   ├── test_memory.py         # ChromaDB + embedding tests
│   ├── test_api.py            # FastAPI endpoint tests
│   └── test_production.py     # middleware + error handling tests
├── .env.example
├── requirements.txt
└── run_api.py                 # server entry point
```

---

## HTTP Status Codes

| Code | Meaning |
|---|---|
| `200` | Task completed successfully |
| `422` | Invalid request (missing/bad fields) |
| `503` | Gemini quota exhausted — retry after 60s |
| `504` | Task timed out (3 min limit) |
| `500` | Internal server error |

Every error response includes a `request_id` field matching the `X-Request-ID` response header — use it to trace issues in logs.

---

## Running Tests

```bash
# Production features (no API calls needed)
python tests/test_production.py

# Full API tests (requires valid GEMINI_API_KEY)
python tests/test_api.py

# Memory + embeddings
python tests/test_memory.py

# Orchestrator pipeline
python tests/test_orchestrator.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Google AI API key |
| `APP_ENV` | No | `development` | `development` / `production` |
| `API_PORT` | No | `8000` | Server port |
| `AGENT_TIMEOUT_SECONDS` | No | `30` | Per-agent timeout |
| `CHROMA_PERSIST_DIR` | No | `./data/chromadb` | Vector DB storage path |

---

## Design Decisions

**Why LangGraph instead of a simple loop?**  
LangGraph gives us a proper state machine with typed state, conditional edges, and a clear separation between routing logic and agent execution. Adding a new agent is one line in the registry.

**Why ChromaDB for memory?**  
Persistent vector search with cosine similarity, zero infrastructure required (embedded DB, no separate server). The retrieval agent injects relevant past context into new tasks automatically.

**Why Gemini?**  
Generous free tier for development, strong structured output support (`response_schema`), and competitive quality on coding and reasoning tasks.

**Why structured output (`response_mime_type: application/json`)?**  
Agents return validated Pydantic models, not freeform text. This makes downstream processing deterministic — no prompt engineering needed to extract data from prose.
