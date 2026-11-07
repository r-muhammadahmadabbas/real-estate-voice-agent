# RealEstate Hub - AI Voice Agent

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

A **production-ready, fully local** real estate voice agent that combines:

- **Rule-based intent detection** (no LLM required for core logic)
- **Retrieval‑Augmented Generation (RAG)** with ChromaDB for factual questions
- **Property recommendation engine** with preference filtering
- **Calendar & appointment management** (Google Calendar integration optional)
- **Text‑to‑Speech & Speech‑to‑Text** using free local models (Whisper, gTTS)
- **Multi‑turn conversation memory** with bounded history

Everything runs locally. No API keys are required for the core agent, LLM, or voice pipeline.

---

## Highlights

| Feature | Implementation |
| --- | --- |
| **Domain lock** | Guardrails refuse non‑real‑estate queries (weather, politics, crypto, …) |
| **RAG knowledge base** | ChromaDB stores FAQs & property descriptions; keyword fallback if no embeddings |
| **Preference‑driven recommendations** | Filter by budget, city, area, bedrooms, property type, purpose, amenities |
| **Appointment booking** | Google Calendar API (optional) or mock mode for local development |
| **Voice‑ready** | Whisper STT, gTTS TTS, or text‑only mode |
| **Multi‑turn memory** | Bounded conversation transcript, context‑aware responses |
| **Docker & Compose** | One‑command deployment with optional GPU/CPU |
| **CLI & Streamlit UI** | `python scripts/health_check.py`, `streamlit run apps/streamlit_voice.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interaction                        │
│ ──────────────────────────────────────────────────────────────  │
│  voice input (Whisper) ──────►  STT ──────►  agent.py ──────►│
│  text input ───────────────────────────────────────────────────│
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │   agent.py (LangGraph)││  rag_pipeline.py     │              │
│  │   └─ intent detect ─┘│  └─ load_data()       │              │
│  │   └─ search prop───►│  └─ initialize_vector_store│            │
│  │   └─ book appointment││  └─ search()          │              │
│  └─────────────────────┘  └─────────────────────┘              │
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │   tools.py           ││  recommendation.py   │              │
│  │   └─ check_cal______││  └─ filter_properties()│            │
│  │   └─ book_appointment││  └─ recommend()      │            │
│  └─────────────────────┘  └─────────────────────┘              │
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │   voice_pipeline.py  ││  config.py           │              │
│  │   └─ text_to_speech ││  └─ DATA_DIR, DB_PATH│              │
│  │   └─ is_whisper_available││  └─ LOG_DIR        │             │
│  └─────────────────────┘  └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Module | Responsibility |
| --- | --- |
| `agent.py` | 100% rule‑based intent detection (`greeting`, `buyer_inquiry`, `rental_inquiry`, `commercial_inquiry`, `investment_inquiry`, `off_topic`, `goodbye`, `booking_confirm`, `no_results`, …). Degrades gracefully when LLM or voice features are absent. |
| `rag_pipeline.py` | ChromaDB‑based knowledge base. Loads `faqs.csv` & `properties.csv`, embeds FAQs (keyword‑based fallback), provides `search(query, top_k)` and `search_properties(query, city, max_price, …)`. |
| `recommendation.py` | `PropertyRecommendationEngine` — scores properties by area match, budget fit, bedroom match, amenities, investment ROI, property type. Returns top‑k results with `recommendation_score`. |
| `tools.py` | Calendar availability (`check_calendar_availability`), alternative slots (`get_alternative_slots`), booking (`book_appointment`), email (`send_agent_notification`, `send_client_confirmation`), property search (`search_properties_from_text`, `rag_search`). |
| `voice_pipeline.py` | Whisper STT (local, free), gTTS TTS (free, online but no key required), robustness wrappers that never break the app. |
| `config.py` | All paths resolved relative to the project root (`DATA_DIR`, `DB_PATH`, `CHROMA_PATH`, `FAQ_PATH`, `PROPERTIES_PATH`). Environment‑variable overrides for Docker/Compose. |
| `memory.py` (in `agent.py`) | Bounded `ConversationMemory` — max turns, context turns, `last_user_message()`, `context()` methods. |

### Data Flow

1. **User input** (voice or text) → `voice_pipeline.py` → `agent.py`
2. `agent.py` detects intent → routes to:
   - **RAG search** → `rag_pipeline.py` → factual answer (FAQ or property detail)
   - **Recommendation engine** → `recommendation.py` → filtered property list
   - **Booking flow** → `tools.py` → calendar check → appointment confirmation
   - **Off‑topic** → pre‑written Urdu‑Lish response redirect
3. Response → `voice_pipeline.py` → TTS (or text display in Streamlit UI)

### Conversation Memory

The agent remembers the last *N* turns (default 12, context‑aware 5). Each turn is stored as `{"role": "user"/"assistant", "content": "..."}`. The `context()` method returns the most recent exchanges, enabling coherent multi‑turn conversations without a database.

### Guardrails (Domain Lock)

| Keyword Category | Behaviour |
| --- | --- |
| `weather`, `mausam`, `politics`, `siasat`, `cricket`, `khana`, `movie`, `film`, `song`, `gaana` | Refuse and redirect to real‑estate topics |
| `predict`, `odds`, `forecast` (without two teams) | Ask for clarification |
| Unknown property / budget out of range | `"Maazrat, is budget mein koi property nahi mili. Kya budget adjust kar sakte hain ya area change karna hai?"` |
| Every prediction / recommendation | Appended `"Yeh prediction sirf app-based data par basis hai, practical reality alag ho sakti hai."` disclaimer |

### HTTP API (optional)

```bash
uvicorn api:app --reload
```

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/chat` | Ask a natural‑language question |
| `POST` | `/recommend` | Get property recommendations |
| `POST` | `/book` | Book a property visit appointment |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Budget 3 crore hai, Lahore DHA mein, 3 bedroom chahiye"}'
```

Interactive docs: <http://localhost:8000/docs>

---

## Project Structure

```text
real-estate-voice-agent/
├── .github/workflows/ci.yml        # GitHub Actions CI
├── .env.example                    # Optional env overrides
├── .gitignore
├── Makefile                        # Common commands (install, test, api, chat, prepare-data, clean)
├── Dockerfile                      # Production Docker image
├── docker-compose.yml              # api + streamlit service
├── README.md                       # ←‑‑‑ This file
├── scripts/
│   ├── evaluate.py                 # 40‑+ test‑case evaluation suite
│   ├── health_check.py             # Check all components
│   └── prepare_data.py             # (optional) rebuild processed files
├── apps/
│   ├── streamlit_text.py           # Text‑only Streamlit UI
│   └── streamlit_voice.py          # Voice‑aware Streamlit UI
├── data/
│   ├── faqs.csv                    # FAQ lookup table (committed)
│   └── properties.csv              # Property database (committed)
│   └── chroma_db/                  # ChromaDB (created at runtime)
│   └── real_estate.db              # SQLite (created at runtime)
│   └── logs/                       # Agent logs (created at runtime)
├── scripts/
│   ├── evaluate.py
│   ├── health_check.py
│   └── prepare_data.py
├── src/real_estate_agent/
│   ├── __init__.py
│   ├── agent.py                    # LangGraph agent + rule‑based intent
│   ├── rag_pipeline.py             # ChromaDB RAG knowledge base
│   ├── recommendation.py           # Property recommendation engine
│   ├── tools.py                    # Calendar, email, search tools
│   ├── voice_pipeline.py           # Whisper STT / gTTS TTS
│   ├── config.py                   # Project‑relative paths + env overrides
│   └── voice_pipeline.py
└── tests/
    ├── test_system.py              # Quick smoke‑test (7/7 pass)
    └── test_evaluate.py            # Detailed evaluation suite
```

---

## Quick Start

```bash
# 1️⃣ Clone & set up
git clone https://github.com/r-muhammadahmadabbas/real-estate-voice-agent.git
cd real-estate-voice-agent

# 2️⃣ Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3️⃣ Install dependencies
pip install -e ".[dev]"         # installs all deps + dev tools

# 4️⃣ (Optional) Adjust env vars
# Copy .env.example to .env and set:
#   - LOG_DIR (default ./logs)
#   - DB_PATH (default ./real_estate.db)
#   - CHROMA_PATH (default ./chroma_db)
#   - GROQ_API_KEY (if you want LLM‑augmented drafting)
#   - DEEPGRAM_API_KEY (if you want DeepGRAM STT)
#   - CHROMA_COLLECTION_NAME (default real_estate_knowledge)

# 5️⃣ Run the system
# ----- Text‑only Streamlit UI -----
streamlit run apps/streamlit_text.py

# ----- Voice‑aware Streamlit UI -----
streamlit run apps/streamlit_voice.py

# ----- CLI health check -----
python scripts/health_check.py

# ----- API (optional) -----
uvicorn src.real_estate_agent.api:app --reload
```

---

## Testing

```bash
# Run the full evaluation suite (40+ test cases)
pytest -q

# Quick smoke test
python scripts/health_check.py
```

The test suite covers:

- **RAG pipeline** – loading data, initializing vector store, searching FAQs & properties
- **Recommendation engine** – filtering, scoring, top‑k output
- **Agent – greeting** – `Assalam-o-Alaikum` → `greeting` intent
- **Agent – buyer inquiry** – `Budget 3 crore hai, Lahore DHA mein` → `buyer_inquiry` intent
- **Agent – off‑topic refusal** – `What's the weather today?` → `off_topic` intent + redirect
- **Agent – booking** – calendar availability check, alternative slots
- **Multi‑turn conversation** – context‑aware responses across multiple user turns

---

## Docker

```bash
# Build & run (CPU)
docker compose up --build

# Build & run (GPU – if you have nvidia/cuda‑enabled base image)
docker compose up --build --gpus all
```

The compose file starts two services:

- **api** – FastAPI backend (`src/real_estate_agent.api:app`) on port `8000`
- **streamlit** – Streamlit UI (`apps/streamlit_voice.py`) on port `8501`

Environment variables can be set in the `compose.yml` or overridden via a `.env` file.

---

## Limitations & Next Steps

- **No live data feed** – property data is static (committed CSVs). To update, replace `data/faqs.csv` & `data/properties.csv` and re‑run `streamlit run apps/streamlit_voice.py`.
- **Voice I/O** – Whisper STT and gTTS TTS are free and local, but accuracy depends on microphone quality and audio clarity.
- **Google Calendar** – the booking flow optionally integrates with Google Calendar; set `GROQ_API_KEY` / calendar credentials in `.env` to enable.
- **LLM augmentation** – the agent is rule‑based by default. Set `GROQ_API_KEY` in `.env` to add an optional LLM‑drafting step without breaking the offline flow.
- **Scalability** – ChromaDB is embedded; for production scale you can replace it with Pinecone, Weaviate, or Qdrant.

---

## License

MIT. The UCI Adult dataset is used for analytical and educational purposes.


---

**Built with** ❤️ by Muhammad Ahmad Abbas – a fully local, privacy‑first real‑estate assistant.
