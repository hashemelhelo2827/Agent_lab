# Agent Lab

A progressive hands-on journey through LLM application development — from raw OpenAI API calls to multi-agent LangGraph workflows.

```mermaid
flowchart TB
    subgraph Lab1["🗄️ 01 — OpenAI Basics"]
        direction TB
        A0["<b>Input</b><br/><i>User provides ingredients,<br/>location, or complaint</i>"]:::input
        A1["<b>recipe_generator.py</b><br/><i>3 ingredients → JSON recipe</i>"]:::file
        A2["<b>trip_planner.py</b><br/><i>Location + days → itinerary</i>"]:::file
        A3["<b>ticket_classifier.py</b><br/><i>Complaint → sentiment,<br/>dept, urgency, order ID</i>"]:::file
        A["<b>⚡ Key Skill</b><br/>JSON structured output<br/>via OpenAI API"]:::skill
        A1 ~~~ A2 ~~~ A3
        A1 --> A
        A2 --> A
        A3 --> A
    end
    subgraph Lab2["📦 02 — LangChain + Pydantic"]
        direction TB
        B1["<b>ticket_router.py</b><br/><i>LangChain ticket classifier<br/>with ChatPromptTemplate</i>"]:::file
        B2["<b>product_tagger.py</b><br/><i>Product categorization<br/>+ restricted item detection</i>"]:::file
        B3["<b>customs_manifest_analyzer.py</b><br/><i>Nested cargo manifests<br/>+ hazard detection</i>"]:::file
        B["<b>⚡ Key Skill</b><br/>Pydantic schemas +<br/>LangChain chains"]:::skill
        B1 --- B2 --- B3
        B1 --> B
        B2 --> B
        B3 --> B
    end
    subgraph Lab3["🔍 03 — RAG + Chroma Vector DB"]
        direction TB
        C1["<b>customs_rag.py</b><br/><i>Cargo analysis vs<br/>rulebook.txt laws</i>"]:::file
        C2["<b>baggage_routing_rag.py</b><br/><i>Baggage screening vs<br/>baggage_rules.txt</i>"]:::file
        C3["<b>biodome_inspection_rag.py</b><br/><i>Bio-dome cargo vs<br/>biodome_rules.txt</i>"]:::file
        R["<b>📜 Rules</b><br/>rulebook.txt<br/>baggage_rules.txt<br/>biodome_rules.txt"]:::rules
        C["<b>⚡ Key Skill</b><br/>Chunk → Embed →<br/>Retrieve → Augment"]:::skill
        C1 --- C2 --- C3
        R -.- C1
        R -.- C2
        R -.- C3
        C1 --> C
        C2 --> C
        C3 --> C
    end
    subgraph Lab4["🤖 04 — LangGraph Agents"]
        direction TB
        D1["<b>code_generator_agent.py</b><br/><i>Generate → Evaluate<br/>→ Loop (max 3x)</i>"]:::file
        D2["<b>email_triage_agent.py</b><br/><i>Triage → Route to<br/>Escalation or Standard</i>"]:::file
        D3["<b>research_agent.py</b><br/><i>Questions → Parallel<br/>Research → Fact-check</i>"]:::file
        D["<b>⚡ Key Skill</b><br/>Stateful graphs +<br/>conditional branching"]:::skill
        D1 --- D2 --- D3
        D1 --> D
        D2 --> D
        D3 --> D
    end
    subgraph Next["⏭️ 05 — Next Lab"]
        direction TB
        E["<b>Coming soon</b>"]:::empty
    end

    Lab1 -->|"<b>LangChain Chains</b>"| Lab2
    Lab2 -->|"<b>RAG Retrieval</b>"| Lab3
    Lab3 -->|"<b>State Machines</b>"| Lab4
    Lab4 -....->|"<b>?</b>"| Next

    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef file fill:#ffffff,stroke:#546e7a,stroke-width:1px,color:#000
    classDef skill fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000
    classDef rules fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,color:#000
    classDef empty fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray:5 5,color:#000

    style Lab1 fill:#e1f5fe,stroke:#0288d1,stroke-width:3px,color:#000
    style Lab2 fill:#e8f5e9,stroke:#388e3c,stroke-width:3px,color:#000
    style Lab3 fill:#fff3e0,stroke:#f57c00,stroke-width:3px,color:#000
    style Lab4 fill:#fce4ec,stroke:#d32f2f,stroke-width:3px,color:#000
```

---

## Project Structure

```
Agent_lab/
├── 01_openai_basics/           # OpenAI API + JSON structured output
│   ├── recipe_generator.py
│   ├── trip_planner.py
│   └── ticket_classifier.py
├── 02_langchain_pydantic/      # LangChain chains + Pydantic schemas
│   ├── ticket_router.py
│   ├── product_tagger.py
│   └── customs_manifest_analyzer.py
├── 03_rag_vector_db/           # RAG with Chroma vector database
│   ├── customs_rag.py
│   ├── baggage_routing_rag.py
│   ├── biodome_inspection_rag.py
│   ├── baggage_rules.txt
│   ├── biodome_rules.txt
│   └── rulebook.txt
├── 04_langgraph_agents/        # LangGraph stateful agent workflows
│   ├── code_generator_agent.py
│   ├── email_triage_agent.py
│   └── research_agent.py
├── 05_/                        # Placeholder — next lab
├── chromadb_practice/          # Raw ChromaDB (no LangChain wrapper)
│   ├── rag_pipeline.py
│   └── NOTES.md
├── chroma_db/                  # Auto-generated vector store
├── openai-venv/                # Python virtual environment (gitignored)
└── .gitignore
```

---

## Labs Breakdown

### 01 — OpenAI Basics

| File | What it does | Key Concepts |
|------|-------------|--------------|
| `recipe_generator.py` | Takes 3 fridge ingredients → returns structured recipe JSON | `response_format={"type": "json_object"}`, basic API call |
| `trip_planner.py` | Given location + days → generates a day-by-day itinerary | Prompt engineering, JSON parsing with `json.loads()` |
| `ticket_classifier.py` | Classifies customer complaints (sentiment, department, urgency, order ID) | Few-shot prompting, structured extraction, conditional routing logic |

**What you learn:** How to call OpenAI-compatible APIs, enforce JSON output, parse structured responses, and use few-shot examples.

---

### 02 — LangChain + Pydantic

| File | What it does | Key Concepts |
|------|-------------|--------------|
| `ticket_router.py` | LangChain reimplementation of the ticket classifier | `ChatPromptTemplate`, `JsonOutputParser`, `ChatOpenAI` |
| `product_tagger.py` | Tags product category/condition, detects restricted items | Optional fields, boolean validation via Pydantic |
| `customs_manifest_analyzer.py` | Analyzes shipping container cargo with nested item manifests | Nested Pydantic models, hazard detection, value calculation |

**What you learn:** Replace manual API calls with LangChain chains, define output schemas with Pydantic, handle nested data structures.

---

### 03 — RAG with Chroma Vector DB

| File | What it does | Key Concepts |
|------|-------------|--------------|
| `customs_rag.py` | Analyzes cargo against `rulebook.txt` customs laws | `TextLoader`, `RecursiveCharacterTextSplitter`, Chroma vector store |
| `baggage_routing_rag.py` | Routes airport baggage per `baggage_rules.txt` security rules | `RunnableParallel`, context retrieval, `format_docs` |
| `biodome_inspection_rag.py` | Inspects bio-dome cargo (biomass/machinery) per `biodome_rules.txt` | Discriminated unions, quarantine protocol override logic |

**Rule files:**
- `rulebook.txt` — 5 customs risk classification rules (101–105)
- `baggage_rules.txt` — 3 airport security & routing rules (A–C)
- `biodome_rules.txt` — 4 planetary bio-dome safety protocols (Alpha–Gamma + Override)

**What you learn:** Load and chunk documents, embed them into a vector DB, retrieve relevant context, and augment LLM prompts with retrieved data.

---

### 04 — LangGraph Agents

| File | What it does | Key Concepts |
|------|-------------|--------------|
| `code_generator_agent.py` | Generates Python code, evaluates syntax + requirements, loops up to 3 iterations for self-correction | `StateGraph`, conditional edges, max-iteration guard |
| `email_triage_agent.py` | Triages incoming email, routes to escalation or standard processing based on sentiment + category | Branching workflows, `TypedDict` state, multi-node graph |
| `research_agent.py` | Generates 3 research questions → parallel research workers → compiles report → fact-checks → recompiles if contradictory | `Send` for parallel fan-out, fact-checking validation loop |

**What you learn:** Build stateful multi-node workflows, use conditional branching, implement parallel execution with `Send`, create self-correcting loops with validation gates.

---

## Prerequisites

- Python 3.11+
- API keys (store in `.env` files — already gitignored):
  - **Gemini API key** — from [Google AI Studio](https://aistudio.google.com/)
  - **Groq API key** — from [Groq Console](https://console.groq.com/) (required only for `research_agent.py`)

## Setup

```bash
git clone https://github.com/hashemelhelo2827/Agent_lab.git
cd Agent_lab

pip install openai langchain langchain-openai langchain-chroma langchain-google-genai langchain-community langgraph chromadb sentence-transformers python-dotenv pydantic
```

Add your API keys to `03_rag_vector_db/.env` and/or `openai-venv/.env` using the format:
```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here    # optional — only for research_agent.py
```

## Usage

Run any lab file independently — no cross-folder dependencies:

```bash
cd 01_openai_basics
python recipe_generator.py

cd ../03_rag_vector_db
python customs_rag.py
```

## Notes

- `.env` files and `chroma_db/` folders are gitignored — secrets stay local, vector stores regenerate on first run
- Each file is self-contained — you can run them in any order
- `05_/` is reserved for the next lab in the progression
- The project uses Gemini models via OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`) and `langchain_google_genai` for embeddings
