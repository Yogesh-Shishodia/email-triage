# AI Email Triage Dashboard — v2

Local-first email triage on macOS using LangGraph + Ollama (Qwen 2.5 7B) + Gmail + Streamlit.

**v2 changes:** send-via-Gmail, concurrent processing, rule-based pre-filter, newsletter unsubscribe surfacing.

## Architecture

```
gmail_engine.py ──► Gmail API (read + send via gmail.modify)
       │
       ▼
   agents.py  ──►  quick_classify (rule-based) ──┐
                                                  ├─► writes
                   LangGraph: triage→analyze→[draft]┘
       │              (Ollama / Qwen 2.5, JSON mode)
       ▼
  database.py ──►  SQLite (idempotent upsert by email_id)
       │
       ▼
     app.py    ──►  Streamlit dashboard (parallel sync, send button)
```

### How emails are processed

1. **Pre-filter (free):** Rule-based check on sender + headers. Catches obvious newsletters and promo blasts without any LLM call. Conservative — defers to the LLM whenever in doubt.
2. **Triage node (LLM):** Few-shot classifier → category + urgency.
3. **Analysis node (LLM):** Structured extraction → summary + entities.
4. **Drafting node (LLM, conditional):** Only runs for `Job Related` or `Important/Action Required`.

Step 1 typically removes 50–70% of an inbox before any LLM is invoked. The remaining emails are processed across `MAX_WORKERS` threads (default 3).

## Setup

### 1. Install Ollama and pull Qwen 2.5

```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:7b
```

### 2. Get Gmail OAuth credentials

1. <https://console.cloud.google.com/> → create or pick a project
2. Enable the **Gmail API**
3. Credentials → **Create OAuth client ID** → application type **Desktop app**
4. Download as `credentials.json` in this folder

**Scope:** This version uses `gmail.modify` — read + send + label changes, no permanent delete. If you previously ran v1 (read-only), the cached token is auto-invalidated and you'll see one fresh consent screen.

### 3. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

| Var              | Default                  | Notes                                       |
| ---------------- | ------------------------ | ------------------------------------------- |
| `OLLAMA_MODEL`   | `qwen2.5:7b`             | Any pulled Ollama model works               |
| `OLLAMA_HOST`    | `http://localhost:11434` |                                             |
| `DAYS_BACK`      | `7`                      | Sync window                                 |
| `MAX_BODY_CHARS` | `4000`                   | Long bodies are truncated for LLM context   |
| `MAX_WORKERS`    | `3`                      | Concurrent LangGraph invocations            |
| `LOG_LEVEL`      | `INFO`                   | `DEBUG` for prompt-level tracing            |

## Sending replies

The Send button is gated by two conditions: the email must have an AI-generated draft, and you must explicitly **Approve** it. Sends use Gmail's API with proper `In-Reply-To`, `References`, and `threadId` so replies thread correctly. Sent emails are recorded with `sent_at` and the buttons disable to prevent duplicates.

## Performance notes

On an M-series Mac with `qwen2.5:7b`, expect roughly:
- 2–4 sec per LLM-classified email
- Pre-filtered emails: <10 ms each
- 100-email week with 60% pre-filter rate, 3 workers ≈ 1–2 minutes

If you have 32GB+ RAM you can try `MAX_WORKERS=4`. Watch `Activity Monitor` for memory pressure.

## Layout

```
.
├── app.py              # Streamlit UI + parallel sync + send button
├── agents.py           # LangGraph workflow + pre-filter + few-shot prompts
├── database.py         # SQLAlchemy ORM + PRAGMA-based migration
├── gmail_engine.py     # OAuth + fetch + send (gmail.modify scope)
├── config.py           # Env-driven config
├── requirements.txt
├── credentials.json    # YOU provide (gitignore this)
└── data/
    ├── token.json      # cached OAuth token
    └── triage.db       # SQLite store
```
