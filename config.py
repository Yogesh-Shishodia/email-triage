"""Central configuration for the triage app.

Override any value via environment variables or a local .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths -----------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---- Gmail OAuth -----------------------------------------------------------
# `gmail.modify` covers read + send + label changes (no permanent delete).
# Switching from readonly invalidates any cached token; the engine handles
# that automatically and re-prompts for consent.
GMAIL_SCOPES: list[str] = ["https://www.googleapis.com/auth/gmail.modify"]
GMAIL_CREDENTIALS_FILE: Path = BASE_DIR / "credentials.json"
GMAIL_TOKEN_FILE: Path = DATA_DIR / "token.json"

# ---- Database --------------------------------------------------------------
DB_PATH: Path = DATA_DIR / "triage.db"
DB_URL: str = f"sqlite:///{DB_PATH}"

# ---- Ollama ----------------------------------------------------------------
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---- Triage runtime --------------------------------------------------------
DAYS_BACK: int = int(os.getenv("DAYS_BACK", "7"))
MAX_BODY_CHARS: int = int(os.getenv("MAX_BODY_CHARS", "4000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
MAX_EMAILS: int = int(os.getenv("MAX_EMAILS", "30"))

# Concurrent LangGraph workers. 2–3 is the sweet spot for a 7B model on
# Apple-silicon Macs; higher risks Ollama queueing or memory thrashing.
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "3"))
