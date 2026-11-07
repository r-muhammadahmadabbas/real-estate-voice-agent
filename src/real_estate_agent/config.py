"""Central configuration for the RealEstate Hub voice agent.

All paths resolve relative to the project root, so the repository can be cloned
anywhere. Every location can also be overridden with an environment variable,
which is what the Docker image and Compose file use.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

# Load .env from the project root (if present) before reading any variables.
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs"))

FAQ_PATH = Path(os.getenv("FAQ_PATH", DATA_DIR / "faqs.csv"))
PROPERTIES_PATH = Path(os.getenv("PROPERTIES_PATH", DATA_DIR / "properties.csv"))
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "real_estate.db"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", PROJECT_ROOT / "chroma_db"))

# Optional integrations. The agent degrades gracefully when they are absent.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# Outbound email (appointment notifications). Left empty in local development.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
AGENT_EMAIL = os.getenv("AGENT_EMAIL", "")

def ensure_directories() -> None:
    """Create the writable directories the application needs."""
    for directory in (DATA_DIR, LOG_DIR, CHROMA_PATH):
        directory.mkdir(parents=True, exist_ok=True)
