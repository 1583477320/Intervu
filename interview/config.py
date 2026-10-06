"""Central configuration, read from environment variables / .env file.

Interviews always run against a real configured LLM (OpenAI-compatible
endpoint). If OPENAI_API_KEY is empty the service can still start, but no
model-backed operation can run until a key is configured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of this package directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    db_path: Path = PROJECT_ROOT / "data" / "interview.db"
    wiki_lang: str = "zh"
    tavily_api_key: str = ""  # 面经检索（Tavily Search，可选）
    # interview tuning
    max_turns: int = 12
    transcript_window: int = 12

    @property
    def effective_base_url(self) -> str:
        return self.base_url.strip() or DEFAULT_BASE_URL

    @property
    def effective_model(self) -> str:
        return self.model.strip() or DEFAULT_MODEL


def load_settings() -> Settings:
    def _opt(name: str) -> str:
        return os.environ.get(name, "").strip()

    db = _opt("DB_PATH")
    return Settings(
        api_key=_opt("OPENAI_API_KEY"),
        base_url=_opt("OPENAI_BASE_URL"),
        model=_opt("OPENAI_MODEL"),
        db_path=Path(db) if db else PROJECT_ROOT / "data" / "interview.db",
        wiki_lang=_opt("WIKI_LANG") or "zh",
        tavily_api_key=_opt("TAVILY_API_KEY"),
    )
