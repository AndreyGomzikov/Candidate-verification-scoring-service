from __future__ import annotations

import os
from dataclasses import dataclass


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int(name: str, default: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return int(raw)


def _int_set(name: str) -> set[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


@dataclass(frozen=True)
class Settings:
    bot_token: str
    target_chat_id: int
    admin_chat_id: int
    admin_user_ids: set[int]
    database_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: int
    auto_reject_spam_score: int
    auto_reject_core_score_max: int
    auto_reject_min_ai_confidence: int
    max_answer_length: int
    interview_timeout_minutes: int = 30
    reapplication_cooldown_minutes: int = 60
    maintenance_interval_seconds: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            bot_token=_required("BOT_TOKEN"),
            target_chat_id=_int("TARGET_CHAT_ID"),
            admin_chat_id=_int("ADMIN_CHAT_ID"),
            admin_user_ids=_int_set("ADMIN_USER_IDS"),
            database_url=os.getenv(
                "DATABASE_URL", "sqlite+aiosqlite:///./data/gatekeeper.db"
            ).strip(),
            llm_base_url=os.getenv(
                "LLM_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            llm_api_key=_required("LLM_API_KEY"),
            llm_model=_required("LLM_MODEL"),
            llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 35),
            auto_reject_spam_score=_int("AUTO_REJECT_SPAM_SCORE", 85),
            auto_reject_core_score_max=_int("AUTO_REJECT_CORE_SCORE_MAX", 15),
            auto_reject_min_ai_confidence=_int(
                "AUTO_REJECT_MIN_AI_CONFIDENCE", 80
            ),
            max_answer_length=_int("MAX_ANSWER_LENGTH", 2500),
            interview_timeout_minutes=max(1, _int("INTERVIEW_TIMEOUT_MINUTES", 30)),
            reapplication_cooldown_minutes=max(
                0, _int("REAPPLICATION_COOLDOWN_MINUTES", 60)
            ),
            maintenance_interval_seconds=max(
                15, _int("MAINTENANCE_INTERVAL_SECONDS", 60)
            ),
        )
