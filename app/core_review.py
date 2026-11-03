"""Conservative evidence gate for automatic low location/business rejection.

A valid quote is a grounded fragment, NOT semantic proof. The second LLM
must independently confirm the negative meaning; uncertainty goes to a human.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .spam_review import quote_in_answer

CORE_FIELDS = frozenset({"location", "business"})


@dataclass(frozen=True)
class CoreEvidence:
    field: str
    quote: str
    explanation: str


@dataclass(frozen=True)
class CoreVerification:
    field: str
    status: str  # confirmed, not_confirmed, insufficient_evidence, invalid_review, error
    reason: str
    quote: str = ""

    @property
    def permits_auto_reject(self) -> bool:
        return self.status == "confirmed"


def parse_core_evidence(raw: object, *, answers: dict[str, str]) -> tuple[CoreEvidence, ...]:
    """Require precise candidate excerpts, not ungrounded model assertions."""
    if not isinstance(raw, dict):
        return ()
    items: list[CoreEvidence] = []
    for field in ("location", "business"):
        entry = raw.get(field)
        if not isinstance(entry, dict):
            continue
        quote = entry.get("quote")
        explanation = entry.get("explanation")
        if (
            quote_in_answer(quote, answers.get(field, ""))
            and isinstance(explanation, str)
            and 10 <= len(explanation.strip()) <= 500
        ):
            items.append(CoreEvidence(field, quote.strip(), explanation.strip()))
    return tuple(items)


def validate_core_review(
    raw: str, *, field: str, answer: str, primary: CoreEvidence
) -> CoreVerification:
    """Require field-specific, matching citations in both independent reviews."""
    if field not in CORE_FIELDS or primary.field != field:
        return CoreVerification(field, "invalid_review", "Некорректный критерий проверки")
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return CoreVerification(field, "invalid_review", "Некорректный JSON второй проверки")
    if not isinstance(payload, dict):
        return CoreVerification(field, "invalid_review", "Некорректный результат проверки")
    verdict = payload.get("verdict")
    reason = payload.get("reason")
    if verdict in {"not_confirmed", "uncertain"}:
        return CoreVerification(
            field, "not_confirmed",
            reason.strip()[:300] if isinstance(reason, str) and len(reason.strip()) >= 10
            else "Критическое несоответствие не подтверждено",
        )
    if verdict != "confirmed" or payload.get("field") != field:
        return CoreVerification(field, "invalid_review", "Неизвестный ответ или неверный критерий")
    quote = payload.get("quote")
    if not (
        isinstance(reason, str) and len(reason.strip()) >= 10
        and quote_in_answer(quote, answer)
    ):
        return CoreVerification(field, "invalid_review", "Не подтверждена цитата из ответа кандидата")
    # Second pass must identify the *same* passage, not another unrelated one.
    normalized_second = " ".join(quote.casefold().split())
    normalized_primary = " ".join(primary.quote.casefold().split())
    if not (normalized_second in normalized_primary or normalized_primary in normalized_second):
        return CoreVerification(field, "invalid_review", "Две проверки не совпали по цитате")
    return CoreVerification(field, "confirmed", reason.strip()[:300], quote.strip())
