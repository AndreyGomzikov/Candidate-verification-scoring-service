"""Conservative evidence checks for automatic *spam* rejection.

The verifier confirms that a quoted statement is actually present in the
candidate's answer. It cannot determine whether a statement is truthful or
whether it semantically amounts to abuse; a separate LLM review does that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# Clear behavioural categories only. Poor fit, curiosity and vague goals are
# intentionally not categories of spam.
SPAM_CATEGORIES = frozenset({
    "mass_advertising",
    "fraudulent_scheme",
    "phishing",
    "aggressive_lead_generation",
    "explicit_abuse",
})
ANSWER_FIELDS = ("location", "business", "goal")


@dataclass(frozen=True)
class SpamEvidence:
    category: str
    source_field: str
    quote: str
    explanation: str


@dataclass(frozen=True)
class SpamVerification:
    # confirmed / not_confirmed / insufficient_evidence / invalid_review / error
    status: str
    reason: str
    category: str = ""
    source_field: str = ""
    quote: str = ""

    @property
    def permits_auto_reject(self) -> bool:
        return self.status == "confirmed"


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def quote_in_answer(quote: str, answer: str) -> bool:
    """Anchored literal excerpt, with whitespace and case normalized."""
    return (
        isinstance(quote, str)
        and 8 <= len(quote.strip()) <= 300
        and _normalized(quote) in _normalized(answer)
    )


def parse_primary_evidence(payload: object, *, answers: dict[str, str]) -> tuple[SpamEvidence, ...]:
    """Reject malformed, ungrounded or unrecognized proposed risk evidence."""
    if not isinstance(payload, list) or len(payload) > 4:
        return ()
    found: list[SpamEvidence] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        category = entry.get("category")
        field = entry.get("source_field")
        quote = entry.get("quote")
        explanation = entry.get("explanation")
        if (
            isinstance(category, str) and category in SPAM_CATEGORIES
            and isinstance(field, str) and field in ANSWER_FIELDS
            and isinstance(explanation, str) and 10 <= len(explanation.strip()) <= 500
            and quote_in_answer(quote, answers[field])
        ):
            found.append(SpamEvidence(category, field, quote.strip(), explanation.strip()))
    return tuple(found)


def validate_second_review(
    raw: str, *, answers: dict[str, str], primary: tuple[SpamEvidence, ...]
) -> SpamVerification:
    """A second LLM confirmation only counts with a separately grounded quote."""
    try:
        payload = json.loads(raw.strip())
    except (ValueError, TypeError):
        return SpamVerification("invalid_review", "Дополнительный AI-анализ вернул некорректный JSON")
    if not isinstance(payload, dict):
        return SpamVerification("invalid_review", "Дополнительный AI-анализ вернул некорректный объект")
    verdict = payload.get("verdict")
    if verdict in ("not_confirmed", "uncertain"):
        reason = payload.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 10:
            reason = "Дополнительный анализ не подтвердил явный спам"
        return SpamVerification("not_confirmed", reason.strip()[:300])
    if verdict != "confirmed":
        return SpamVerification("invalid_review", "Неизвестный результат повторной проверки")
    category = payload.get("category")
    field = payload.get("source_field")
    quote = payload.get("quote")
    reason = payload.get("reason")
    if (
        not isinstance(category, str) or category not in SPAM_CATEGORIES
        or not isinstance(field, str) or field not in ANSWER_FIELDS
        or not isinstance(reason, str) or len(reason.strip()) < 10
        or not quote_in_answer(quote, answers[field])
    ):
        return SpamVerification("invalid_review", "Неподтверждённые основания повторной проверки")
    # Both model passes must independently identify the same concrete passage
    # and kind of abuse, not just two unrelated snippets in the same application.
    if not any(
        item.category == category and item.source_field == field
        and (
            _normalized(quote) in _normalized(item.quote)
            or _normalized(item.quote) in _normalized(quote)
        )
        for item in primary
    ):
        return SpamVerification("invalid_review", "Две проверки не подтвердили один и тот же признак")
    return SpamVerification(
        "confirmed", reason.strip()[:300], category, field, quote.strip()
    )
