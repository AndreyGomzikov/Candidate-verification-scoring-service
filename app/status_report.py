"""Plain text Telegram history report; no HTML parsing or implied AI verdicts."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from .models import Candidate, CooldownDeclinedJoinRequest, JoinRequestEvent

# Telegram has a 4096 UTF-16 code-unit text limit, including astral emoji.
_TEXT_LIMIT = 3800
_ANSWER_LIMIT = 100  # compact view; full answers remain in PostgreSQL
_REASON_LIMIT = 120


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _clip(value: str | None, max_units: int) -> str:
    clean = " ".join((value or "").split()) or "—"
    if _utf16_length(clean) <= max_units:
        return clean
    used = 0
    output = []
    for character in clean:
        units = _utf16_length(character)
        if used + units > max_units - 1:
            break
        output.append(character)
        used += units
    return "".join(output) + "…"


def _time(value: datetime | None) -> str:
    if not value:
        return "время неизвестно"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")


def _request_missing(candidate: Candidate) -> bool:
    """Legacy rows can be expired even when Telegram request disappeared."""
    reason = (candidate.decision_reason or "").casefold()
    return candidate.status == "request_missing" or (
        candidate.status == "expired" and (
            "заявка отсутствует в telegram" in reason or "hide_requester_missing" in reason
        )
    )


def _verdict(candidate: Candidate) -> str:
    if _request_missing(candidate):
        return "ЗАЯВКА ОТСУТСТВУЕТ В TELEGRAM"
    if candidate.status == "pending_admin" and any(
        word in (candidate.decision_reason or "").casefold() for word in (
            "не удалось начать лс", "не удалось повторно отправить вопрос",
            "не подтверждена отправка первого вопроса",
            "telegram decline failed", "interview timeout, telegram decline failed",
        )
    ):
        return "ТЕХНИЧЕСКАЯ ПРОБЛЕМА — РЕШЕНИЕ АДМИНИСТРАТОРА"
    return {
        "approved": "ПРОШЁЛ (одобрен администратором)",
        "rejected": "НЕ ПРОШЁЛ (отклонён администратором)",
        "auto_rejected": "НЕ ПРОШЁЛ (автоматический отказ)",
        "expired": "Анкета не завершена. Баллы: не рассчитывались",
        "superseded": "ЗАЯВКА ЗАМЕНЕНА ПОВТОРНОЙ",
        "request_missing": "ЗАЯВКА ОТСУТСТВУЕТ В TELEGRAM",
        "pending_admin": "ОЖИДАЕТ РЕШЕНИЯ АДМИНИСТРАТОРА",
        "interview": "ПРОХОДИТ АНКЕТУ",
        "scoring": "ОЦЕНИВАЕТСЯ ИИ",
        "error": "ТЕХНИЧЕСКАЯ ОШИБКА",
    }.get(candidate.status, f"СТАТУС: {candidate.status}")


def _record(candidate: Candidate) -> str:
    """Concise record that never treats an unfinished interview as zero points."""
    nickname = ("@" + _clip(candidate.username, 60)) if candidate.username else "без username"
    full_name = _clip(candidate.first_name, 80)
    date = _time(candidate.request_date or candidate.created_at)
    verdict = _verdict(candidate)
    if candidate.status == "expired" and not _request_missing(candidate):
        # Exact wording requested by the moderator, no extra score line.
        result = f"Итог: {verdict}"
    else:
        score = (f"{candidate.confidence_score}/100" if candidate.confidence_score is not None
                 else "не рассчитывались")
        result = f"Итог: {verdict} | Баллы: {score}"
    return "\n".join((
        f"№{candidate.id} | {date} | {full_name} ({nickname}) | TG ID {candidate.user_id}",
        result,
        "Ответы:",
        f"1) Локация: {_clip(candidate.location_answer, _ANSWER_LIMIT)}",
        f"2) Бизнес: {_clip(candidate.business_answer, _ANSWER_LIMIT)}",
        f"3) Цель вступления: {_clip(candidate.goal_answer, _ANSWER_LIMIT)}",
        f"Причина: {_clip(candidate.decision_reason, _REASON_LIMIT)}",
    ))


def _confirmed_cooldown_decline(item: CooldownDeclinedJoinRequest) -> bool:
    """Confirmed Telegram declines, including legacy rows without confirmed_at."""
    return item.status == "declined" or item.confirmed_at is not None


def _cooldown_record(item: CooldownDeclinedJoinRequest) -> str:
    if _confirmed_cooldown_decline(item):
        status = "НЕ ПРОШЁЛ (повторная заявка в период ограничения)"
    else:
        status = "ИТОГ НЕ ПОДТВЕРЖДЁН (попытка отклонения по cooldown)"
    return "\n".join((
        f"Повторная попытка C{item.id} | {_time(item.request_date)} | TG ID {item.user_id}",
        f"Итог: {status}",
        "Ответы: не запрашивались | Баллы: не рассчитывались",
    ))


def build_status_report(
    candidates: list[Candidate],
    cooldowns: list[CooldownDeclinedJoinRequest],
    unlinked_events: list[JoinRequestEvent] | None = None,
) -> tuple[str, list[str]]:
    """A summary and Telegram-safe pages containing every persisted attempt."""
    unlinked_events = unlinked_events or []
    counts = Counter(c.status for c in candidates)
    missing = sum(1 for c in candidates if _request_missing(c))
    timed_out = sum(1 for c in candidates if c.status == "expired" and not _request_missing(c))
    # Same confirmation predicate as each cooldown history entry: pending and
    # uncertain Telegram outcomes are NOT counted as confirmed rejections.
    confirmed_cooldown_rejections = sum(1 for item in cooldowns if _confirmed_cooldown_decline(item))
    items: list[tuple[datetime, int, int, str]] = []
    for candidate in candidates:
        moment = candidate.request_date or candidate.created_at or datetime.min.replace(tzinfo=timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        items.append((moment, 0, candidate.id, _record(candidate)))
    for item in cooldowns:
        moment = item.request_date
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        items.append((moment, 1, item.id, _cooldown_record(item)))
    for event in unlinked_events:
        moment = event.request_date or event.created_at
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        description = (
            f"Событие Telegram U{event.update_id} | {_time(moment)} | TG ID {event.user_id}\n"
            "Итог: ТЕХНИЧЕСКАЯ ПРОБЛЕМА — АНКЕТА НЕ СОЗДАНА\n"
            f"Этап: {event.status}; ошибка: {event.error_type or 'неизвестна'}\n"
            "Ответы: не запрашивались | Баллы: не рассчитывались"
        )
        items.append((moment, 2, event.id, description))
    items.sort(key=lambda item: (item[0], item[1], item[2]))
    # Report covers every persisted join attempt (including cooldown and
    # technical events without an interview), not only candidate rows.
    # The report still uses UTC timestamps internally; the header shows dates
    # only, as requested by moderators.
    if items:
        first_day = items[0][0].astimezone(timezone.utc).strftime("%d.%m.%Y")
        last_day = items[-1][0].astimezone(timezone.utc).strftime("%d.%m.%Y")
        heading = f"SOLIDUS — история заявок Период: {first_day} - {last_day}"
    else:
        heading = "SOLIDUS — история заявок Период: нет заявок"
    summary = "\n".join((
        heading,
        f"Заявок в анкете: {len(candidates)} | Повторных попыток по cooldown: {len(cooldowns)}",
        f"ПРОШЁЛ: {counts['approved']} | НЕ ПРОШЁЛ: {counts['rejected'] + counts['auto_rejected'] + confirmed_cooldown_rejections}",
        f"Анкета не завершена: {timed_out} | Заявка отсутствовала: {missing}",
        f"Ожидают модератора: {counts['pending_admin']}",
        f"Технические события без анкеты: {len(unlinked_events)}",
        "Остальные состояния из БД (обработка/замена/ошибки) показаны ниже.",
        "Ответы сокращены до 100 символов, баллы показаны только при наличии оценки.",
    ))

    pages: list[str] = []
    current = ""
    for _, _, _, body in items:
        candidate_text = body if not current else current + "\n\n" + body
        if _utf16_length(candidate_text) > _TEXT_LIMIT and current:
            pages.append(current)
            current = body
        else:
            current = candidate_text
        if _utf16_length(current) > _TEXT_LIMIT:
            # Defensive for future schema changes; at present each row < 1000.
            raise ValueError("One status row is too long for Telegram")
    if current:
        pages.append(current)
    return summary, pages

