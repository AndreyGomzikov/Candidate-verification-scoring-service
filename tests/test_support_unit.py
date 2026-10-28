"""Support routing tests that work without optional async database drivers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from telegram.error import TelegramError

from app import texts
from app.config import Settings
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus


class Bot:
    id = 9000

    def __init__(self):
        self.sent = []
        self.fail_to = set()

    async def send_message(self, chat_id, text, **kwargs):
        if chat_id in self.fail_to:
            raise TelegramError("unavailable")
        result = NS(message_id=len(self.sent) + 101, chat_id=chat_id, text=text, kwargs=kwargs)
        self.sent.append(result)
        return result


class DB:
    def __init__(self, status="approved"):
        self.candidate = Candidate(
            id=1, user_id=111, user_chat_id=777, target_chat_id=-1001,
            username="test", first_name="Test", status=status, stage=4,
        )
        self.tickets = []
        self.entries = []
        self.admin_map = {}

    async def get_candidate(self, candidate_id):
        return self.candidate if candidate_id == 1 else None

    async def get_active_interview(self, user_id):
        return self.candidate if self.candidate.status == "interview" else None

    async def get_latest_candidate_for_user(self, user_id):
        return self.candidate if user_id == 111 else None

    async def count_recent_support_messages(self, user_id, since):
        return sum(i.direction == "candidate_to_admin" for i in self.entries)

    async def get_or_create_open_support_ticket(self, candidate, user_chat_id):
        existing = next((x for x in self.tickets if x.status == "open"), None)
        if existing:
            existing.user_chat_id = user_chat_id
            return existing
        obj = NS(id=len(self.tickets) + 1, candidate_id=candidate.id, user_id=candidate.user_id,
                 user_chat_id=user_chat_id, status="open")
        self.tickets.append(obj)
        return obj

    async def store_candidate_support_message(self, ticket_id, text, private_chat_id, message_id):
        if any(getattr(i, "source_private_message_id", None) == message_id for i in self.entries):
            return None
        entry = NS(id=len(self.entries) + 1, ticket_id=ticket_id, direction="candidate_to_admin",
                   text=text, source_private_message_id=message_id, delivery_error=None,
                   delivery_status="pending", delivery_attempts=0,
                   next_attempt_at=None, bot_admin_message_id=None)
        self.entries.append(entry)
        return entry

    async def list_due_support_message_ids(self, now, *, limit=20):
        return [i.id for i in self.entries
                if i.delivery_status in ("pending", "retry", "sending")
                and (i.next_attempt_at is None or i.next_attempt_at <= now)][:limit]

    async def claim_support_delivery(self, message_id, now, **kwargs):
        entry = self.entries[message_id - 1]
        if entry.delivery_status not in ("pending", "retry", "sending"):
            return None
        if entry.direction == "ticket_closed" and any(
            other.ticket_id == entry.ticket_id and other.id < entry.id
            and other.direction == "admin_to_candidate"
            and other.delivery_status in ("pending", "retry", "sending", "failed")
            for other in self.entries
        ):
            return None
        if entry.next_attempt_at is not None and entry.next_attempt_at > now:
            return None
        entry.delivery_attempts += 1
        entry.delivery_status = "sending"
        entry.next_attempt_at = now + __import__("datetime").timedelta(seconds=300)
        return entry

    async def mark_support_delivery(self, message_id, *, attempt,
                                    admin_message_id=None, private_message_id=None):
        entry = self.entries[message_id - 1]
        if entry.delivery_status != "sending" or entry.delivery_attempts != attempt:
            return False
        entry.delivery_status = "delivered"
        entry.next_attempt_at = None
        entry.delivery_error = None
        if admin_message_id is not None:
            entry.bot_admin_message_id = admin_message_id
            if entry.direction == "candidate_to_admin":
                self.admin_map[admin_message_id] = self.tickets[entry.ticket_id - 1]
        if private_message_id is not None:
            entry.delivered_private_message_id = private_message_id
        return True

    async def fail_support_delivery(self, message_id, *, attempt, error, now, **kwargs):
        entry = self.entries[message_id-1]
        entry.delivery_error = error
        entry.delivery_status = "retry" if attempt < 8 else "failed"
        entry.next_attempt_at = (now + __import__("datetime").timedelta(seconds=30)
                                 if attempt < 8 else None)
        if attempt >= 8 and entry.direction != "delivery_alert":
            existing = next((i for i in self.entries
                             if i.direction == "delivery_alert" and i.related_message_id == message_id), None)
            if existing is None:
                self.entries.append(NS(
                    id=len(self.entries) + 1, ticket_id=entry.ticket_id,
                    direction="delivery_alert", text="", related_message_id=message_id,
                    delivery_error=None, delivery_status="pending", delivery_attempts=0,
                    next_attempt_at=None,
                ))
            else:
                existing.delivery_status = "pending"
                existing.delivery_attempts = 0
                existing.delivery_error = None
                existing.next_attempt_at = None
        return entry.delivery_status

    async def cancel_support_alert(self, message_id, *, attempt):
        entry = self.entries[message_id - 1]
        if entry.delivery_status == "sending" and entry.delivery_attempts == attempt:
            entry.delivery_status = "cancelled"
            entry.next_attempt_at = None
            return True
        return False

    async def recover_support_card_mapping(self, ticket_id, message_id, admin_message_id):
        if not 0 < message_id <= len(self.entries):
            return None
        entry = self.entries[message_id - 1]
        if entry.ticket_id != ticket_id or entry.direction != "candidate_to_admin":
            return None
        if entry.bot_admin_message_id is None:
            entry.bot_admin_message_id = admin_message_id
            entry.delivery_status = "delivered"
            entry.next_attempt_at = None
            self.admin_map[admin_message_id] = self.tickets[ticket_id - 1]
        return self.tickets[ticket_id - 1]

    async def get_ticket_for_admin_reply(self, bot_admin_message_id):
        return self.admin_map.get(bot_admin_message_id)

    async def store_admin_support_message(self, ticket_id, text, admin_user_id, message_id):
        if any(getattr(i, "source_admin_message_id", None) == message_id for i in self.entries):
            return None
        entry = NS(id=len(self.entries) + 1, ticket_id=ticket_id,
                   direction="admin_to_candidate", text=text,
                   source_admin_message_id=message_id, delivery_error=None,
                   delivery_status="pending", delivery_attempts=0, next_attempt_at=None,
                   delivered_private_message_id=None)
        self.entries.append(entry)
        return entry

    async def close_support_ticket(self, ticket_id, user_id):
        if not 0 < ticket_id <= len(self.tickets):
            return False
        item = self.tickets[ticket_id - 1]
        if item.status == "closed":
            return False
        item.status = "closed"
        self.entries.append(NS(
            id=len(self.entries)+1, ticket_id=ticket_id, direction="ticket_closed",
            text=texts.SUPPORT_CLOSED, admin_user_id=user_id, delivery_error=None,
            delivery_status="pending", delivery_attempts=0, next_attempt_at=None,
        ))
        return True

    async def get_ticket_close_notification_id(self, ticket_id):
        for i in reversed(self.entries):
            if i.ticket_id == ticket_id and i.direction == "ticket_closed":
                return i.id
        return None

    async def get_support_message(self, message_id):
        return self.entries[message_id-1] if 0 < message_id <= len(self.entries) else None

    async def retry_failed_support_message(self, message_id):
        i = await self.get_support_message(message_id)
        if i is None or i.direction == "delivery_alert" or i.delivery_status != "failed":
            return False
        for alert in self.entries:
            if (alert.direction == "delivery_alert" and alert.related_message_id == message_id
                    and alert.delivery_status in ("pending", "retry")):
                alert.delivery_status = "cancelled"
                alert.next_attempt_at = None
        i.delivery_status = "pending"
        i.delivery_attempts = 0
        i.next_attempt_at = None
        i.delivery_error = None
        return True

    async def list_failed_support_messages(self, *, limit=15):
        return [i for i in reversed(self.entries) if i.delivery_status == "failed"][:limit]

    async def get_support_ticket(self, ticket_id):
        return self.tickets[ticket_id - 1]


def cfg(rate=5):
    return Settings(
        bot_token="test", target_chat_id=-1001, admin_chat_id=-2002,
        admin_user_ids={42}, database_url="sqlite+aiosqlite:///:memory:",
        llm_base_url="https://invalid.test", llm_api_key="test", llm_model="test",
        llm_timeout_seconds=1, auto_reject_spam_score=85,
        auto_reject_core_score_max=15, auto_reject_min_ai_confidence=80,
        max_answer_length=2500, support_rate_limit_count=rate,
    )


def private(n, text, user_id=111):
    return NS(
        effective_chat=NS(id=777, type="private"), effective_user=NS(id=user_id),
        effective_message=NS(message_id=n, text=text, reply_text=AsyncMock()),
    )


def reply(n, reply_to, text, sender_id=42, origin_id=9000):
    return NS(
        effective_chat=NS(id=-2002),
        effective_user=NS(id=sender_id, full_name="Администратор"),
        effective_message=NS(message_id=n, text=text,
                             reply_to_message=NS(message_id=reply_to,
                                                 from_user=NS(id=origin_id)),
                             reply_text=AsyncMock()),
    )


def close(ticket_id, sender_id=42):
    return NS(
        callback_query=NS(data=f"support:close:{ticket_id}", message=NS(chat=NS(id=-2002)),
                          answer=AsyncMock(), edit_message_reply_markup=AsyncMock()),
        effective_user=NS(id=sender_id, full_name="Администратор"),
    )


def setup(status="approved", rate=5):
    db, bot = DB(status), Bot()
    handlers = GatekeeperHandlers(settings=cfg(rate), db=db, scorer=object())
    return db, bot, handlers, NS(bot=bot)


def test_candidate_to_admin_to_candidate_and_same_ticket():
    async def run():
        db, bot, h, ctx = setup()
        one = private(10, "Не могу открыть группу")
        await h.on_private_message(one, ctx)
        assert "Обращение #1" in bot.sent[-1].text
        assert one.effective_message.reply_text.await_args.args == (texts.SUPPORT_ACK,)
        card_id = bot.sent[-1].message_id
        answer = reply(20, card_id, "Попробуйте ещё раз")
        await h.support.on_admin_reply(answer, ctx)
        assert bot.sent[-1].chat_id == 777
        assert bot.sent[-1].text == "💬 Ответ администрации Solidus:\n\nПопробуйте ещё раз"
        assert "Администратор" not in bot.sent[-1].text
        await h.on_private_message(private(11, "Не работает"), ctx)
        assert len(db.tickets) == 1 and len(db.entries) == 3
        assert db.candidate.status == CandidateStatus.APPROVED.value
    asyncio.run(run())


def test_no_forward_unknown_user_or_during_interview():
    async def run():
        db, bot, h, ctx = setup(status="interview")
        await h.support.on_candidate_message(private(10, "Первый ответ"), ctx)
        await h.support.on_candidate_message(private(11, "Мимо", user_id=999), ctx)
        assert not db.tickets and not bot.sent
    asyncio.run(run())


def test_unauthorized_admin_and_non_bot_reply_are_ignored():
    async def run():
        db, bot, h, ctx = setup(status="rejected")
        await h.on_private_message(private(10, "Почему отказали?"), ctx)
        original = bot.sent[-1].message_id
        await h.support.on_admin_reply(reply(21, original, "x", sender_id=13), ctx)
        await h.support.on_admin_reply(reply(22, original, "x", origin_id=4), ctx)
        await h.support.on_admin_reply(reply(23, original + 500, "x"), ctx)
        assert len(bot.sent) == 1
        assert len(db.entries) == 1
        blocked = close(1, 13)
        await h.support.on_close_ticket(blocked, ctx)
        assert blocked.callback_query.answer.await_args.kwargs["show_alert"] is True
        assert db.tickets[0].status == "open"
    asyncio.run(run())


def test_closed_ticket_blocks_reply_and_reopening_creates_new_number():
    async def run():
        db, bot, h, ctx = setup()
        await h.on_private_message(private(10, "Вопрос"), ctx)
        original = bot.sent[-1].message_id
        await h.support.on_close_ticket(close(1), ctx)
        assert db.tickets[0].status == "closed"
        count = len(bot.sent)
        old_reply = reply(20, original, "Запоздалый ответ")
        await h.support.on_admin_reply(old_reply, ctx)
        assert len(bot.sent) == count
        await h.on_private_message(private(11, "Новый вопрос"), ctx)
        assert "Обращение #2" in bot.sent[-1].text
    asyncio.run(run())


def test_rate_limit_and_duplicate_delivery():
    async def run():
        db, bot, h, ctx = setup(rate=2)
        one = private(10, "Первое")
        await h.on_private_message(one, ctx)
        await h.on_private_message(one, ctx)
        await h.on_private_message(private(11, "Второе"), ctx)
        third = private(12, "Третье")
        await h.on_private_message(third, ctx)
        assert third.effective_message.reply_text.await_args.args == (texts.SUPPORT_RATE_LIMIT,)
        assert len(db.entries) == 2
        assert len(bot.sent) == 2
    asyncio.run(run())


def test_telegram_errors_persist_message_and_do_not_claim_success():
    async def run():
        db, bot, h, ctx = setup()
        bot.fail_to.add(-2002)
        one = private(10, "Вопрос")
        await h.on_private_message(one, ctx)
        assert len(db.entries) == 1 and db.entries[0].delivery_error == "TelegramError"
        assert "повторена автоматически" in one.effective_message.reply_text.await_args.args[0]
        bot.fail_to.clear()
        await h.on_private_message(private(11, "Новая попытка"), ctx)
        original = bot.sent[-1].message_id
        bot.fail_to.add(777)
        response = reply(20, original, "Ответ")
        await h.support.on_admin_reply(response, ctx)
        assert db.entries[-1].direction == "admin_to_candidate"
        assert db.entries[-1].delivery_error == "TelegramError"
        assert "повторит отправку" in response.effective_message.reply_text.await_args.args[0]
    asyncio.run(run())
