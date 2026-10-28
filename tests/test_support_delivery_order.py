"""Regression: a terminally failed admin reply must block a later close notice."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import texts
from app.db import Database
from app.models import Base, SupportMessage, SupportTicket
from test_support_unit import close, private, reply, setup


class _SyncSQLiteAsAsyncSession:
    """Execute the production async DB's SQL with SQLite without aiosqlite."""

    def __init__(self, engine):
        self._engine = engine
        self._session = None

    async def __aenter__(self):
        self._session = Session(self._engine, expire_on_commit=False)
        return self

    async def __aexit__(self, *_exc):
        self._session.close()

    async def execute(self, statement):
        return self._session.execute(statement)

    async def commit(self):
        self._session.commit()

    async def rollback(self):
        self._session.rollback()

    async def get(self, model, key):
        return self._session.get(model, key)


def test_sql_claim_blocks_close_on_failed_reply_until_delivered():
    """Exercise the actual Database.claim_support_delivery UPDATE, not a mock."""
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            ticket = SupportTicket(
                candidate_id=1, user_id=111, user_chat_id=777, status="closed",
            )
            session.add(ticket)
            session.flush()
            earlier = SupportMessage(
                ticket_id=ticket.id, direction="admin_to_candidate", text="Reply",
                delivery_status="failed", delivery_attempts=8,
            )
            notice = SupportMessage(
                ticket_id=ticket.id, direction="ticket_closed", text=texts.SUPPORT_CLOSED,
                delivery_status="pending", delivery_attempts=0,
            )
            session.add_all([earlier, notice])
            session.commit()
            reply_id, notice_id = earlier.id, notice.id

        db = object.__new__(Database)
        db.sessions = lambda: _SyncSQLiteAsAsyncSession(engine)
        now = datetime.now(timezone.utc)

        async def run():
            # The exhausted reply stays failed until a moderator explicitly retries.
            assert await db.claim_support_delivery(notice_id, now) is None
            assert await db.retry_failed_support_message(reply_id)
            # A retry that has not actually been delivered must still block closing.
            assert await db.claim_support_delivery(notice_id, now) is None
            claimed_reply = await db.claim_support_delivery(reply_id, now)
            assert claimed_reply is not None
            assert await db.claim_support_delivery(notice_id, now) is None
            assert await db.mark_support_delivery(
                reply_id, attempt=claimed_reply.delivery_attempts, private_message_id=501,
            )
            # Once the earlier reply was delivered, the close notice can go out.
            claimed_notice = await db.claim_support_delivery(notice_id, now)
            assert claimed_notice is not None
            assert claimed_notice.direction == "ticket_closed"

        asyncio.run(run())
        with Session(engine) as session:
            reply_row = session.get(SupportMessage, reply_id)
            notice_row = session.get(SupportMessage, notice_id)
            assert reply_row.delivery_status == "delivered"
            assert notice_row.delivery_status == "sending"
            assert notice_row.delivery_attempts == 1
    finally:
        engine.dispose()


def test_failed_reply_retried_by_moderator_precedes_close_notice():
    """End-to-end routing with fake Telegram and DB must preserve chronology."""

    async def run():
        db, bot, handlers, context = setup()
        await handlers.on_private_message(private(10, "Не работает ссылка"), context)
        card_id = bot.sent[-1].message_id
        bot.fail_to.add(777)
        await handlers.support.on_admin_reply(
            reply(20, card_id, "Ответ по вопросу"), context,
        )
        failed_reply = next(m for m in db.entries if m.direction == "admin_to_candidate")
        # Exhaust all eight delivery attempts.
        for _ in range(7):
            failed_reply.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            await handlers.support.retry_due_messages(bot)
        assert failed_reply.delivery_status == "failed"

        await handlers.support.on_close_ticket(close(1), context)
        notice = next(m for m in db.entries if m.direction == "ticket_closed")
        assert notice.delivery_status == "pending"
        assert notice.delivery_attempts == 0

        bot.fail_to.clear()
        # Terminally failed reply must NOT be bypassed by close notification.
        await handlers.support.retry_due_messages(bot)
        assert notice.delivery_status == "pending"
        assert not any(m.chat_id == 777 for m in bot.sent)

        # Once manually retried, the reply goes out before the close notice.
        assert await db.retry_failed_support_message(failed_reply.id)
        await handlers.support.retry_due_messages(bot)
        await handlers.support.retry_due_messages(bot)
        delivered = [m.text for m in bot.sent if m.chat_id == 777]
        assert delivered == [
            "💬 Ответ администрации Solidus:\n\nОтвет по вопросу",
            texts.SUPPORT_CLOSED,
        ]

    asyncio.run(run())
