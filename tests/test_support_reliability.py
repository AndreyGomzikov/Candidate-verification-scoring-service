"""Failure recovery and database invariants for the support relay."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Base, SupportTicket
from test_support_unit import private, reply, setup


def test_failed_candidate_forward_is_retried_automatically():
    async def run():
        db, bot, handlers, context = setup()
        bot.fail_to.add(-2002)
        first = private(10, "Не могу вступить")
        await handlers.on_private_message(first, context)
        entry = db.entries[0]
        assert entry.delivery_status == "retry"
        assert entry.delivery_attempts == 1
        assert "повторена автоматически" in first.effective_message.reply_text.await_args.args[0]
        assert await handlers.support.retry_due_messages(bot) == 0
        bot.fail_to.clear()
        entry.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert await handlers.support.retry_due_messages(bot) == 1
        assert entry.delivery_status == "delivered"
        assert entry.bot_admin_message_id == bot.sent[-1].message_id
        assert "🆔 Сообщение #1" in bot.sent[-1].text
        assert await handlers.support.retry_due_messages(bot) == 0
    asyncio.run(run())


def test_failed_admin_reply_is_retried_without_leaking_name():
    async def run():
        db, bot, handlers, context = setup()
        await handlers.on_private_message(private(10, "Ссылка недействительна"), context)
        card_id = bot.sent[-1].message_id
        bot.fail_to.add(777)
        answer = reply(31, card_id, "Попробуйте ещё раз")
        await handlers.support.on_admin_reply(answer, context)
        assert db.entries[-1].delivery_status == "retry"
        assert "повторит отправку" in answer.effective_message.reply_text.await_args.args[0]
        bot.fail_to.clear()
        db.entries[-1].next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert await handlers.support.retry_due_messages(bot) == 1
        assert bot.sent[-1].chat_id == 777
        assert bot.sent[-1].text == "💬 Ответ администрации Solidus:\n\nПопробуйте ещё раз"
        assert await handlers.support.retry_due_messages(bot) == 0
    asyncio.run(run())


def test_reply_recovers_card_mapping_after_db_commit_failure():
    async def run():
        db, bot, handlers, context = setup()
        await handlers.on_private_message(private(10, "Помогите"), context)
        card = bot.sent[-1]
        entry = db.entries[0]
        # Reproduce: Telegram accepted the card, but database lost its ID.
        db.admin_map.clear()
        entry.bot_admin_message_id = None
        entry.delivery_status = "sending"
        entry.next_attempt_at = datetime.now(timezone.utc) + timedelta(minutes=2)
        answer = reply(31, card.message_id, "Поможем!")
        answer.effective_message.reply_to_message.text = card.text
        await handlers.support.on_admin_reply(answer, context)
        assert entry.bot_admin_message_id == card.message_id
        assert entry.delivery_status == "delivered"
        assert bot.sent[-1].chat_id == 777
        assert "Поможем!" in bot.sent[-1].text
        assert await handlers.support.retry_due_messages(bot) == 0
    asyncio.run(run())


def test_reply_does_not_recover_mapping_for_forged_or_wrong_card():
    async def run():
        db, bot, handlers, context = setup()
        await handlers.on_private_message(private(10, "Помогите"), context)
        card = bot.sent[-1]
        db.admin_map.clear()
        entry = db.entries[0]
        entry.bot_admin_message_id = None
        entry.delivery_status = "sending"
        fake = reply(31, card.message_id, "Уберите риск")
        fake.effective_message.reply_to_message.text = card.text.replace(
            "🆔 Сообщение #1", "🆔 Сообщение #99999"
        )
        await handlers.support.on_admin_reply(fake, context)
        assert len(bot.sent) == 1
        non_bot = reply(32, card.message_id, "Чужой", origin_id=321)
        non_bot.effective_message.reply_to_message.text = card.text
        await handlers.support.on_admin_reply(non_bot, context)
        assert len(bot.sent) == 1
        assert entry.bot_admin_message_id is None
        assert entry.delivery_status == "sending"
    asyncio.run(run())


def test_one_live_delivery_lease_prevents_duplicate_forward():
    async def run():
        db, bot, handlers, context = setup()
        await handlers.on_private_message(private(10, "Вопрос"), context)
        entry = db.entries[0]
        entry.delivery_status = "retry"
        entry.bot_admin_message_id = None
        entry.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        first = await db.claim_support_delivery(entry.id, datetime.now(timezone.utc))
        second = await db.claim_support_delivery(entry.id, datetime.now(timezone.utc))
        assert first is entry
        assert second is None
        assert await handlers.support.retry_due_messages(bot) == 0
    asyncio.run(run())


def test_open_ticket_is_unique_in_sqlite_and_closed_ticket_can_be_replaced():
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(SupportTicket(user_id=111, user_chat_id=777,
                                      candidate_id=1, status="open"))
            session.commit()
            session.add(SupportTicket(user_id=111, user_chat_id=777,
                                      candidate_id=1, status="open"))
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()
            ticket = session.scalar(select(SupportTicket).where(
                SupportTicket.user_id == 111, SupportTicket.status == "open"))
            ticket.status = "closed"
            session.commit()
            session.add(SupportTicket(user_id=111, user_chat_id=777,
                                      candidate_id=1, status="open"))
            session.commit()
            assert len(session.scalars(select(SupportTicket)).all()) == 2
    finally:
        engine.dispose()


def test_failed_delivery_stops_after_retry_budget():
    async def run():
        db, bot, handlers, context = setup()
        bot.fail_to.add(-2002)
        await handlers.on_private_message(private(10, "Нужна помощь"), context)
        entry = db.entries[0]
        for attempt in range(2, 9):
            entry.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            assert await handlers.support.retry_due_messages(bot) == 0
            assert entry.delivery_attempts == attempt
        assert entry.delivery_status == "failed"
        entry.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert await handlers.support.retry_due_messages(bot) == 0
        assert entry.delivery_attempts == 8
    asyncio.run(run())
