"""Operator escalation, manual retries, and reliable close notifications."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import texts
from app.models import Base, SupportMessage
from test_support_unit import close, private, reply, setup


def make_retry_callback(message_id: int, user_id: int = 42):
    return NS(
        callback_query=NS(
            data=f"support:retry:{message_id}",
            message=NS(chat=NS(id=-2002)),
            answer=AsyncMock(),
        ),
        effective_user=NS(id=user_id, full_name="Moderator"),
    )


def fast_forward(entry):
    entry.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)


async def exhaust(db, handler, bot, entry):
    for attempt in range(2, 9):
        fast_forward(entry)
        await handler.support.retry_due_messages(bot)
        assert entry.delivery_attempts == attempt
    assert entry.delivery_status == "failed"


def test_terminal_failure_queues_persistent_alert_and_retry_button():
    async def run():
        db, bot, h, ctx = setup()
        await h.on_private_message(private(10, "Не открывается ссылка"), ctx)
        card = bot.sent[-1]
        bot.fail_to.add(777)
        await h.support.on_admin_reply(reply(22, card.message_id, "Используйте новую ссылку"), ctx)
        original = db.entries[-1]
        await exhaust(db, h, bot, original)
        alerts = [m for m in db.entries if m.direction == "delivery_alert"]
        assert len(alerts) == 1
        assert alerts[0].related_message_id == original.id
        assert await h.support.retry_due_messages(bot) == 1
        alert = bot.sent[-1]
        assert alert.chat_id == -2002
        assert "Не удалось доставить ответ администрации" in alert.text
        assert alert.kwargs["reply_markup"].inline_keyboard[0][0].callback_data == f"support:retry:{original.id}"

        unauthorized = make_retry_callback(original.id, 77)
        await h.support.on_retry_delivery(unauthorized, ctx)
        assert unauthorized.callback_query.answer.await_args.kwargs["show_alert"] is True
        assert original.delivery_status == "failed"

        bot.fail_to.clear()
        allowed = make_retry_callback(original.id)
        await h.support.on_retry_delivery(allowed, ctx)
        assert original.delivery_status == "delivered"
        assert any(m.chat_id == 777 and "Используйте новую ссылку" in m.text for m in bot.sent)
        assert not any("Moderator" in m.text for m in bot.sent if m.chat_id == 777)
        await h.support.on_retry_delivery(allowed, ctx)
        assert allowed.callback_query.answer.await_args.kwargs.get("show_alert") is True
    asyncio.run(run())


def test_close_notice_is_queued_atomically_and_retried_without_duplicates():
    async def run():
        db, bot, h, ctx = setup()
        await h.on_private_message(private(10, "Помогите со вступлением"), ctx)
        bot.fail_to.add(777)
        await h.support.on_close_ticket(close(1), ctx)
        assert db.tickets[0].status == "closed"
        notices = [m for m in db.entries if m.direction == "ticket_closed"]
        assert len(notices) == 1
        assert notices[0].delivery_status == "retry"
        assert not any(m.chat_id == 777 for m in bot.sent)
        # Another click cannot enqueue a second closing message.
        await h.support.on_close_ticket(close(1), ctx)
        assert len([m for m in db.entries if m.direction == "ticket_closed"]) == 1
        bot.fail_to.clear()
        fast_forward(notices[0])
        assert await h.support.retry_due_messages(bot) == 1
        delivered = [m for m in bot.sent if m.chat_id == 777]
        assert len(delivered) == 1 and delivered[0].text == texts.SUPPORT_CLOSED
        assert notices[0].delivery_status == "delivered"
        assert await h.support.retry_due_messages(bot) == 0
    asyncio.run(run())


def test_terminal_close_notice_failure_alerts_moderators_and_can_be_retried():
    async def run():
        db, bot, h, ctx = setup()
        await h.on_private_message(private(10, "Вопрос"), ctx)
        bot.fail_to.add(777)
        await h.support.on_close_ticket(close(1), ctx)
        item = next(m for m in db.entries if m.direction == "ticket_closed")
        await exhaust(db, h, bot, item)
        await h.support.retry_due_messages(bot)
        assert any("уведомление о закрытии обращения" in m.text for m in bot.sent)
        bot.fail_to.clear()
        await h.support.on_retry_delivery(make_retry_callback(item.id), ctx)
        assert item.delivery_status == "delivered"
        assert any(m.text == texts.SUPPORT_CLOSED and m.chat_id == 777 for m in bot.sent)
    asyncio.run(run())


def test_reply_queued_before_close_is_delivered_first():
    async def run():
        db, bot, h, ctx = setup()
        await h.on_private_message(private(10, "Вопрос"), ctx)
        bot.fail_to.add(777)
        await h.support.on_admin_reply(reply(35, bot.sent[-1].message_id, "Сначала ответ"), ctx)
        await h.support.on_close_ticket(close(1), ctx)
        notices = [m for m in db.entries if m.direction == "ticket_closed"]
        assert len(notices) == 1 and notices[0].delivery_attempts == 0
        bot.fail_to.clear()
        previous = next(m for m in db.entries if m.direction == "admin_to_candidate")
        fast_forward(previous)
        assert await h.support.retry_due_messages(bot) >= 1
        if notices[0].delivery_status != "delivered":
            assert await h.support.retry_due_messages(bot) == 1
        delivered = [m.text for m in bot.sent if m.chat_id == 777]
        assert delivered == ["💬 Ответ администрации Solidus:\n\nСначала ответ", texts.SUPPORT_CLOSED]
    asyncio.run(run())


def test_failed_alert_is_visible_by_command_and_can_be_retried_by_command():
    async def run():
        db, bot, h, ctx = setup()
        bot.fail_to.add(-2002)
        await h.on_private_message(private(10, "Помощь"), ctx)
        original = db.entries[0]
        await exhaust(db, h, bot, original)
        alert = next(m for m in db.entries if m.direction == "delivery_alert")
        await h.support.retry_due_messages(bot)
        assert alert.delivery_status == "retry"
        for _ in range(7):
            fast_forward(alert)
            await h.support.retry_due_messages(bot)
        assert alert.delivery_status == "failed"
        admin_update = NS(
            effective_chat=NS(id=-2002),
            effective_user=NS(id=42),
            effective_message=NS(reply_text=AsyncMock()),
        )
        await h.support.on_failed_messages(admin_update, ctx)
        status = admin_update.effective_message.reply_text.await_args.args[0]
        assert f"#{original.id}" in status and f"#{alert.id}" in status
        bot.fail_to.clear()
        admin_ctx = NS(bot=bot, args=[str(original.id)])
        await h.support.on_retry_command(admin_update, admin_ctx)
        assert original.delivery_status == "delivered"
        assert bot.sent[-1].chat_id == -2002
        assert any("Обращение" in msg.text for msg in bot.sent)
    asyncio.run(run())


def test_delivery_alert_has_one_record_per_original():
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(SupportMessage(
                id=1, ticket_id=1, direction="admin_to_candidate", text="Message"))
            session.add(SupportMessage(
                id=2, ticket_id=1, direction="delivery_alert", text="", related_message_id=1))
            session.commit()
            session.add(SupportMessage(
                id=3, ticket_id=1, direction="delivery_alert", text="", related_message_id=1))
            with pytest.raises(IntegrityError):
                session.commit()
    finally:
        engine.dispose()


def test_reply_to_failure_alert_is_not_forwarded_to_candidate():
    async def run():
        db, bot, h, ctx = setup()
        bot.fail_to.add(-2002)
        await h.on_private_message(private(10, "Help"), ctx)
        original = db.entries[0]
        await exhaust(db, h, bot, original)
        bot.fail_to.clear()
        await h.support.retry_due_messages(bot)
        card_id = bot.sent[-1].message_id
        await h.support.on_admin_reply(reply(50, card_id, "Это внутренняя диагностика"), ctx)
        assert len([m for m in db.entries if m.direction == "admin_to_candidate"]) == 0
        assert not any(m.chat_id == 777 for m in bot.sent)
    asyncio.run(run())


def test_previously_enqueued_alert_is_suppressed_after_manual_retry():
    async def run():
        db, bot, h, ctx = setup()
        bot.fail_to.add(-2002)
        await h.on_private_message(private(10, "Help"), ctx)
        original = db.entries[0]
        await exhaust(db, h, bot, original)
        alert = db.entries[-1]
        assert alert.direction == "delivery_alert" and alert.delivery_status == "pending"
        bot.fail_to.clear()
        assert await db.retry_failed_support_message(original.id)
        assert alert.delivery_status == "cancelled"
        assert await h.support.retry_due_messages(bot) == 1
        assert not any("🚨" in m.text for m in bot.sent)
    asyncio.run(run())
