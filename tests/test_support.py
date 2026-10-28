"""Local integration tests of support routing with real SQLite and a fake Telegram API."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("aiosqlite", reason="Real SQLite async integration requires project requirements.txt")

from sqlalchemy import func, select
from telegram.error import TelegramError

from app import texts
from app.config import Settings
from app.db import Database
from app.handlers import GatekeeperHandlers
from app.models import CandidateStatus, SupportMessage, SupportTicket


class Bot:
    id = 9000

    def __init__(self, fail_chats=()):
        self.sent = []
        self.fail_chats = set(fail_chats)
        self.next_id = 100

    async def send_message(self, chat_id, text, **kwargs):
        if chat_id in self.fail_chats:
            raise TelegramError("fake transport failure")
        self.next_id += 1
        message = NS(message_id=self.next_id, chat_id=chat_id, text=text, kwargs=kwargs)
        self.sent.append(message)
        return message


def cfg(rate_limit=5):
    return Settings(
        bot_token="test", target_chat_id=-1001, admin_chat_id=-2002,
        admin_user_ids={42}, database_url="sqlite+aiosqlite:///:memory:",
        llm_base_url="https://invalid.test", llm_api_key="test", llm_model="test",
        llm_timeout_seconds=1, auto_reject_spam_score=85,
        auto_reject_core_score_max=15, auto_reject_min_ai_confidence=80,
        max_answer_length=2500, support_rate_limit_count=rate_limit,
    )


def private_msg(message_id, text, user_id=111, chat_id=777):
    message = NS(message_id=message_id, text=text, reply_text=AsyncMock())
    return NS(
        effective_chat=NS(type="private", id=chat_id),
        effective_user=NS(id=user_id, full_name="Test Person"),
        effective_message=message,
    )


def admin_msg(message_id, reply_id, text, user_id=42, origin_bot_id=9000):
    message = NS(
        message_id=message_id, text=text,
        reply_to_message=NS(message_id=reply_id, from_user=NS(id=origin_bot_id)),
        reply_text=AsyncMock(),
    )
    return NS(
        effective_chat=NS(type="supergroup", id=-2002),
        effective_user=NS(id=user_id, full_name=f"Admin {user_id}"),
        effective_message=message,
    )


def close_button(ticket_id, user_id=42):
    query = NS(
        data=f"support:close:{ticket_id}", message=NS(chat=NS(id=-2002)),
        answer=AsyncMock(), edit_message_reply_markup=AsyncMock(),
    )
    return NS(callback_query=query, effective_user=NS(id=user_id, full_name="Admin"))


class NoopScorer:
    pass


async def fixture(tmp_path, *, status=CandidateStatus.APPROVED, rate_limit=5):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'support.db'}")
    await db.init()
    candidate = await db.create_candidate(
        user_id=111, user_chat_id=777, target_chat_id=-1001,
        username="test", first_name="Test",
    )
    await db.set_status(candidate.id, status)
    bot = Bot()
    handler = GatekeeperHandlers(settings=cfg(rate_limit), db=db, scorer=NoopScorer())
    return db, candidate, bot, handler


async def totals(db):
    async with db.sessions() as s:
        return (
            await s.scalar(select(func.count(SupportTicket.id))),
            await s.scalar(select(func.count(SupportMessage.id))),
        )


def test_two_way_delivery_same_ticket_and_no_status_change(tmp_path):
    async def scenario():
        db, candidate, bot, handlers = await fixture(tmp_path)
        try:
            context = NS(bot=bot)
            first = private_msg(10, "Не открывается ссылка")
            await handlers.on_private_message(first, context)
            assert first.effective_message.reply_text.await_args.args == (texts.SUPPORT_ACK,)
            assert len(bot.sent) == 1 and bot.sent[0].chat_id == -2002
            card = bot.sent[-1]
            assert "Обращение #1" in card.text
            assert "Не открывается ссылка" in card.text
            assert card.kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "support:close:1"
            reply = admin_msg(50, card.message_id, "Откройте новую ссылку")
            await handlers.support.on_admin_reply(reply, context)
            assert bot.sent[-1].chat_id == 777
            assert bot.sent[-1].text == "💬 Ответ администрации Solidus:\n\nОткройте новую ссылку"
            assert "Admin " not in bot.sent[-1].text
            assert reply.effective_message.reply_text.await_args.args[0].startswith("✅")
            new = private_msg(11, "Попробовал, не работает")
            await handlers.on_private_message(new, context)
            assert await totals(db) == (1, 3)
            assert (await db.get_candidate(candidate.id)).status == CandidateStatus.APPROVED.value
        finally:
            await db.close()
    asyncio.run(scenario())


def test_close_then_new_ticket_and_old_reply_blocked(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path, status=CandidateStatus.REJECTED)
        try:
            context = NS(bot=bot)
            await handlers.on_private_message(private_msg(10, "Почему отказ?"), context)
            old_card = bot.sent[-1]
            close = close_button(1)
            await handlers.support.on_close_ticket(close, context)
            assert close.callback_query.answer.await_count == 1
            assert (await db.get_support_ticket(1)).status == "closed"
            sent_so_far = len(bot.sent)
            old_reply = admin_msg(51, old_card.message_id, "Уже закрыто")
            await handlers.support.on_admin_reply(old_reply, context)
            assert len(bot.sent) == sent_so_far
            assert "закрыто" in old_reply.effective_message.reply_text.await_args.args[0]
            await handlers.on_private_message(private_msg(11, "Новый вопрос"), context)
            assert "Обращение #2" in bot.sent[-1].text
            assert await totals(db) == (2, 3)
        finally:
            await db.close()
    asyncio.run(scenario())


def test_unauthorized_moderator_cannot_answer_or_close(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path)
        try:
            context = NS(bot=bot)
            await handlers.on_private_message(private_msg(10, "Вопрос"), context)
            card = bot.sent[-1]
            reply = admin_msg(31, card.message_id, "Злоумышленник", user_id=77)
            await handlers.support.on_admin_reply(reply, context)
            assert len(bot.sent) == 1
            close = close_button(1, user_id=77)
            await handlers.support.on_close_ticket(close, context)
            assert (await db.get_support_ticket(1)).status == "open"
            assert close.callback_query.answer.await_args.kwargs.get("show_alert") is True
        finally:
            await db.close()
    asyncio.run(scenario())


def test_messages_not_from_bot_support_card_are_ignored(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path)
        try:
            context = NS(bot=bot)
            await handlers.on_private_message(private_msg(10, "Вопрос"), context)
            card = bot.sent[-1]
            await handlers.support.on_admin_reply(admin_msg(20, card.message_id, "x", origin_bot_id=99), context)
            await handlers.support.on_admin_reply(admin_msg(21, 1234567, "x"), context)
            assert len(bot.sent) == 1
            assert await totals(db) == (1, 1)
        finally:
            await db.close()
    asyncio.run(scenario())


def test_candidate_rate_limit_and_duplicate_update(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path, rate_limit=1)
        try:
            context = NS(bot=bot)
            first = private_msg(10, "Первое")
            await handlers.on_private_message(first, context)
            second = private_msg(11, "Второе")
            await handlers.on_private_message(second, context)
            assert second.effective_message.reply_text.await_args.args == (texts.SUPPORT_RATE_LIMIT,)
            assert len(bot.sent) == 1
            assert await totals(db) == (1, 1)
        finally:
            await db.close()
    asyncio.run(scenario())


def test_support_forward_failure_stays_persisted_without_false_ack(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path)
        bot.fail_chats.add(-2002)
        try:
            update = private_msg(10, "Ошибка открытия")
            await handlers.on_private_message(update, NS(bot=bot))
            assert "повторена автоматически" in update.effective_message.reply_text.await_args.args[0]
            assert await totals(db) == (1, 1)
            async with db.sessions() as s:
                item = await s.scalar(select(SupportMessage))
                assert item.delivery_error == "TelegramError"
                assert item.bot_admin_message_id is None
                assert item.text == "Ошибка открытия"
        finally:
            await db.close()
    asyncio.run(scenario())


def test_admin_reply_failure_is_not_claimed_delivered(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path)
        try:
            ctx = NS(bot=bot)
            await handlers.on_private_message(private_msg(10, "Текст обращения"), ctx)
            card = bot.sent[-1]
            bot.fail_chats.add(777)
            response = admin_msg(33, card.message_id, "Попробуйте снова")
            await handlers.support.on_admin_reply(response, ctx)
            assert "повторит отправку" in response.effective_message.reply_text.await_args.args[0]
            async with db.sessions() as s:
                item = await s.scalar(select(SupportMessage).where(SupportMessage.direction == "admin_to_candidate"))
                assert item.text == "Попробуйте снова"
                assert item.delivery_error == "TelegramError"
                assert item.delivered_private_message_id is None
        finally:
            await db.close()
    asyncio.run(scenario())


def test_ticket_mapping_survives_database_restart(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path, status=CandidateStatus.AUTO_REJECTED)
        ctx = NS(bot=bot)
        await handlers.on_private_message(private_msg(10, "Не согласен с оценкой"), ctx)
        bot_msg_id = bot.sent[-1].message_id
        await db.close()
        db2 = Database(f"sqlite+aiosqlite:///{tmp_path / 'support.db'}")
        await db2.init()
        try:
            handlers2 = GatekeeperHandlers(settings=cfg(), db=db2, scorer=NoopScorer())
            await handlers2.support.on_admin_reply(
                admin_msg(42, bot_msg_id, "Перепроверим"), ctx
            )
            assert bot.sent[-1].chat_id == 777
            assert "Перепроверим" in bot.sent[-1].text
            assert await totals(db2) == (1, 2)
        finally:
            await db2.close()
    asyncio.run(scenario())


def test_no_candidate_no_support_ticket(tmp_path):
    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}")
        await db.init()
        try:
            handlers = GatekeeperHandlers(settings=cfg(), db=db, scorer=NoopScorer())
            bot = Bot()
            await handlers.on_private_message(private_msg(10, "Hello"), NS(bot=bot))
            assert await totals(db) == (0, 0)
            assert not bot.sent
        finally:
            await db.close()
    asyncio.run(scenario())


def test_interview_response_is_not_forwarded(tmp_path):
    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'interview.db'}")
        await db.init()
        try:
            await db.create_candidate(user_id=111, user_chat_id=777, target_chat_id=-1001,
                                      username="test", first_name="Test")
            handlers = GatekeeperHandlers(settings=cfg(), db=db, scorer=NoopScorer())
            bot = Bot()
            update = private_msg(10, "Тбилиси")
            await handlers.on_private_message(update, NS(bot=bot))
            assert update.effective_message.reply_text.await_args.args == (texts.QUESTION_BUSINESS,)
            assert await totals(db) == (0, 0)
        finally:
            await db.close()
    asyncio.run(scenario())


def test_second_same_candidate_message_not_forwarded_twice(tmp_path):
    async def scenario():
        db, _, bot, handlers = await fixture(tmp_path)
        try:
            ctx = NS(bot=bot)
            same = private_msg(10, "Повтор")
            await handlers.on_private_message(same, ctx)
            await handlers.on_private_message(same, ctx)
            assert len(bot.sent) == 1
            assert await totals(db) == (1, 1)
        finally:
            await db.close()
    asyncio.run(scenario())
