"""SQLite integration for channel reply source uniqueness, restart, and closure.

CI installs requirements.txt and runs these with aiosqlite; locally without
async SQLite drivers they are explicitly skipped rather than fabricated.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("aiosqlite")

from sqlalchemy import func, select
from app import texts
from app.db import Database
from app.handlers import GatekeeperHandlers
from app.models import CandidateStatus, SupportMessage, SupportTicket
from test_channel_support_reply import ChannelBot, admin_reply, admin_start
from test_support import cfg, private_msg


async def provision(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'channel_support.db'}")
    await db.init()
    candidate = await db.create_candidate(
        user_id=111, user_chat_id=777, target_chat_id=-1001,
        username="candidate", first_name="Имя",
    )
    await db.set_status(candidate.id, CandidateStatus.APPROVED)
    settings = replace(cfg(), admin_chat_id=-1003852580363, admin_user_ids={42,43})
    bot = ChannelBot()
    h = GatekeeperHandlers(settings=settings, db=db, scorer=object())
    return db, candidate, settings, bot, h, NS(bot=bot, args=[])


def test_db_persists_channel_dm_reply_and_dedups_per_admin_chat(tmp_path):
    async def run():
        db, candidate, settings, bot, h, ctx = await provision(tmp_path)
        try:
            await h.on_private_message(private_msg(10, "Не могу войти"), ctx)
            assert len(bot.sent) == 1
            assert bot.sent[0].kwargs['reply_markup'].inline_keyboard[0][0].url.endswith('sreply_1')
            start = admin_start(1)
            ctx.args = ['sreply_1']
            await h.on_start(start, ctx)
            prompt = start.effective_message.reply_text.await_args.args[0]
            # Simulate process restart: same DB, new handler instance.
            new_h = GatekeeperHandlers(settings=settings, db=db, scorer=object())
            reply_a = admin_reply(prompt, 'Ответ модератора А', user_id=42, message_id=51)
            reply_b = admin_reply(prompt, 'Ответ модератора Б', user_id=43, message_id=51)
            await new_h.on_private_message(reply_a, ctx)
            await new_h.on_private_message(reply_b, ctx)
            await new_h.on_private_message(reply_a, ctx)  # duplicate Telegram delivery
            assert [x.text for x in bot.sent if x.chat_id == 777] == [
                '💬 Ответ администрации Solidus:\n\nОтвет модератора А',
                '💬 Ответ администрации Solidus:\n\nОтвет модератора Б',
            ]
            async with db.sessions() as session:
                rows = (await session.scalars(select(SupportMessage).order_by(SupportMessage.id))).all()
                assert len(rows) == 3  # candidate + two distinct admin DMs
                assert (rows[1].source_private_chat_id, rows[1].source_private_message_id) == (42, 51)
                assert (rows[2].source_private_chat_id, rows[2].source_private_message_id) == (43, 51)
                assert rows[1].source_admin_message_id is None
                assert rows[2].source_admin_message_id is None
            assert (await db.get_candidate(candidate.id)).status == CandidateStatus.APPROVED.value
        finally:
            await db.close()
    asyncio.run(run())


def test_closed_or_missing_ticket_cannot_receive_admin_dm_reply(tmp_path):
    async def run():
        db, _, settings, bot, h, ctx = await provision(tmp_path)
        try:
            await h.on_private_message(private_msg(10, 'Вопрос'), ctx)
            assert await db.close_support_ticket(1, 42)
            start = admin_start(1)
            ctx.args = ['sreply_1']
            await h.on_start(start, ctx)
            assert 'закрыто' in start.effective_message.reply_text.await_args.args[0]
            reply = admin_reply('✉️ Ответ на обращение #1\nТекст', 'Запоздалый ответ')
            await h.on_private_message(reply, ctx)
            assert 'закрыто' in reply.effective_message.reply_text.await_args.args[0]
            async with db.sessions() as session:
                assert await session.scalar(select(func.count(SupportMessage.id)).where(
                    SupportMessage.direction == 'admin_to_candidate')) == 0
            ctx.args = ['sreply_999']
            missing = admin_start(999)
            await h.on_start(missing, ctx)
            assert 'не найдено' in missing.effective_message.reply_text.await_args.args[0]
        finally:
            await db.close()
    asyncio.run(run())
