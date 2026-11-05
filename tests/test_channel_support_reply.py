"""Channel-safe support replies: channel button -> /start deep link -> admin DM ForceReply.

These offline unit cases intentionally do not depend on Telegram network or LLM.
DB integration is covered separately with real SQLite (when aiosqlite is present).
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from app.config import Settings
from app.handlers import GatekeeperHandlers
from app.models import CandidateStatus
from test_support_unit import Bot, DB, cfg, private


class ChannelBot(Bot):
    async def get_me(self):
        return NS(id=self.id, username="NewSolidusGatekeeperBot")


class ChannelDB(DB):
    async def get_support_ticket(self, ticket_id):
        if ticket_id < 1 or ticket_id > len(self.tickets):
            return None
        return self.tickets[ticket_id - 1]

    async def store_admin_support_message(
        self, ticket_id, text, admin_user_id, message_id, *, private_chat_id=None,
    ):
        if private_chat_id is not None:
            if private_chat_id != admin_user_id:
                raise ValueError("Admin DM chat must match actor")
            if any(getattr(entry, "dm_key", None) == (private_chat_id, message_id)
                   for entry in self.entries):
                return None
            if (await self.get_support_ticket(ticket_id)).status != "open":
                return None
            # FakeDB's legacy group-only method keys on the bare message ID.
            # Call with a unique synthetic key to model the real DM composite.
            result = await super().store_admin_support_message(
                ticket_id, text, admin_user_id, private_chat_id * 100000 + message_id,
            )
            if result is not None:
                result.dm_key = (private_chat_id, message_id)
            return result
        return await super().store_admin_support_message(
            ticket_id, text, admin_user_id, message_id,
        )


def setup_channel():
    bot, db = ChannelBot(), ChannelDB()
    settings = replace(cfg(), admin_chat_id=-1003852580363)
    handlers = GatekeeperHandlers(settings=settings, db=db, scorer=object())
    return bot, db, handlers, NS(bot=bot, args=[])


def admin_start(ticket_id, user_id=42):
    msg = NS(message_id=10, text=f"/start sreply_{ticket_id}", reply_text=AsyncMock())
    return NS(
        effective_chat=NS(type="private", id=user_id),
        effective_user=NS(id=user_id), effective_message=msg,
    )


def admin_reply(prompt, content="Вот ответ", user_id=42, message_id=44,
                origin_bot_id=9000):
    msg = NS(
        message_id=message_id, text=content, reply_text=AsyncMock(),
        reply_to_message=NS(text=prompt, from_user=NS(id=origin_bot_id)),
    )
    return NS(
        effective_chat=NS(type="private", id=user_id),
        effective_user=NS(id=user_id), effective_message=msg,
    )


def test_channel_card_deep_link_and_dm_reply_delivered_without_group():
    async def run():
        bot, db, handlers, ctx = setup_channel()
        await handlers.on_private_message(private(11, "Как вступить?"), ctx)
        card = bot.sent[-1]
        assert card.chat_id == -1003852580363
        keyboard = card.kwargs["reply_markup"].inline_keyboard[0]
        assert keyboard[0].text == "✉️ Ответить"
        assert keyboard[0].url == "https://t.me/NewSolidusGatekeeperBot?start=sreply_1"
        assert keyboard[1].callback_data == "support:close:1"
        assert "кнопку" in card.text and "Обращение #1" in card.text
        start = admin_start(1)
        ctx.args = ["sreply_1"]
        await handlers.on_start(start, ctx)
        prompt = start.effective_message.reply_text.await_args.args[0]
        assert prompt.startswith("✉️ Ответ на обращение #1\n")
        assert start.effective_message.reply_text.await_args.kwargs["reply_markup"].selective is False
        incoming = admin_reply(prompt, "Пожалуйста, воспользуйтесь ссылкой")
        await handlers.on_private_message(incoming, ctx)
        assert bot.sent[-1].chat_id == 777
        assert bot.sent[-1].text == (
            "💬 Ответ администрации Solidus:\n\nПожалуйста, воспользуйтесь ссылкой"
        )
        assert db.candidate.status == CandidateStatus.APPROVED.value
        assert len(db.entries) == 2
        assert incoming.effective_message.reply_text.await_args.args[0].startswith("✅")
        # Telegram may redeliver the same update; never send a duplicate reply.
        await handlers.on_private_message(incoming, ctx)
        assert len(db.entries) == 2
    asyncio.run(run())


def test_nonadmin_cannot_use_direct_link_or_forward_private_reply():
    async def run():
        bot, db, handlers, ctx = setup_channel()
        await handlers.on_private_message(private(11, "Вопрос"), ctx)
        ctx.args = ["sreply_1"]
        start = admin_start(1, user_id=13)
        await handlers.on_start(start, ctx)
        assert "Недостаточно прав" in start.effective_message.reply_text.await_args.args[0]
        forged = admin_reply("✉️ Ответ на обращение #1\nНапишите текст", user_id=13)
        await handlers.on_private_message(forged, ctx)
        assert "Недостаточно прав" in forged.effective_message.reply_text.await_args.args[0]
        assert len(db.entries) == 1
    asyncio.run(run())


def test_closed_ticket_and_non_bot_prompt_never_forwarded():
    async def run():
        bot, db, handlers, ctx = setup_channel()
        await handlers.on_private_message(private(11, "Вопрос"), ctx)
        not_bot = admin_reply("✉️ Ответ на обращение #1\nТекст", origin_bot_id=666)
        await handlers.on_private_message(not_bot, ctx)
        assert len(db.entries) == 1
        db.tickets[0].status = "closed"
        ctx.args = ["sreply_1"]
        start = admin_start(1)
        await handlers.on_start(start, ctx)
        assert "закрыто" in start.effective_message.reply_text.await_args.args[0]
        closed = admin_reply("✉️ Ответ на обращение #1\nТекст")
        await handlers.on_private_message(closed, ctx)
        assert "закрыто" in closed.effective_message.reply_text.await_args.args[0]
        assert len(db.entries) == 1
    asyncio.run(run())


def test_reply_survives_bot_restart_and_reuses_existing_outbox():
    async def run():
        bot, db, handlers, ctx = setup_channel()
        await handlers.on_private_message(private(11, "Ссылка сломана"), ctx)
        ctx.args = ["sreply_1"]
        start = admin_start(1)
        await handlers.on_start(start, ctx)
        prompt = start.effective_message.reply_text.await_args.args[0]
        # No persistent in-memory state: the exact Telegram bot message is
        # sufficient to recover context even if a new handler object is made.
        fresh = GatekeeperHandlers(settings=handlers.settings, db=db, scorer=object())
        bot.fail_to.add(777)
        reply = admin_reply(prompt, "Повторите попытку")
        await fresh.on_private_message(reply, ctx)
        assert db.entries[-1].delivery_status == "retry"
        assert "автоматически" in reply.effective_message.reply_text.await_args.args[0]
        assert db.entries[-1].text == "Повторите попытку"
        bot.fail_to.remove(777)
        db.entries[-1].next_attempt_at = None
        assert await fresh.support.retry_due_messages(bot) == 1
        assert bot.sent[-1].text.endswith("Повторите попытку")
    asyncio.run(run())


def test_two_mods_same_message_id_not_conflated_and_unknown_ticket_blocked():
    async def run():
        bot, db, handlers, ctx = setup_channel()
        # Permit two independent moderators via same allowlist mechanism.
        handlers.settings.admin_user_ids.add(43)
        await handlers.on_private_message(private(11, "Вопрос"), ctx)
        a = admin_reply("✉️ Ответ на обращение #1\nТекст", user_id=42, message_id=55,
                        content="Ответ первый")
        b = admin_reply("✉️ Ответ на обращение #1\nТекст", user_id=43, message_id=55,
                        content="Ответ второй")
        await handlers.on_private_message(a, ctx)
        await handlers.on_private_message(b, ctx)
        assert len(db.entries) == 3
        assert [x.text for x in db.entries if x.direction == 'admin_to_candidate'] == [
            'Ответ первый', 'Ответ второй',
        ]
        ctx.args = ["sreply_999"]
        invalid = admin_start(999)
        await handlers.on_start(invalid, ctx)
        assert "не найдено" in invalid.effective_message.reply_text.await_args.args[0]
    asyncio.run(run())


def test_reply_text_max_length_uses_existing_settings():
    async def run():
        bot, db, handlers, ctx = setup_channel()
        await handlers.on_private_message(private(11, "Вопрос"), ctx)
        too_long = admin_reply("✉️ Ответ на обращение #1\nТекст", content="x" * 2501)
        await handlers.on_private_message(too_long, ctx)
        assert len(db.entries) == 1
        assert "слишком длинный" in too_long.effective_message.reply_text.await_args.args[0].lower()
    asyncio.run(run())
