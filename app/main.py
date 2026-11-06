from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatJoinRequestHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from .config import Settings
from .db import Database
from .handlers import GatekeeperHandlers
from .scoring import AIScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Prevent Telegram bot tokens from appearing in HTTP request logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main() -> None:
    settings = Settings.from_env()
    db = Database(settings.database_url)
    scorer = AIScorer(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    handlers = GatekeeperHandlers(settings=settings, db=db, scorer=scorer)
    maintenance_task: asyncio.Task | None = None

    async def maintenance_loop(application: Application) -> None:
        while True:
            try:
                await handlers.recover_stale_scoring(application.bot)
                await handlers.recover_cooldown_decline_intents(application.bot)
                await handlers.expire_stale_interviews(application.bot)
                await handlers.support.retry_due_messages(application.bot)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Maintenance loop failed")
            await asyncio.sleep(settings.maintenance_interval_seconds)

    async def post_init(application: Application) -> None:
        nonlocal maintenance_task
        await db.init()
        # The old process is gone. Recover interrupted LLM work before polling.
        await handlers.recover_stale_scoring(application.bot, startup=True)
        await handlers.recover_cooldown_decline_intents(application.bot)
        me = await application.bot.get_me()
        logger.info("Solidus AI Gatekeeper started as @%s", me.username)
        maintenance_task = asyncio.create_task(
            maintenance_loop(application), name="gatekeeper-maintenance"
        )

    async def post_shutdown(application: Application) -> None:
        if maintenance_task is not None:
            maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await maintenance_task
        await db.close()

    application = (
        Application.builder()
        .token(settings.bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Handler structure intentionally follows Ringo's successful pattern:
    # join request -> questionnaire -> callback-based admin verdict.
    application.add_handler(ChatJoinRequestHandler(handlers.on_join_request))
    application.add_handler(
        CallbackQueryHandler(
            handlers.on_admin_decision,
            pattern=r"^candidate:(approve|reject):\d+$",
        )
    )
    application.add_handler(CommandHandler("start", handlers.on_start))
    application.add_handler(CommandHandler("status", handlers.on_status))
    application.add_handler(CommandHandler("support_failed", handlers.support.on_failed_messages))
    application.add_handler(CommandHandler("support_retry", handlers.support.on_retry_command))
    application.add_handler(
        CallbackQueryHandler(
            handlers.support.on_close_ticket,
            pattern=r"^support:close:\d+$",
        )
    )
    application.add_handler(
        CallbackQueryHandler(
            handlers.support.on_retry_delivery,
            pattern=r"^support:retry:\d+$",
        )
    )
    # Privacy mode supports replies to messages originally sent by this bot.
    application.add_handler(
        MessageHandler(
            filters.ChatType.GROUPS & filters.TEXT & filters.REPLY & ~filters.COMMAND,
            handlers.support.on_admin_reply,
        )
    )
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
            handlers.on_private_message,
        )
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )


if __name__ == "__main__":
    main()
