from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatType
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from . import texts
from .config import Settings
from .db import Database
from .keyboards import support_close_keyboard, support_retry_keyboard
from .models import Candidate, CandidateStatus

logger = logging.getLogger(__name__)

STATUS_LABELS = {
    CandidateStatus.INTERVIEW.value: "Анкетирование",
    CandidateStatus.SCORING.value: "AI-проверка",
    CandidateStatus.PENDING_ADMIN.value: "На рассмотрении",
    CandidateStatus.APPROVED.value: "Одобрена",
    CandidateStatus.REJECTED.value: "Отклонена администратором",
    CandidateStatus.AUTO_REJECTED.value: "Автоматически отклонена",
    CandidateStatus.EXPIRED.value: "Анкета просрочена",
    CandidateStatus.SUPERSEDED.value: "Заменена новой заявкой",
    CandidateStatus.ERROR.value: "Ошибка обработки",
}


class SupportHandlers:
    """Bot-mediated text support using admin-chat replies to bot-authored messages."""

    def __init__(
        self, *, settings: Settings, db: Database,
        is_allowed_admin: Callable[[ContextTypes.DEFAULT_TYPE, int], Awaitable[bool]],
    ):
        self.settings = settings
        self.db = db
        self.is_allowed_admin = is_allowed_admin

    async def on_candidate_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (
            update.effective_chat is None
            or update.effective_chat.type != ChatType.PRIVATE
            or update.effective_user is None
            or update.effective_message is None
            or not update.effective_message.text
        ):
            return

        # Unknown users cannot turn this into an open relay to moderators.
        candidate = await self.db.get_latest_candidate_for_user(update.effective_user.id)
        if candidate is None:
            return
        # The regular interview handler owns every answer until the 3rd answer.
        if candidate.status == CandidateStatus.INTERVIEW.value:
            return
        answer = update.effective_message.text.strip()
        if not answer:
            return
        if len(answer) > self.settings.max_answer_length:
            await update.effective_message.reply_text(texts.TOO_LONG)
            return

        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.settings.support_rate_limit_seconds
        )
        if await self.db.count_recent_support_messages(candidate.user_id, cutoff) >= (
            self.settings.support_rate_limit_count
        ):
            await update.effective_message.reply_text(texts.SUPPORT_RATE_LIMIT)
            return

        ticket = await self.db.get_or_create_open_support_ticket(
            candidate, update.effective_chat.id
        )
        entry = await self.db.store_candidate_support_message(
            ticket.id, answer, update.effective_chat.id, update.effective_message.message_id
        )
        if entry is None:
            return  # Duplicate Telegram update; never forward it twice.

        delivered = await self._deliver_message(context.bot, entry.id)
        if delivered:
            await update.effective_message.reply_text(texts.SUPPORT_ACK)
        else:
            await update.effective_message.reply_text(
                "⚠️ Сообщение сохранено. Доставка администраторам будет "
                "повторена автоматически; повторно отправлять текст не нужно."
            )

    async def _deliver_message(self, bot, message_id: int) -> bool:
        """Lease, send, and record a support message without exposing admin IDs.

        Telegram Bot API does not provide a send idempotency key. If a process
        crashes after send_message but before DB commit, retries can create an
        additional Telegram message. The immutable message marker allows replies
        to *either* card to resolve to the original ticket.
        """
        entry = await self.db.claim_support_delivery(
            message_id, datetime.now(timezone.utc)
        )
        if entry is None:
            return False
        ticket = await self.db.get_support_ticket(entry.ticket_id)
        if ticket is None:
            raise LookupError(f"Support ticket missing for message={message_id}")
        try:
            if entry.direction == "candidate_to_admin":
                candidate = await self.db.get_candidate(ticket.candidate_id)
                if candidate is None:
                    raise LookupError(f"Candidate missing for ticket={ticket.id}")
                name = (candidate.first_name or candidate.username or str(candidate.user_id))[:140]
                username = f"@{candidate.username[:100]}" if candidate.username else "—"
                status = STATUS_LABELS.get(candidate.status, candidate.status)
                # First two lines are immutable recovery markers; only bot-authored
                # Telegram cards in ADMIN_CHAT_ID may be used to recover a mapping.
                content = (
                    f"📩 Обращение #{ticket.id}\n"
                    f"🆔 Сообщение #{entry.id}\n"
                    f"Кандидат: {name}\n"
                    f"Telegram: {username}\n"
                    f"Telegram ID: {candidate.user_id}\n"
                    f"Статус заявки: {status}\n\n"
                    f"Сообщение:\n{entry.text}\n\n"
                    "Чтобы ответить кандидату, нажмите «Ответить» на ЭТО сообщение бота."
                )
                sent = await bot.send_message(
                    chat_id=self.settings.admin_chat_id,
                    text=content,
                    reply_markup=support_close_keyboard(ticket.id),
                )
                success = await self.db.mark_support_delivery(
                    entry.id, attempt=entry.delivery_attempts,
                    admin_message_id=sent.message_id,
                )
            elif entry.direction in ("admin_to_candidate", "ticket_closed"):
                text = (
                    f"{texts.SUPPORT_REPLY_PREFIX}:\n\n{entry.text}"
                    if entry.direction == "admin_to_candidate" else (entry.text or texts.SUPPORT_CLOSED)
                )
                sent = await bot.send_message(
                    chat_id=ticket.user_chat_id,
                    text=text,
                )
                success = await self.db.mark_support_delivery(
                    entry.id, attempt=entry.delivery_attempts,
                    private_message_id=sent.message_id,
                )
            else:
                # A terminal failure produces an operator alert in the same
                # persistent outbox, not an ephemeral best-effort notification.
                original = await self.db.get_support_message(entry.related_message_id)
                if original is None or original.ticket_id != ticket.id:
                    raise LookupError(f"Missing failed support message={entry.related_message_id}")
                if original.delivery_status != "failed":
                    # This failure was already fixed before its alert was sent.
                    await self.db.cancel_support_alert(entry.id, attempt=entry.delivery_attempts)
                    return False
                direction_names = {
                    "candidate_to_admin": "сообщение кандидата модераторам",
                    "admin_to_candidate": "ответ администрации кандидату",
                    "ticket_closed": "уведомление о закрытии обращения",
                }
                purpose = direction_names.get(original.direction, original.direction)
                content = (
                    f"🚨 Не удалось доставить {purpose}\n"
                    f"Обращение #{ticket.id} · сообщение #{original.id}\n"
                    f"Попыток: {original.delivery_attempts}\n"
                    f"Причина: {original.delivery_error or 'Неизвестная ошибка'}\n"
                    f"Статус: требуется вмешательство администратора."
                )
                sent = await bot.send_message(
                    chat_id=self.settings.admin_chat_id,
                    text=content,
                    reply_markup=support_retry_keyboard(original.id),
                )
                success = await self.db.mark_support_delivery(
                    entry.id, attempt=entry.delivery_attempts,
                    admin_message_id=sent.message_id,
                )
            return success
        except TelegramError as exc:
            logger.warning(
                "Support delivery error message=%s attempt=%s type=%s",
                entry.id, entry.delivery_attempts, type(exc).__name__,
            )
            status = await self.db.fail_support_delivery(
                entry.id, attempt=entry.delivery_attempts,
                error=type(exc).__name__, now=datetime.now(timezone.utc),
            )
            if status == "failed":
                logger.error("Support delivery exhausted retries message=%s", entry.id)
            return False

    async def retry_due_messages(self, bot, *, batch_size: int = 20) -> int:
        """Run in the existing maintenance loop; leases survive bot restarts."""
        ids = await self.db.list_due_support_message_ids(
            datetime.now(timezone.utc), limit=batch_size
        )
        delivered = 0
        for message_id in ids:
            try:
                if await self._deliver_message(bot, message_id):
                    delivered += 1
            except Exception:
                # DB failures must not stop retrying other pending deliveries.
                logger.exception("Unable to process support delivery message=%s", message_id)
        return delivered

    async def on_admin_reply(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        if (
            update.effective_chat is None
            or update.effective_chat.id != self.settings.admin_chat_id
            or update.effective_user is None
            or msg is None
            or msg.reply_to_message is None
            or not msg.text
        ):
            return

        # Only replies to bot's support card. No arbitrary admin-group chatter.
        original = msg.reply_to_message
        if original.from_user is None or original.from_user.id != context.bot.id:
            return
        if not await self.is_allowed_admin(context, update.effective_user.id):
            return
        ticket = await self.db.get_ticket_for_admin_reply(original.message_id)
        if ticket is None and isinstance(getattr(original, "text", None), str):
            # The database might have been unavailable just after Telegram
            # accepted the card. Recover using the bot-authored marker.
            marker = re.match(
                r"^📩 Обращение #(\d+)\n🆔 Сообщение #(\d+)\n",
                original.text,
            )
            if marker:
                ticket = await self.db.recover_support_card_mapping(
                    int(marker.group(1)), int(marker.group(2)), original.message_id
                )
        if ticket is None:
            return
        if ticket.status != "open":
            await msg.reply_text("Обращение закрыто. Попросите кандидата написать заново.")
            return

        response = msg.text.strip()
        if not response:
            return
        if len(response) > self.settings.max_answer_length:
            await msg.reply_text(texts.TOO_LONG)
            return

        entry = await self.db.store_admin_support_message(
            ticket.id, response, update.effective_user.id, msg.message_id
        )
        if entry is None:
            return
        delivered = await self._deliver_message(context.bot, entry.id)
        if delivered:
            await msg.reply_text(f"✅ Ответ по обращению #{ticket.id} доставлен кандидату.")
        else:
            await msg.reply_text(
                "⚠️ Ответ сохранён в БД, но пока не доставлен. "
                "Бот повторит отправку автоматически."
            )

    async def on_retry_delivery(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """An authorized moderator can requeue a terminally failed delivery."""
        query = update.callback_query
        if (
            query is None or query.data is None or query.message is None
            or query.message.chat.id != self.settings.admin_chat_id
            or update.effective_user is None
        ):
            return
        if not await self.is_allowed_admin(context, update.effective_user.id):
            await query.answer("Недостаточно прав", show_alert=True)
            return
        try:
            message_id = int(query.data.split(":", 2)[2])
        except (ValueError, IndexError):
            await query.answer("Некорректный номер сообщения", show_alert=True)
            return
        if not await self.db.retry_failed_support_message(message_id):
            await query.answer("Сообщение уже повторяется или не найдено", show_alert=True)
            return
        await query.answer("Повторная доставка запущена")
        delivered = await self._deliver_message(context.bot, message_id)
        if delivered:
            try:
                await context.bot.send_message(
                    self.settings.admin_chat_id,
                    f"✅ Сообщение #{message_id} доставлено после повторной попытки.",
                )
            except TelegramError:
                logger.warning("Failed to send retry receipt for message=%s", message_id)
        else:
            # Message is again scheduled via persistent queue; no false success.
            logger.info("Requeued support delivery message=%s", message_id)

    async def on_failed_messages(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Backstop if Telegram could not deliver the failure alert itself."""
        if (
            update.effective_chat is None
            or update.effective_chat.id != self.settings.admin_chat_id
            or update.effective_user is None or update.effective_message is None
        ):
            return
        if not await self.is_allowed_admin(context, update.effective_user.id):
            return
        failed = await self.db.list_failed_support_messages(limit=15)
        if not failed:
            await update.effective_message.reply_text("✅ Недоставленных сообщений нет.")
            return
        rows = ["🚨 Недоставленные сообщения (последние 15):"]
        for m in failed:
            detail = (
                f"(оповещение о сообщении #{m.related_message_id})"
                if m.direction == "delivery_alert" else ""
            )
            rows.append(
                f"• #{m.id}, обращение #{m.ticket_id}, тип {m.direction} {detail}, "
                f"попыток {m.delivery_attempts}, ошибка {m.delivery_error or 'неизвестна'}"
            )
        rows.append("Для повторной отправки используйте кнопку под оповещением. "
                    "Если оповещение тоже не доставлено, используйте "
                    "/support_retry <ID сообщения>.")
        await update.effective_message.reply_text("\n".join(rows))

    async def on_retry_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Fallback to retry a failed entry when its alert is undeliverable."""
        if (
            update.effective_chat is None
            or update.effective_chat.id != self.settings.admin_chat_id
            or update.effective_user is None or update.effective_message is None
        ):
            return
        if not await self.is_allowed_admin(context, update.effective_user.id):
            return
        args = getattr(context, "args", None) or []
        if len(args) != 1 or not args[0].isdigit():
            await update.effective_message.reply_text("Формат: /support_retry <ID сообщения>")
            return
        message_id = int(args[0])
        if not await self.db.retry_failed_support_message(message_id):
            await update.effective_message.reply_text("Сообщение не найдено или уже обрабатывается.")
            return
        delivered = await self._deliver_message(context.bot, message_id)
        await update.effective_message.reply_text(
            "✅ Сообщение доставлено." if delivered
            else "🔄 Доставка поставлена на повторную попытку."
        )

    async def on_close_ticket(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if (
            query is None
            or query.data is None
            or query.message is None
            or query.message.chat.id != self.settings.admin_chat_id
            or update.effective_user is None
        ):
            return
        if not await self.is_allowed_admin(context, update.effective_user.id):
            await query.answer("Недостаточно прав", show_alert=True)
            return
        try:
            ticket_id = int(query.data.split(":", 2)[2])
        except (ValueError, IndexError):
            await query.answer("Некорректный номер обращения", show_alert=True)
            return
        closed = await self.db.close_support_ticket(ticket_id, update.effective_user.id)
        if not closed:
            await query.answer("Обращение уже закрыто или не найдено", show_alert=True)
            return
        await query.answer("Обращение закрыто")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except TelegramError:
            logger.warning("Unable to remove close button for ticket=%s", ticket_id)
        # The close notification was inserted atomically with the ticket close.
        # Send via the same durable queue; a Telegram failure will be retried.
        notification_id = await self.db.get_ticket_close_notification_id(ticket_id)
        notified = (
            await self._deliver_message(context.bot, notification_id)
            if notification_id is not None else False
        )
        try:
            await context.bot.send_message(
                self.settings.admin_chat_id,
                f"✅ Обращение #{ticket_id} закрыто администратором "
                f"{update.effective_user.full_name[:100]}. "
                + ("Уведомление кандидату доставлено." if notified
                   else "Уведомление кандидату находится в очереди доставки."),
            )
        except TelegramError:
            logger.warning("Unable to send close notice for ticket=%s", ticket_id)
