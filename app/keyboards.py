from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def moderation_keyboard(candidate_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Одобрить", callback_data=f"candidate:approve:{candidate_id}"
                ),
                InlineKeyboardButton(
                    "❌ Отклонить", callback_data=f"candidate:reject:{candidate_id}"
                ),
            ]
        ]
    )



def support_close_keyboard(
    ticket_id: int, *, bot_username: str | None = None,
) -> InlineKeyboardMarkup:
    """Keep close callback; optionally offer channel-safe DM reply deep link.

    The deep link only selects a ticket. The DM handler ALWAYS checks the
    administrator's permissions before showing details or accepting a reply.
    """
    buttons = []
    if bot_username:
        buttons.append(InlineKeyboardButton(
            "✉️ Ответить", url=f"https://t.me/{bot_username}?start=sreply_{ticket_id}"
        ))
    buttons.append(InlineKeyboardButton(
        "✅ Закрыть обращение", callback_data=f"support:close:{ticket_id}"
    ))
    return InlineKeyboardMarkup([buttons])


def support_retry_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """Manual requeue is restricted by support handler admin authorization."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(
            "🔄 Повторить доставку", callback_data=f"support:retry:{message_id}"
        )]]
    )
