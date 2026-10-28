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



def support_close_keyboard(ticket_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(
            "✅ Закрыть обращение", callback_data=f"support:close:{ticket_id}"
        )]]
    )


def support_retry_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """Manual requeue is restricted by support handler admin authorization."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(
            "🔄 Повторить доставку", callback_data=f"support:retry:{message_id}"
        )]]
    )
