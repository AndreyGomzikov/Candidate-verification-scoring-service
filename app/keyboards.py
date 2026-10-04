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
