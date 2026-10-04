"""Test-only compatibility stubs when python-telegram-bot is unavailable.

Production installs the real dependency from requirements.txt. The execution
sandbox used for repository checks may be offline, so these minimal stubs let
unit tests exercise our own routing logic without pretending to test Telegram.
"""
from __future__ import annotations

import sys
import types

try:
    import telegram  # noqa: F401
except ModuleNotFoundError:
    telegram = types.ModuleType("telegram")
    constants = types.ModuleType("telegram.constants")
    errors = types.ModuleType("telegram.error")
    ext = types.ModuleType("telegram.ext")

    class TelegramError(Exception):
        pass

    class Bot:
        pass

    class Update:
        pass

    class InlineKeyboardButton:
        def __init__(self, text: str, callback_data: str | None = None):
            self.text = text
            self.callback_data = callback_data

    class InlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    class ChatMemberStatus:
        OWNER = "owner"
        ADMINISTRATOR = "administrator"

    class ChatType:
        PRIVATE = "private"

    class ParseMode:
        HTML = "HTML"

    class ContextTypes:
        DEFAULT_TYPE = object

    telegram.Bot = Bot
    telegram.Update = Update
    telegram.InlineKeyboardButton = InlineKeyboardButton
    telegram.InlineKeyboardMarkup = InlineKeyboardMarkup
    constants.ChatMemberStatus = ChatMemberStatus
    constants.ChatType = ChatType
    constants.ParseMode = ParseMode
    errors.TelegramError = TelegramError
    ext.ContextTypes = ContextTypes

    sys.modules["telegram"] = telegram
    sys.modules["telegram.constants"] = constants
    sys.modules["telegram.error"] = errors
    sys.modules["telegram.ext"] = ext
