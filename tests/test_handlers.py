from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram.error import TelegramError

from app import texts
from app.config import Settings
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus
from app.scoring import ScoreResult


def settings() -> Settings:
    return Settings(
        bot_token="test",
        target_chat_id=-1001,
        admin_chat_id=-2002,
        admin_user_ids={42},
        database_url="sqlite+aiosqlite:///:memory:",
        llm_base_url="https://example.invalid/v1",
        llm_api_key="test",
        llm_model="test",
        llm_timeout_seconds=1,
        auto_reject_spam_score=85,
        auto_reject_core_score_max=15,
        auto_reject_min_ai_confidence=80,
        max_answer_length=2500,
    )


def candidate(**overrides) -> Candidate:
    values = dict(
        id=1,
        user_id=111,
        user_chat_id=777,
        target_chat_id=-1001,
        username="candidate",
        first_name="Test",
        status=CandidateStatus.INTERVIEW.value,
        stage=1,
        location_answer="Батуми",
        business_answer="Основатель IT-компании",
        goal_answer="Нетворкинг и партнёрства",
    )
    values.update(overrides)
    return Candidate(**values)


def score(**overrides) -> ScoreResult:
    values = dict(
        location_score=90,
        business_score=90,
        intent_score=80,
        spam_score=5,
        ai_confidence=95,
        confidence_score=85,
        summary="Подходящий кандидат.",
        risk_flags=[],
    )
    values.update(overrides)
    return ScoreResult(**values)


class NoopScorer:
    async def score(self, **kwargs):
        return score(), "{}"


class RoutingDB:
    def __init__(self, item: Candidate):
        self.item = item
        self.statuses: list[CandidateStatus] = []

    async def save_score(self, candidate_id, score_result, raw):
        return self.item

    async def set_status(self, candidate_id, status, **kwargs):
        self.statuses.append(status)
        self.item.status = status.value
        return self.item


class RejectFailBot:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []

    async def decline_chat_join_request(self, **kwargs):
        raise TelegramError("temporary failure")

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))
        return SimpleNamespace(message_id=1)


class SpamScorer:
    async def score(self, **kwargs):
        result = score(spam_score=99, ai_confidence=99, confidence_score=20)
        return result, "{}"


def test_admin_card_stays_under_telegram_limit_with_hostile_html():
    item = candidate(
        first_name="<" * 255,
        username="&" * 255,
        location_answer="<" * 2500,
        business_answer="&" * 2500,
        goal_answer='"' * 2500,
    )
    result = score(
        summary="<" * 1000,
        risk_flags=["&" * 180 for _ in range(8)],
    )
    handler = GatekeeperHandlers(settings=settings(), db=object(), scorer=NoopScorer())
    card = handler._build_admin_card(
        item,
        score=result,
        status_note="<" * 500,
    )
    assert len(card) <= 3900
    assert "&lt;" in card


def test_candidate_notifications_use_user_chat_id_not_user_id():
    item = candidate(user_id=111, user_chat_id=999999)
    bot = RejectFailBot()
    handler = GatekeeperHandlers(settings=settings(), db=object(), scorer=NoopScorer())

    asyncio.run(handler._send_candidate_message(bot, item, "hello"))

    assert bot.sent == [(999999, "hello")]


def test_failed_auto_decline_does_not_mark_candidate_rejected():
    item = candidate(status=CandidateStatus.SCORING.value, stage=4)
    db = RoutingDB(item)
    bot = RejectFailBot()
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=SpamScorer())

    asyncio.run(handler._score_and_route(bot, item))

    assert db.statuses[-1] == CandidateStatus.PENDING_ADMIN
    assert CandidateStatus.AUTO_REJECTED not in db.statuses
    candidate_texts = [text for chat_id, text in bot.sent if chat_id == item.user_chat_id]
    assert texts.REJECTED not in candidate_texts
    assert texts.TECHNICAL_REVIEW in candidate_texts


def test_admin_api_failure_keeps_pending_status_and_buttons():
    item = candidate(status=CandidateStatus.PENDING_ADMIN.value, stage=4)
    db = SimpleNamespace(
        get_candidate=AsyncMock(return_value=item),
        set_status=AsyncMock(),
    )
    bot = SimpleNamespace(
        approve_chat_join_request=AsyncMock(side_effect=TelegramError("request gone")),
        decline_chat_join_request=AsyncMock(),
        send_message=AsyncMock(),
    )
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    query = SimpleNamespace(
        data="candidate:approve:1",
        message=SimpleNamespace(chat=SimpleNamespace(id=-2002)),
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=42, full_name="Admin"),
    )
    context = SimpleNamespace(bot=bot)

    asyncio.run(handler.on_admin_decision(update, context))

    db.set_status.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()
    assert query.answer.await_args.kwargs.get("show_alert") is True


def test_initial_dm_failure_is_not_automatic_decline():
    item = candidate()

    class DB:
        def __init__(self):
            self.last_status = None

        async def get_active_candidate_for_chat(self, *args):
            return None

        async def get_recent_rejected_candidate(self, *args):
            return None

        async def create_candidate(self, **kwargs):
            return item

        async def set_status(self, candidate_id, status, **kwargs):
            self.last_status = status
            item.status = status.value
            return item

    db = DB()

    class Bot:
        def __init__(self):
            self.decline_calls = 0
            self.admin_messages = []

        async def send_message(self, chat_id, text, **kwargs):
            if chat_id == item.user_chat_id:
                raise TelegramError("temporary send failure")
            self.admin_messages.append((chat_id, text))

        async def decline_chat_join_request(self, **kwargs):
            self.decline_calls += 1

    bot = Bot()
    req = SimpleNamespace(
        chat=SimpleNamespace(id=-1001),
        user_chat_id=item.user_chat_id,
        from_user=SimpleNamespace(
            id=item.user_id, username=item.username, first_name=item.first_name
        ),
    )
    update = SimpleNamespace(chat_join_request=req)
    context = SimpleNamespace(bot=bot)
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())

    asyncio.run(handler.on_join_request(update, context))

    assert bot.decline_calls == 0
    assert db.last_status == CandidateStatus.PENDING_ADMIN
    assert bot.admin_messages
