"""Regression tests for interrupted scoring, including Telegram transport failures.

External APIs are mocked; optional SQLite integration runs in CI where aiosqlite
is installed. No DeepSeek/Telegram token or real join requests are required.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import TelegramError

from app import texts
from app.config import Settings
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus


def cfg() -> Settings:
    return Settings(
        bot_token="test", target_chat_id=-1001, admin_chat_id=-2002,
        admin_user_ids={42}, database_url="sqlite+aiosqlite:///:memory:",
        llm_base_url="https://example.invalid", llm_api_key="test",
        llm_model="test", llm_timeout_seconds=35,
        auto_reject_spam_score=85, auto_reject_core_score_max=15,
        auto_reject_min_ai_confidence=80, max_answer_length=2500,
    )


def scoring_candidate(*, age_minutes=0, status=CandidateStatus.SCORING):
    return Candidate(
        id=14, user_id=2124382147, user_chat_id=777, target_chat_id=-1001,
        username="test", first_name="Test", status=status.value,
        stage=4, location_answer="Нет", business_answer="Нет",
        goal_answer="Нет", updated_at=datetime.now(timezone.utc) - timedelta(minutes=age_minutes),
    )


class RecoveryDB:
    """Simulate atomic compare-and-set plus durable unacknowledged notice."""
    def __init__(self, item):
        self.item = item
        self.claims = 0
        self.notices_sent = 0
        self.reason = None

    async def list_stale_scoring(self, cutoff):
        item = self.item
        was_processing = item.status == "scoring" or (item.status == "interview" and item.stage == 4)
        return [item] if (was_processing and item.updated_at < cutoff) else []

    async def recover_scoring_to_admin(self, candidate_id, cutoff):
        item = self.item
        was_processing = item.status == "scoring" or (item.status == "interview" and item.stage == 4)
        if item.id != candidate_id or not was_processing or not item.updated_at < cutoff:
            return None
        item.status = "pending_admin"
        self.reason = "notice pending"
        self.claims += 1
        return item

    async def list_pending_recovery_notices(self):
        return [self.item] if self.item.status == "pending_admin" and self.reason == "notice pending" else []

    async def mark_recovery_notice_sent(self, candidate_id):
        if self.item.id != candidate_id or self.reason != "notice pending":
            return False
        self.reason = "notice sent"
        self.notices_sent += 1
        return True


class RecoveryBot:
    def __init__(self, *, reject_admin=False):
        self.sent = []
        self.reject_admin = reject_admin
        self.decline_calls = 0

    async def send_message(self, chat_id=None, text=None, **kwargs):
        if self.reject_admin and chat_id == -2002:
            raise TelegramError("admin chat temporarily unavailable")
        self.sent.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=len(self.sent))

    async def decline_chat_join_request(self, **kwargs):
        self.decline_calls += 1
        raise AssertionError("Recovery must NEVER auto-decline")


def test_processing_message_failure_does_not_block_ai_route():
    item = scoring_candidate(status=CandidateStatus.INTERVIEW)
    item.stage = 3
    item.goal_answer = None

    class DB:
        async def get_active_interview(self, user_id):
            return item if item.status == "interview" else None

        async def complete_interview(self, candidate_id, answer):
            assert candidate_id == 14
            item.goal_answer = answer
            item.stage = 4
            item.status = "scoring"
            return item

    bot = RecoveryBot()
    handler = GatekeeperHandlers(settings=cfg(), db=DB(), scorer=object())
    routed = []

    async def score_route(actual_bot, actual_candidate):
        routed.append(actual_candidate.id)

    handler._score_and_route = score_route
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(type="private", id=777),
        effective_user=SimpleNamespace(id=2124382147),
        effective_message=SimpleNamespace(
            text="Нет дополнительных сведений.",
            reply_text=AsyncMock(side_effect=TelegramError("sendMessage failed")),
        ),
    )

    asyncio.run(handler.on_private_message(update, SimpleNamespace(bot=bot)))

    assert routed == [14]
    assert item.status == "scoring"
    assert item.stage == 4
    assert item.goal_answer == "Нет дополнительных сведений."


def test_startup_recovers_recent_scoring_immediately_once():
    item = scoring_candidate(age_minutes=0)
    db, bot = RecoveryDB(item), RecoveryBot()
    handler = GatekeeperHandlers(settings=cfg(), db=db, scorer=object())

    async def scenario():
        assert await handler.recover_stale_scoring(bot, startup=True) == 1
        assert await handler.recover_stale_scoring(bot, startup=True) == 0

    asyncio.run(scenario())
    assert item.status == "pending_admin"
    assert db.claims == 1
    assert db.notices_sent == 1
    assert bot.decline_calls == 0
    admin_cards = [m for m in bot.sent if m[0] == -2002]
    assert len(admin_cards) == 1
    assert "прервалась" in admin_cards[0][1]
    assert admin_cards[0][2]["reply_markup"] is not None
    assert [m[1] for m in bot.sent if m[0] == 777] == [texts.PENDING_ADMIN]


def test_periodic_recovery_is_delayed_and_skips_active_work():
    item = scoring_candidate(age_minutes=2)
    db, bot = RecoveryDB(item), RecoveryBot()
    handler = GatekeeperHandlers(settings=cfg(), db=db, scorer=object())

    async def scenario():
        assert await handler.recover_stale_scoring(bot) == 0
        item.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
        handler._active_scoring_ids.add(14)
        assert await handler.recover_stale_scoring(bot) == 0
        handler._active_scoring_ids.discard(14)
        assert await handler.recover_stale_scoring(bot) == 1

    asyncio.run(scenario())
    assert bot.decline_calls == 0
    assert db.claims == 1


def test_legacy_stage4_interview_is_recovered_not_auto_expired():
    item = scoring_candidate(status=CandidateStatus.INTERVIEW)
    # Old release could save stage=4 while status remained interview.
    db, bot = RecoveryDB(item), RecoveryBot()
    handler = GatekeeperHandlers(settings=cfg(), db=db, scorer=object())

    asyncio.run(handler.recover_stale_scoring(bot, startup=True))

    assert item.status == "pending_admin"
    assert db.notices_sent == 1
    assert bot.decline_calls == 0


def test_failed_recovery_card_retries_without_repeating_transition():
    item = scoring_candidate(age_minutes=60)
    db, bot = RecoveryDB(item), RecoveryBot(reject_admin=True)
    handler = GatekeeperHandlers(settings=cfg(), db=db, scorer=object())

    async def scenario():
        assert await handler.recover_stale_scoring(bot) == 1
        assert item.status == "pending_admin"
        assert db.reason == "notice pending"
        assert not bot.sent
        bot.reject_admin = False
        assert await handler.recover_stale_scoring(bot) == 0
        assert db.reason == "notice sent"
        assert await handler.recover_stale_scoring(bot) == 0

    asyncio.run(scenario())
    assert db.claims == 1
    assert db.notices_sent == 1
    assert len([m for m in bot.sent if m[0] == -2002]) == 1
    assert bot.decline_calls == 0


def test_db_atomic_finish_and_recovery_compete_safely(tmp_path):
    pytest.importorskip("aiosqlite")
    from app.db import Database

    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'recovery.db'}")
        await db.init()
        try:
            created = await db.create_candidate(
                user_id=100, user_chat_id=777, target_chat_id=-1001,
                username="test", first_name="Test",
            )
            await db.save_answer(created.id, field="location_answer", answer="A", next_stage=2)
            await db.save_answer(created.id, field="business_answer", answer="B", next_stage=3)
            scoring = await db.complete_interview(created.id, "C")
            assert scoring is not None
            assert scoring.stage == 4 and scoring.status == "scoring"
            assert scoring.goal_answer == "C"
            assert await db.complete_interview(created.id, "Duplicate") is None
            current = await db.get_candidate(created.id)
            assert current.goal_answer == "C"

            cutoff = datetime.now(timezone.utc) + timedelta(seconds=1)
            stale = await db.list_stale_scoring(cutoff)
            assert [row.id for row in stale] == [created.id]
            first = await db.recover_scoring_to_admin(created.id, cutoff)
            assert first is not None and first.status == "pending_admin"
            assert await db.recover_scoring_to_admin(created.id, cutoff) is None
            assert len(await db.list_pending_recovery_notices()) == 1
            assert await db.mark_recovery_notice_sent(created.id)
            assert not await db.mark_recovery_notice_sent(created.id)
            assert not await db.list_pending_recovery_notices()
            assert (await db.get_candidate(created.id)).status == "pending_admin"

            # A legacy interrupted transaction (stage=4, interview) must not
            # be treated as an expired unanswered questionnaire.
            legacy = await db.create_candidate(
                user_id=101, user_chat_id=778, target_chat_id=-1001,
                username="legacy", first_name="Legacy",
            )
            await db.save_answer(legacy.id, field="goal_answer", answer="C", next_stage=4)
            cutoff = datetime.now(timezone.utc) + timedelta(seconds=1)
            assert legacy.id not in [c.id for c in await db.list_expired_interviews(cutoff)]
            assert legacy.id in [c.id for c in await db.list_stale_scoring(cutoff)]
            assert (await db.recover_scoring_to_admin(legacy.id, cutoff)).status == "pending_admin"
        finally:
            await db.close()

    asyncio.run(scenario())
