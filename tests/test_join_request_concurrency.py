"""Interleaving regressions for one community+applicant (no live Telegram API)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from app.handlers import GatekeeperHandlers
from app.models import CandidateStatus
from tests.test_handlers import NoopScorer, candidate, settings


def _join_update(item, date):
    return NS(chat_join_request=NS(
        chat=NS(id=item.target_chat_id),
        from_user=NS(id=item.user_id, username='new', first_name='New'),
        user_chat_id=item.user_chat_id + 1,
        date=date,
    ))


class FakeConcurrentDB:
    """Only imitates committed candidate rows; Telegram interactions are mocked."""
    def __init__(self, old):
        self.old = old
        self.new = None
        self.transfers = []
        self.started_new = 0

    async def list_expired_interviews(self, _cutoff):
        return [self.old]

    async def get_candidate(self, candidate_id):
        if candidate_id == self.old.id:
            return self.old
        return self.new if self.new and self.new.id == candidate_id else None

    async def get_active_candidate_for_chat(self, user_id, chat_id):
        active = self.new if self.new is not None else self.old
        if active.status not in ('interview', 'scoring', 'pending_admin'):
            return None
        return active

    async def get_latest_candidate_for_chat(self, user_id, chat_id):
        return self.new if self.new is not None else self.old

    async def last_cooldown_declined_request_date(self, user_id, target_chat_id):
        return None

    async def get_recent_rejected_candidate(self, *args):
        return None

    async def set_user_chat_id(self, candidate_id, chat_id):
        active = await self.get_candidate(candidate_id)
        active.user_chat_id = chat_id
        return active

    async def create_candidate(self, *, user_id, user_chat_id, target_chat_id,
                               username, first_name, request_date=None):
        self.started_new += 1
        self.old.status = CandidateStatus.SUPERSEDED.value
        self.new = candidate(
            id=2, user_id=user_id, user_chat_id=user_chat_id,
            target_chat_id=target_chat_id, username=username,
            first_name=first_name, status=CandidateStatus.INTERVIEW.value, stage=1,
            request_date=request_date, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        return self.new

    async def transition_status_if(self, candidate_id, status, *, expected_status,
                                   expected_stage=None, older_than=None,
                                   reason=None, **kwargs):
        item = await self.get_candidate(candidate_id)
        assert item is not None
        if item.status != expected_status.value:
            return None
        if expected_stage is not None and item.stage != expected_stage:
            return None
        if older_than is not None and item.updated_at >= older_than:
            return None
        item.status = status.value
        item.decision_reason = reason
        self.transfers.append(status)
        return item


def _old_candidate():
    return candidate(
        status=CandidateStatus.INTERVIEW.value, stage=1,
        updated_at=datetime.now(timezone.utc) - timedelta(hours=3),
        created_at=datetime.now(timezone.utc) - timedelta(hours=4),
        request_date=datetime.now(timezone.utc) - timedelta(hours=4),
    )


def test_new_application_is_processed_before_expiration_decline():
    """If a newer join event wins the lock, cleanup must not touch it."""
    async def scenario():
        old = _old_candidate()
        db = FakeConcurrentDB(old)
        bot = NS(decline_chat_join_request=AsyncMock(), send_message=AsyncMock())
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await h.on_join_request(_join_update(old, datetime.now(timezone.utc)), NS(bot=bot))
        assert db.new is not None
        assert await h.expire_stale_interviews(bot) == 0
        bot.decline_chat_join_request.assert_not_awaited()
        assert old.status == CandidateStatus.SUPERSEDED.value
        assert db.new.status == CandidateStatus.INTERVIEW.value
    asyncio.run(scenario())


def test_expiry_holds_lock_until_telegram_and_database_finish():
    """New join events cannot interleave with an in-flight cleanup coroutine."""
    async def scenario():
        old = _old_candidate()
        db = FakeConcurrentDB(old)
        arrived, release = asyncio.Event(), asyncio.Event()

        async def slow_decline(**kwargs):
            arrived.set()
            await release.wait()

        bot = NS(decline_chat_join_request=AsyncMock(side_effect=slow_decline),
                 send_message=AsyncMock())
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        expiry = asyncio.create_task(h.expire_stale_interviews(bot))
        await asyncio.wait_for(arrived.wait(), 2)
        join = asyncio.create_task(h.on_join_request(
            _join_update(old, datetime.now(timezone.utc)), NS(bot=bot)
        ))
        await asyncio.sleep(0)
        assert db.started_new == 0  # join processing is serialized
        release.set()
        assert await asyncio.wait_for(expiry, 2) == 1
        await asyncio.wait_for(join, 2)
        assert db.started_new == 1
        assert db.new.status == CandidateStatus.INTERVIEW.value
        assert db.transfers == [CandidateStatus.EXPIRED]
        bot.decline_chat_join_request.assert_awaited_once()
    asyncio.run(scenario())


def test_cas_rejects_late_expiry_update_after_supersede():
    async def scenario():
        old = _old_candidate()
        db = FakeConcurrentDB(old)
        await db.create_candidate(
            user_id=old.user_id, user_chat_id=old.user_chat_id+1,
            target_chat_id=old.target_chat_id,
            username='new', first_name='New'
        )
        changed = await db.transition_status_if(
            old.id, CandidateStatus.EXPIRED,
            expected_status=CandidateStatus.INTERVIEW,
            expected_stage=1,
            older_than=datetime.now(timezone.utc),
        )
        assert changed is None
        assert old.status == CandidateStatus.SUPERSEDED.value
    asyncio.run(scenario())


def test_old_score_result_after_reapplication_cannot_decline_new_request():
    """Scoring may finish after supersede; no old decision is allowed to act."""
    async def scenario():
        old = candidate(status=CandidateStatus.SCORING.value, stage=4)
        db = FakeConcurrentDB(old)
        db.old.status = CandidateStatus.SUPERSEDED.value

        async def save_score(*args):
            return None  # Real DB also refuses save if no longer scoring.

        db.save_score = save_score
        bot = NS(decline_chat_join_request=AsyncMock(), send_message=AsyncMock())
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await h._score_and_route_untracked(bot, old)
        bot.decline_chat_join_request.assert_not_awaited()
        bot.send_message.assert_not_awaited()
    asyncio.run(scenario())


def test_old_moderator_card_cannot_decide_new_application():
    async def scenario():
        old = _old_candidate()
        old.status = CandidateStatus.SUPERSEDED.value
        db = FakeConcurrentDB(old)
        db.new = candidate(
            id=2, user_id=old.user_id, target_chat_id=old.target_chat_id,
            status=CandidateStatus.INTERVIEW.value, stage=1,
        )
        bot = NS(
            approve_chat_join_request=AsyncMock(),
            decline_chat_join_request=AsyncMock(),
            send_message=AsyncMock(),
        )
        query = NS(
            data=f'candidate:reject:{old.id}',
            message=NS(chat=NS(id=settings().admin_chat_id)),
            answer=AsyncMock(), edit_message_reply_markup=AsyncMock(),
        )
        update = NS(callback_query=query, effective_user=NS(id=42, full_name='Admin'))
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await h.on_admin_decision(update, NS(bot=bot))
        bot.approve_chat_join_request.assert_not_awaited()
        bot.decline_chat_join_request.assert_not_awaited()
        assert db.new.status == CandidateStatus.INTERVIEW.value
        query.answer.assert_awaited_once()
    asyncio.run(scenario())
