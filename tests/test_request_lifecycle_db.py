"""Real SQL regression checks. CI installs aiosqlite via requirements.txt."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip('aiosqlite')

from app.db import Database
from app.models import CandidateStatus


def test_missing_telegram_request_does_not_trigger_cooldown(tmp_path):
    async def scenario():
        db = Database(f'sqlite+aiosqlite:///{tmp_path / "gatekeeper.db"}')
        try:
            await db.init()
            old = await db.create_candidate(
                user_id=8852071958, user_chat_id=1122,
                target_chat_id=-1002094364257, username='Jennifer', first_name='Jennifer',
            )
            await db.set_status(
                old.id, CandidateStatus.EXPIRED,
                reason='Заявка отсутствует в Telegram (Hide_requester_missing)',
            )
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            assert await db.get_recent_rejected_candidate(
                8852071958, -1002094364257, cutoff,
            ) is None

            # Ordinary timeouts remain under the configured cooldown.
            normal = await db.create_candidate(
                user_id=8852071958, user_chat_id=3344,
                target_chat_id=-1002094364257, username='Jennifer', first_name='Jennifer',
            )
            await db.set_status(
                normal.id, CandidateStatus.EXPIRED,
                reason='Анкета не завершена до истечения таймаута',
            )
            recent = await db.get_recent_rejected_candidate(
                8852071958, -1002094364257, cutoff,
            )
            assert recent is not None and recent.id == normal.id
        finally:
            await db.close()
    asyncio.run(scenario())


def test_new_candidate_supersedes_previous_pending_one(tmp_path):
    async def scenario():
        db = Database(f'sqlite+aiosqlite:///{tmp_path / "gatekeeper.db"}')
        try:
            await db.init()
            old = await db.create_candidate(
                user_id=123, user_chat_id=123, target_chat_id=-1001,
                username='candidate', first_name='Test',
            )
            await db.set_status(old.id, CandidateStatus.PENDING_ADMIN,
                                reason='Ожидает модератора')
            newer = await db.create_candidate(
                user_id=123, user_chat_id=124, target_chat_id=-1001,
                username='candidate', first_name='Test',
            )
            assert (await db.get_candidate(old.id)).status == CandidateStatus.SUPERSEDED
            assert (await db.get_active_candidate_for_chat(123, -1001)).id == newer.id
        finally:
            await db.close()
    asyncio.run(scenario())


def test_status_cas_does_not_overwrite_superseded_or_fresh_interview(tmp_path):
    """CI executes this with real SQLite; PostgreSQL uses the same SQL CAS."""
    async def scenario():
        db = Database(f'sqlite+aiosqlite:///{tmp_path / "state_cas.db"}')
        try:
            await db.init()
            original = await db.create_candidate(
                user_id=101, user_chat_id=101, target_chat_id=-1001,
                username=None, first_name='Test',
            )
            assert await db.transition_status_if(
                original.id, CandidateStatus.EXPIRED,
                expected_status=CandidateStatus.PENDING_ADMIN,
            ) is None
            assert await db.transition_status_if(
                original.id, CandidateStatus.EXPIRED,
                expected_status=CandidateStatus.INTERVIEW, expected_stage=2,
            ) is None
            assert await db.transition_status_if(
                original.id, CandidateStatus.EXPIRED,
                expected_status=CandidateStatus.INTERVIEW,
                older_than=datetime.now(timezone.utc) - timedelta(hours=1),
            ) is None

            newer = await db.create_candidate(
                user_id=101, user_chat_id=102, target_chat_id=-1001,
                username=None, first_name='New',
            )
            assert (await db.get_candidate(original.id)).status == 'superseded'
            assert await db.transition_status_if(
                original.id, CandidateStatus.EXPIRED,
                expected_status=CandidateStatus.INTERVIEW,
            ) is None
            assert (await db.get_candidate(newer.id)).status == 'interview'

            completed = await db.transition_status_if(
                newer.id, CandidateStatus.PENDING_ADMIN,
                expected_status=CandidateStatus.INTERVIEW, expected_stage=1,
                reason='manual review',
            )
            assert completed.id == newer.id
            assert completed.status == 'pending_admin'
            assert await db.transition_status_if(
                newer.id, CandidateStatus.EXPIRED,
                expected_status=CandidateStatus.INTERVIEW,
            ) is None
        finally:
            await db.close()
    asyncio.run(scenario())


def test_old_scoring_cannot_store_decision_after_new_application(tmp_path):
    async def scenario():
        db = Database(f'sqlite+aiosqlite:///{tmp_path / "score_cas.db"}')
        try:
            await db.init()
            old = await db.create_candidate(
                user_id=100, user_chat_id=100, target_chat_id=-1001,
                username=None, first_name='Test',
            )
            await db.set_status(old.id, CandidateStatus.SCORING)
            new = await db.create_candidate(
                user_id=100, user_chat_id=101, target_chat_id=-1001,
                username=None, first_name='Test',
            )
            from tests.test_handlers import score
            assert await db.save_score(old.id, score(), '{}') is None
            assert (await db.get_candidate(old.id)).status == 'superseded'
            assert (await db.get_candidate(new.id)).status == 'interview'
        finally:
            await db.close()
    asyncio.run(scenario())


def test_latest_candidate_for_chat_includes_terminal_and_is_chat_scoped(tmp_path):
    async def scenario():
        db = Database(f'sqlite+aiosqlite:///{tmp_path / "latest.db"}')
        try:
            await db.init()
            t0 = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
            previous = await db.create_candidate(
                user_id=101, user_chat_id=202, target_chat_id=-1001,
                username=None, first_name='Test', request_date=t0,
            )
            await db.set_status(previous.id, CandidateStatus.APPROVED)
            other_chat = await db.create_candidate(
                user_id=101, user_chat_id=203, target_chat_id=-1002,
                username=None, first_name='Test', request_date=t0 + timedelta(minutes=1),
            )
            latest = await db.get_latest_candidate_for_chat(101, -1001)
            assert latest.id == previous.id
            assert latest.status == CandidateStatus.APPROVED.value
            assert (await db.get_latest_candidate_for_chat(101, -1002)).id == other_chat.id
            assert await db.get_latest_candidate_for_chat(999, -1001) is None
        finally:
            await db.close()
    asyncio.run(scenario())



def test_cooldown_declines_are_durable_unique_and_chat_scoped(tmp_path):
    async def scenario():
        url = f'sqlite+aiosqlite:///{tmp_path / "cooldown_ledger.db"}'
        initial = Database(url)
        t0 = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)
        try:
            await initial.init()
            assert await initial.last_cooldown_declined_request_date(123, -1001) is None
            await initial.record_cooldown_declined_request(123, -1001, t0)
            # Same-second retries are idempotent, even across database instances.
            await initial.record_cooldown_declined_request(123, -1001, t0)
            assert await initial.last_cooldown_declined_request_date(123, -1001) == t0
            assert await initial.last_cooldown_declined_request_date(123, -1002) is None
            assert await initial.last_cooldown_declined_request_date(456, -1001) is None
        finally:
            await initial.close()

        restarted = Database(url)
        try:
            await restarted.init()
            assert await restarted.last_cooldown_declined_request_date(123, -1001) == t0
            await restarted.record_cooldown_declined_request(
                123, -1001, t0 + timedelta(minutes=3)
            )
            assert await restarted.last_cooldown_declined_request_date(123, -1001) == (
                t0 + timedelta(minutes=3)
            )
            from sqlalchemy import func, select
            from app.models import CooldownDeclinedJoinRequest
            async with restarted.sessions() as session:
                count = await session.scalar(select(func.count(CooldownDeclinedJoinRequest.id)))
            assert count == 2
        finally:
            await restarted.close()

    asyncio.run(scenario())



def test_cooldown_intent_is_durable_and_recovers_without_retry(tmp_path):
    async def scenario():
        from sqlalchemy import select
        from app.models import CooldownDeclinedJoinRequest as Attempt
        url = f'sqlite+aiosqlite:///{tmp_path / "intent.db"}'
        now = datetime.now(timezone.utc)
        request_date = datetime(2026, 11, 6, 11, tzinfo=timezone.utc)
        first = Database(url)
        try:
            await first.init()
            assert await first.begin_cooldown_decline(123, -1001, request_date)
            # Cross-connection uniqueness must be durable, not an in-memory set.
            assert not await first.begin_cooldown_decline(123, -1001, request_date)
            assert await first.last_cooldown_declined_request_date(123, -1001) == request_date
            assert await first.last_cooldown_declined_request_date(123, -1002) is None
            async with first.sessions() as session:
                record = (await session.scalars(select(Attempt))).one()
                assert record.status == "pending"
        finally:
            await first.close()

        recovered = Database(url)
        try:
            await recovered.init()
            # Recent operations are not reclassified while the handler runs.
            assert await recovered.mark_stale_cooldown_intents(now - timedelta(minutes=30)) == 0
            assert await recovered.mark_stale_cooldown_intents(now + timedelta(minutes=1)) == 1
            assert await recovered.mark_stale_cooldown_intents(now + timedelta(minutes=1)) == 0
            items = await recovered.list_unalerted_cooldown_intents()
            assert len(items) == 1 and items[0].status == "uncertain"
            assert await recovered.mark_cooldown_intent_alerted(items[0].id)
            assert not await recovered.mark_cooldown_intent_alerted(items[0].id)
            assert await recovered.list_unalerted_cooldown_intents() == []
            assert not await recovered.begin_cooldown_decline(123, -1001, request_date)
            assert not await recovered.complete_cooldown_decline(
                123, -1001, request_date, confirmed=True,
            )
            later = request_date + timedelta(minutes=5)
            assert await recovered.begin_cooldown_decline(123, -1001, later)
            assert await recovered.complete_cooldown_decline(123, -1001, later, confirmed=True)
            assert await recovered.last_cooldown_declined_request_date(123, -1001) == later
            async with recovered.sessions() as session:
                attempts = (await session.scalars(select(Attempt).order_by(Attempt.id))).all()
                assert [a.status for a in attempts] == ["uncertain", "declined"]
                assert attempts[1].confirmed_at is not None
        finally:
            await recovered.close()
    asyncio.run(scenario())


def test_legacy_sqlite_ledger_migrates_without_data_loss(tmp_path):
    import sqlite3
    from sqlalchemy import select
    from app.models import CooldownDeclinedJoinRequest as Attempt
    path = tmp_path / 'legacy_cooldown.db'
    # Emulate an older deployed table with no new state columns.
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE cooldown_declined_join_requests (
            id INTEGER PRIMARY KEY,
            target_chat_id BIGINT NOT NULL,
            user_id BIGINT NOT NULL,
            request_date DATETIME NOT NULL,
            declined_at DATETIME NOT NULL,
            CONSTRAINT uq_cooldown_declined_join_request
              UNIQUE (target_chat_id, user_id, request_date)
        )""")
        conn.execute("""INSERT INTO cooldown_declined_join_requests
            (target_chat_id, user_id, request_date, declined_at)
            VALUES (-1001, 101, '2026-11-06 11:00:00', '2026-11-06 11:01:00')""")

    async def scenario():
        db = Database(f'sqlite+aiosqlite:///{path}')
        try:
            await db.init()
            await db.init()  # A restarted deployment must not fail migration.
            async with db.sessions() as session:
                record = (await session.scalars(select(Attempt))).one()
                assert (record.target_chat_id, record.user_id) == (-1001, 101)
                assert record.status == "declined"  # Backfill old successes.
                assert record.confirmed_at is None
                assert record.alerted_at is None
            assert await db.last_cooldown_declined_request_date(101, -1001) == (
                datetime(2026, 11, 6, 11, tzinfo=timezone.utc)
            )
        finally:
            await db.close()
    asyncio.run(scenario())
