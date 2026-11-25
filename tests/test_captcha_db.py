"""Real SQL checks for the captcha state machine. CI installs aiosqlite."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("aiosqlite")

from app.db import Database  # noqa: E402


def _run_db(tmp_path, scenario):
    async def main():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'captcha.db'}")
        try:
            await db.init()
            await scenario(db)
        finally:
            await db.close()
    asyncio.run(main())


async def _new(db, user_id=1, stage=0):
    return await db.create_candidate(
        user_id=user_id, user_chat_id=user_id + 1000, target_chat_id=-100,
        username="u", first_name="U", stage=stage,
    )


def test_correct_answer_moves_to_stage_1_and_clears_state(tmp_path):
    async def scenario(db):
        c = await _new(db)
        assert c.stage == 0
        await db.set_captcha(c.id, 2, "aaaaaaaa")
        outcome, cur = await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=2, max_attempts=2)
        assert outcome == "passed" and cur.stage == 1
        assert cur.captcha_correct_index is None and cur.captcha_token is None
        # replaying the same button press is "stale", never graded twice
        again, _ = await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=2, max_attempts=2)
        assert again == "stale"
    _run_db(tmp_path, scenario)


def test_captcha_pass_and_first_question_delivery_update_join_event_atomically(tmp_path):
    from app.models import JoinRequestEvent
    from sqlalchemy import select

    async def scenario(db):
        c = await _new(db)
        await db.record_join_request_event(
            update_id=91001, target_chat_id=-100, user_id=c.user_id,
            request_date=None,
        )
        await db.update_join_request_event(91001, "captcha_sent", candidate_id=c.id)
        await db.set_captcha(c.id, 2, "captcha-token")

        outcome, passed = await db.answer_captcha(
            c.id, token="captcha-token", chosen_index=2, max_attempts=2,
        )
        assert outcome == "passed"
        async with db.sessions() as session:
            event = await session.scalar(select(JoinRequestEvent).where(
                JoinRequestEvent.update_id == 91001,
            ))
            assert event.status == "captcha_passed"
            assert event.first_message_id is None

        assert await db.confirm_captcha_delivery(
            c.id, "first_question", first_message_id=456,
        )
        async with db.sessions() as session:
            event = await session.scalar(select(JoinRequestEvent).where(
                JoinRequestEvent.update_id == 91001,
            ))
            assert event.status == "first_question_sent"
            assert event.first_message_id == 456
        current = await db.get_candidate(c.id)
        assert current.captcha_pending_action is None
        assert passed.stage == 1
    _run_db(tmp_path, scenario)


def test_captcha_terminal_actions_update_join_event(tmp_path):
    from app.models import JoinRequestEvent
    from sqlalchemy import select

    async def scenario(db):
        failed = await _new(db, user_id=11)
        await db.record_join_request_event(
            update_id=91002, target_chat_id=-100, user_id=failed.user_id,
            request_date=None,
        )
        await db.update_join_request_event(91002, "captcha_sent", candidate_id=failed.id)
        await db.set_captcha(failed.id, 1, "failed-token", max_attempts=1)
        outcome, row = await db.answer_captcha(
            failed.id, token="failed-token", chosen_index=0, max_attempts=1,
        )
        assert outcome == "failed"
        closed = await db.finish_captcha_action(
            failed.id, "decline_failed", expired=True, reason="failed",
        )
        assert closed.status == "expired"

        timed_out = await _new(db, user_id=12)
        await db.record_join_request_event(
            update_id=91003, target_chat_id=-100, user_id=timed_out.user_id,
            request_date=None,
        )
        await db.update_join_request_event(91003, "captcha_sent", candidate_id=timed_out.id)
        await db.set_captcha(timed_out.id, 1, "timeout-token")
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import update as sql_update
        from app.models import Candidate
        async with db.sessions() as session:
            await session.execute(sql_update(Candidate).where(
                Candidate.id == timed_out.id,
            ).values(captcha_deadline_at=datetime.now(timezone.utc)-timedelta(seconds=1)))
            await session.commit()
        outcome, row = await db.answer_captcha(
            timed_out.id, token="timeout-token", chosen_index=1, max_attempts=2,
        )
        assert outcome == "expired"
        await db.finish_captcha_action(
            timed_out.id, "decline_timeout", expired=True, reason="timeout",
        )

        delivery_failed = await _new(db, user_id=13)
        await db.record_join_request_event(
            update_id=91004, target_chat_id=-100, user_id=delivery_failed.user_id,
            request_date=None,
        )
        await db.update_join_request_event(91004, "captcha_sent", candidate_id=delivery_failed.id)
        await db.set_captcha(delivery_failed.id, 1, "delivery-token")
        outcome, _ = await db.answer_captcha(
            delivery_failed.id, token="delivery-token", chosen_index=1, max_attempts=2,
        )
        assert outcome == "passed"
        closed = await db.finish_captcha_action(
            delivery_failed.id, "first_question", reason="Forbidden",
            error_type="Forbidden",
        )
        assert closed.status == "pending_admin"

        async with db.sessions() as session:
            failed_event = await session.scalar(select(JoinRequestEvent).where(
                JoinRequestEvent.update_id == 91002,
            ))
            timeout_event = await session.scalar(select(JoinRequestEvent).where(
                JoinRequestEvent.update_id == 91003,
            ))
            delivery_event = await session.scalar(select(JoinRequestEvent).where(
                JoinRequestEvent.update_id == 91004,
            ))
            assert failed_event.status == "captcha_failed"
            assert timeout_event.status == "captcha_timeout"
            assert delivery_event.status == "first_question_failed"
            assert delivery_event.error_type == "Forbidden"
    _run_db(tmp_path, scenario)


def test_wrong_answers_retry_then_fail_and_replay_is_stale(tmp_path):
    async def scenario(db):
        c = await _new(db)
        await db.set_captcha(c.id, 1, "aaaaaaaa")
        outcome, cur = await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=0, max_attempts=2)
        assert outcome == "retry" and cur.captcha_attempts == 1 and cur.stage == 0
        # same keyboard again (double tap / replay): no second attempt counted
        outcome, _ = await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=0, max_attempts=2)
        assert outcome == "stale"
        await db.set_captcha(c.id, 3, "bbbbbbbb")
        await db.confirm_captcha_delivery(c.id, "retry_delivery")
        outcome, cur = await db.answer_captcha(c.id, token="bbbbbbbb", chosen_index=0, max_attempts=2)
        assert outcome == "failed" and cur.captcha_attempts == 2
    _run_db(tmp_path, scenario)


def test_old_keyboard_cannot_answer_a_newer_captcha(tmp_path):
    async def scenario(db):
        c = await _new(db)
        await db.set_captcha(c.id, 1, "oldtoken")
        await db.set_captcha(c.id, 3, "newtoken")  # applicant re-applied
        outcome, _ = await db.answer_captcha(c.id, token="oldtoken", chosen_index=3, max_attempts=2)
        assert outcome == "stale"
        outcome, _ = await db.answer_captcha(c.id, token="newtoken", chosen_index=3, max_attempts=2)
        assert outcome == "passed"
    _run_db(tmp_path, scenario)


def test_expiry_uses_shorter_captcha_cutoff_only_for_stage_0(tmp_path):
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    from app.models import Candidate as C

    async def scenario(db):
        waiting = await _new(db, user_id=1, stage=0)
        asking = await _new(db, user_id=2, stage=1)
        old = datetime.now(timezone.utc) - timedelta(minutes=10)
        async with db.sessions() as session:
            await session.execute(update(C).values(updated_at=old))
            await session.execute(update(C).where(C.id == waiting.id).values(captcha_deadline_at=old))
            await session.commit()
        now = datetime.now(timezone.utc)
        found = await db.list_expired_interviews(
            now - timedelta(minutes=30), captcha_cutoff=now - timedelta(minutes=3),
        )
        assert [x.id for x in found] == [waiting.id]   # stage 1 is not 30 min old yet
        assert await db.list_expired_interviews(now - timedelta(minutes=30)) == []
        assert asking.id not in [x.id for x in found]
    _run_db(tmp_path, scenario)


def test_absolute_deadline_and_late_correct_answer(tmp_path):
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    from app.models import Candidate

    async def scenario(db):
        c = await _new(db)
        first = await db.set_captcha(c.id, 1, "aaaaaaaa")
        assert first.captcha_deadline_at is not None
        assert (await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=0, max_attempts=2))[0] == "retry"
        second = await db.set_captcha(c.id, 2, "bbbbbbbb")
        assert second.captcha_deadline_at == first.captcha_deadline_at
        assert await db.confirm_captcha_delivery(c.id, "retry_delivery")
        async with db.sessions() as session:
            await session.execute(update(Candidate).where(Candidate.id == c.id).values(
                captcha_deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
            await session.commit()
        outcome, row = await db.answer_captcha(c.id, token="bbbbbbbb", chosen_index=2, max_attempts=2)
        assert outcome == "expired" and row.stage == 0
        assert row.captcha_pending_action == "decline_timeout"
        assert await db.set_captcha(c.id, 0, "cccccccc") is None
        assert await db.claim_captcha_timeout(c.id, timeout_minutes=3, max_attempts=2) is None
    _run_db(tmp_path, scenario)


def test_exhausted_intent_survives_restart_and_notice_ack(tmp_path):
    from app.models import CandidateStatus

    async def scenario(db):
        c = await _new(db)
        await db.set_captcha(c.id, 1, "aaaaaaaa", max_attempts=1)
        outcome, row = await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=0, max_attempts=1)
        assert outcome == "failed" and row.captcha_pending_action == "decline_failed"
        # Separate database instance models a restarted process.
        other = Database(f"sqlite+aiosqlite:///{tmp_path / 'captcha.db'}")
        try:
            await other.init()
            assert await other.set_captcha(c.id, 0, "bbbbbbbb", max_attempts=1) is None
            assert (await other.answer_captcha(c.id, token="aaaaaaaa", chosen_index=1, max_attempts=1))[0] == "stale"
            assert [r.id for r in await other.list_captcha_recovery(1)] == [c.id]
            row = await other.finish_captcha_action(c.id, "decline_failed", reason="uncertain")
            assert row.status == CandidateStatus.PENDING_ADMIN.value and row.captcha_notice_pending
            assert [r.id for r in await db.list_captcha_notices()] == [c.id]
            assert await other.mark_captcha_notice_sent(c.id)
            assert await db.list_captcha_notices() == []
        finally:
            await other.close()
    _run_db(tmp_path, scenario)


def test_legacy_deadline_uses_creation_not_last_retry(tmp_path):
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    from app.models import Candidate

    async def scenario(db):
        c = await _new(db)
        now = datetime.now(timezone.utc)
        async with db.sessions() as session:
            await session.execute(update(Candidate).where(Candidate.id == c.id).values(
                created_at=now-timedelta(minutes=4), updated_at=now,
                captcha_deadline_at=None, captcha_token="aaaaaaaa", captcha_correct_index=1))
            await session.commit()
        assert await db.set_captcha(c.id, 1, "bbbbbbbb") is None
        assert (await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=1, max_attempts=2))[0] == "expired"
    _run_db(tmp_path, scenario)


def test_recovery_of_legacy_exhausted_row_and_superseded_notice(tmp_path):
    from sqlalchemy import update
    from app.models import Candidate, CandidateStatus

    async def scenario(db):
        c = await _new(db)
        async with db.sessions() as session:
            await session.execute(update(Candidate).where(Candidate.id == c.id).values(
                captcha_attempts=2, captcha_pending_action=None,
                captcha_token="aaaaaaaa", captcha_correct_index=1))
            await session.commit()
        assert await db.set_captcha(c.id, 0, "bbbbbbbb") is None
        assert (await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=1, max_attempts=2))[0] == "stale"
        assert [r.id for r in await db.list_captcha_recovery(2)] == [c.id]
        row = await db.finish_captcha_action(c.id, None, reason="legacy", exhausted_limit=2)
        assert row.status == CandidateStatus.PENDING_ADMIN.value
        await _new(db)
        assert await db.list_captcha_notices() == []
    _run_db(tmp_path, scenario)


def test_unexpired_legacy_row_can_receive_and_answer_captcha(tmp_path):
    from sqlalchemy import update
    from app.models import Candidate

    async def scenario(db):
        c = await _new(db)
        async with db.sessions() as session:
            await session.execute(update(Candidate).where(Candidate.id == c.id).values(
                captcha_deadline_at=None))
            await session.commit()
        issued = await db.set_captcha(c.id, 1, "aaaaaaaa")
        assert issued is not None and issued.captcha_deadline_at is not None
        assert (await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=1, max_attempts=2))[0] == "passed"
    _run_db(tmp_path, scenario)


def test_additive_migration_keeps_existing_candidate(tmp_path):
    from sqlalchemy import text

    async def scenario(db):
        c = await _new(db)
        await db.set_captcha(c.id, 1, "aaaaaaaa")
        async with db.engine.begin() as conn:
            for name in ("captcha_deadline_at", "captcha_pending_action", "captcha_notice_pending"):
                await conn.execute(text(f"ALTER TABLE candidates DROP COLUMN {name}"))
        await db.init()
        await db.init()
        row = await db.get_candidate(c.id)
        assert row.captcha_deadline_at is None
        assert row.captcha_pending_action is None
        assert row.captcha_notice_pending is False
        assert row.captcha_token == "aaaaaaaa"
        assert (await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=1, max_attempts=2))[0] == "passed"
    _run_db(tmp_path, scenario)


def test_maintenance_claims_timeout_from_fixed_deadline(tmp_path):
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    from app.models import Candidate

    async def scenario(db):
        c = await _new(db)
        await db.set_captcha(c.id, 1, "aaaaaaaa")
        async with db.sessions() as session:
            await session.execute(update(Candidate).where(Candidate.id == c.id).values(
                captcha_deadline_at=datetime.now(timezone.utc)-timedelta(seconds=1)))
            await session.commit()
        row = await db.claim_captcha_timeout(c.id, timeout_minutes=3, max_attempts=2)
        assert row.captcha_pending_action == "decline_timeout"
        assert await db.claim_captcha_timeout(c.id, timeout_minutes=3, max_attempts=2) is None
        assert (await db.answer_captcha(c.id, token="aaaaaaaa", chosen_index=1, max_attempts=2))[0] == "stale"
        closed = await db.finish_captcha_action(c.id, "decline_timeout", expired=True, reason="timeout")
        assert closed.status == "expired" and not closed.captcha_notice_pending
    _run_db(tmp_path, scenario)
