"""Durable first-question delivery and fail-closed alert regression tests."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, update
from telegram.error import TelegramError

from app.db import Database
from app.handlers import GatekeeperHandlers
from app.models import CandidateStatus, JoinRequestEvent
from app.status_report import build_status_report
from tests.test_handlers import NoopScorer, settings

pytest.importorskip("aiosqlite")


class Forbidden(TelegramError):
    """PTB-compatible exception used to assert the persisted class name."""


def make_update(update_id: int, *, user_id: int = 900, date=None):
    return NS(
        update_id=update_id,
        chat_join_request=NS(
            chat=NS(id=-1001),
            from_user=NS(id=user_id, username="sample", first_name="Пример"),
            date=date or datetime.now(timezone.utc), user_chat_id=20000 + user_id,
        ),
    )


async def get_event(db: Database, update_id: int):
    async with db.sessions() as session:
        return await session.scalar(select(JoinRequestEvent).where(JoinRequestEvent.update_id == update_id))


def with_db(tmp_path, coro):
    async def run():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'join.sqlite3'}")
        try:
            await db.init()
            return await coro(db)
        finally:
            await db.close()
    return asyncio.run(run())


def test_join_event_is_committed_before_first_send_and_records_telegram_message_id(tmp_path):
    async def scenario(db):
        bot = NS(send_message=AsyncMock(return_value=NS(message_id=456)))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await handler.on_join_request(make_update(5001), NS(bot=bot))
        e = await get_event(db, 5001)
        assert e.status == "first_question_sent"
        assert e.candidate_id is not None
        assert e.first_message_id == 456
        assert e.alert_required is False
        c = await db.get_candidate(e.candidate_id)
        assert c.status == CandidateStatus.INTERVIEW.value
        await db.record_join_request_event(
            update_id=5001, target_chat_id=-1001, user_id=900,
            request_date=datetime.now(timezone.utc),
        )
        assert (await get_event(db, 5001)).first_message_id == 456
    with_db(tmp_path, scenario)


def test_captcha_send_is_not_recorded_as_first_question_and_pass_records_real_message_id(tmp_path):
    async def scenario(db):
        bot = NS(send_message=AsyncMock(side_effect=[NS(message_id=123), NS(message_id=456)]))
        captcha_settings = replace(settings(), captcha_enabled=True)
        handler = GatekeeperHandlers(settings=captcha_settings, db=db, scorer=NoopScorer())
        update = make_update(5101)

        await handler.on_join_request(update, NS(bot=bot))
        event = await get_event(db, 5101)
        assert event.status == "captcha_sent"
        assert event.first_message_id is None
        candidate = await db.get_candidate(event.candidate_id)
        assert candidate.stage == 0
        assert candidate.captcha_token
        assert bot.send_message.await_count == 1

        query = NS(
            data=f"captcha:{candidate.id}:{candidate.captcha_token}:{candidate.captcha_correct_index}",
            from_user=NS(id=candidate.user_id),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )
        await handler.on_captcha(NS(callback_query=query), NS(bot=bot))

        event = await get_event(db, 5101)
        assert event.status == "first_question_sent"
        assert event.first_message_id == 456
        candidate = await db.get_candidate(candidate.id)
        assert candidate.stage == 1
        assert candidate.captcha_pending_action is None
        assert bot.send_message.await_count == 2
    with_db(tmp_path, scenario)


def test_captcha_passed_but_forbidden_first_question_persists_error_type(tmp_path):
    async def scenario(db):
        bot = NS(
            send_message=AsyncMock(side_effect=[
                NS(message_id=123),
                Forbidden("bot can't initiate conversation with a user"),
                NS(message_id=82),  # administrator notification
            ]),
            decline_chat_join_request=AsyncMock(),
        )
        handler = GatekeeperHandlers(
            settings=replace(settings(), captcha_enabled=True), db=db, scorer=NoopScorer(),
        )
        await handler.on_join_request(make_update(5104), NS(bot=bot))
        event = await get_event(db, 5104)
        candidate = await db.get_candidate(event.candidate_id)
        assert event.status == "captcha_sent"
        assert event.error_type is None

        query = NS(
            data=f"captcha:{candidate.id}:{candidate.captcha_token}:{candidate.captcha_correct_index}",
            from_user=NS(id=candidate.user_id),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )
        await handler.on_captcha(NS(callback_query=query), NS(bot=bot))

        event = await get_event(db, 5104)
        candidate = await db.get_candidate(candidate.id)
        assert candidate.status == CandidateStatus.PENDING_ADMIN.value
        assert candidate.stage == 1
        assert "Forbidden" in candidate.decision_reason
        assert event.status == "first_question_failed"
        assert event.error_type == "Forbidden"
        assert event.first_message_id is None
        assert bot.send_message.await_count == 3
        bot.decline_chat_join_request.assert_not_awaited()
    with_db(tmp_path, scenario)


def test_captcha_delivery_failure_is_not_misreported_as_first_question_failure(tmp_path):
    async def scenario(db):
        bot = NS(send_message=AsyncMock(side_effect=[TelegramError("captcha offline"), NS(message_id=82)]))
        handler = GatekeeperHandlers(
            settings=replace(settings(), captcha_enabled=True), db=db, scorer=NoopScorer(),
        )
        await handler.on_join_request(make_update(5102), NS(bot=bot))
        event = await get_event(db, 5102)
        candidate = await db.get_candidate(event.candidate_id)
        assert event.status == "captcha_delivery_failed"
        assert event.error_type == "TelegramError"
        assert event.alert_required is True
        assert candidate.status == CandidateStatus.PENDING_ADMIN.value
        assert candidate.stage == 0
        assert event.first_message_id is None
    with_db(tmp_path, scenario)


def test_first_question_forbidden_is_manual_not_expired_and_admin_notified(tmp_path):
    async def scenario(db):
        send = AsyncMock(side_effect=[TelegramError("bot can't initiate conversation"), NS(message_id=82)])
        bot = NS(send_message=send)
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await handler.on_join_request(make_update(5002), NS(bot=bot))
        e = await get_event(db, 5002)
        c = await db.get_candidate(e.candidate_id)
        assert c.status == CandidateStatus.PENDING_ADMIN.value
        assert "Не удалось начать ЛС" in c.decision_reason
        assert e.status == "first_question_failed"
        assert e.error_type == "TelegramError"
        assert e.admin_alerted_at is not None
        assert await handler.expire_stale_interviews(bot) == 0
        summary, pages = build_status_report([c], [])
        assert "ТЕХНИЧЕСКАЯ ПРОБЛЕМА" in "\n".join(pages)
    with_db(tmp_path, scenario)


def test_admin_notification_retried_after_first_delivery_failure(tmp_path):
    async def scenario(db):
        bot = NS(send_message=AsyncMock(side_effect=TelegramError("Bot blocked")))
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await h.on_join_request(make_update(5003), NS(bot=bot))
        e = await get_event(db, 5003)
        assert e.alert_required and e.admin_alerted_at is None
        retry_bot = NS(send_message=AsyncMock(return_value=NS(message_id=34)))
        await h.recover_and_alert_join_requests(retry_bot)
        assert (await get_event(db, 5003)).admin_alerted_at is not None
        assert retry_bot.send_message.await_count == 1
        await h.recover_and_alert_join_requests(retry_bot)
        assert retry_bot.send_message.await_count == 1
    with_db(tmp_path, scenario)


def test_crash_between_candidate_and_send_is_not_auto_declined(tmp_path):
    async def scenario(db):
        when = datetime.now(timezone.utc) - timedelta(minutes=4)
        await db.record_join_request_event(
            update_id=5004, target_chat_id=-1001, user_id=900, request_date=when,
        )
        c = await db.create_candidate(user_id=900, user_chat_id=20900,
            target_chat_id=-1001, username="sample", first_name="X", request_date=when)
        await db.update_join_request_event(5004, "candidate_created", candidate_id=c.id)
        async with db.sessions() as session:
            await session.execute(update(JoinRequestEvent)
                                  .where(JoinRequestEvent.update_id == 5004)
                                  .values(created_at=when))
            await session.commit()
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = NS(send_message=AsyncMock(return_value=NS(message_id=88)),
                 decline_chat_join_request=AsyncMock())
        assert await h.recover_and_alert_join_requests(bot) == 1
        c = await db.get_candidate(c.id)
        assert c.status == CandidateStatus.PENDING_ADMIN.value
        assert "Не подтверждена отправка" in c.decision_reason
        assert (await get_event(db, 5004)).status == "delivery_unconfirmed"
        assert (await get_event(db, 5004)).admin_alerted_at is not None
        assert await h.expire_stale_interviews(bot) == 0
        bot.decline_chat_join_request.assert_not_awaited()
    with_db(tmp_path, scenario)


def test_restart_after_captcha_pass_before_first_question_fails_closed(tmp_path):
    async def scenario(db):
        when = datetime.now(timezone.utc) - timedelta(minutes=4)
        await db.record_join_request_event(
            update_id=5103, target_chat_id=-1001, user_id=903, request_date=when,
        )
        candidate = await db.create_candidate(
            user_id=903, user_chat_id=20903, target_chat_id=-1001,
            username="sample", first_name="X", request_date=when, stage=0,
        )
        await db.update_join_request_event(5103, "captcha_sent", candidate_id=candidate.id)
        await db.set_captcha(candidate.id, 2, "recovery-token")
        outcome, candidate = await db.answer_captcha(
            candidate.id, token="recovery-token", chosen_index=2, max_attempts=2,
        )
        assert outcome == "passed"
        assert (await get_event(db, 5103)).status == "captcha_passed"
        async with db.sessions() as session:
            await session.execute(update(JoinRequestEvent).where(
                JoinRequestEvent.update_id == 5103,
            ).values(created_at=when))
            await session.commit()

        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = NS(send_message=AsyncMock(return_value=NS(message_id=99)),
                 decline_chat_join_request=AsyncMock())
        assert await handler.recover_and_alert_join_requests(bot) == 1
        recovered = await db.get_candidate(candidate.id)
        event = await get_event(db, 5103)
        assert recovered.status == CandidateStatus.PENDING_ADMIN.value
        assert "первого вопроса" in recovered.decision_reason
        assert event.status == "delivery_unconfirmed"
        assert event.admin_alerted_at is not None
        bot.decline_chat_join_request.assert_not_awaited()
    with_db(tmp_path, scenario)


def test_orphan_join_event_is_visible_in_status_and_alerted(tmp_path):
    async def scenario(db):
        when = datetime.now(timezone.utc) - timedelta(minutes=5)
        await db.record_join_request_event(
            update_id=5005, target_chat_id=-1001, user_id=901, request_date=when,
        )
        async with db.sessions() as session:
            await session.execute(update(JoinRequestEvent)
                                  .where(JoinRequestEvent.update_id == 5005)
                                  .values(created_at=when))
            await session.commit()
        bot = NS(send_message=AsyncMock(return_value=NS(message_id=75)))
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        assert await h.recover_and_alert_join_requests(bot) == 1
        orphaned = await db.list_unlinked_join_request_events(-1001)
        assert len(orphaned) == 1
        assert orphaned[0].candidate_id is None
        summary, pages = build_status_report([], [], orphaned)
        assert "Технические события без анкеты: 1" in summary
        assert "U5005" in "\n".join(pages)
        assert "АНКЕТА НЕ СОЗДАНА" in "\n".join(pages)
    with_db(tmp_path, scenario)


def test_recovery_does_not_override_candidate_who_already_replied(tmp_path):
    async def scenario(db):
        when = datetime.now(timezone.utc) - timedelta(minutes=5)
        await db.record_join_request_event(
            update_id=5006, target_chat_id=-1001, user_id=900, request_date=when,
        )
        c = await db.create_candidate(user_id=900, user_chat_id=20900,
                                      target_chat_id=-1001, username="x", first_name="X", request_date=when)
        await db.save_answer(c.id, field="location_answer", answer="Батуми", next_stage=2)
        await db.update_join_request_event(5006, "candidate_created", candidate_id=c.id)
        async with db.sessions() as session:
            await session.execute(update(JoinRequestEvent)
                                  .where(JoinRequestEvent.update_id == 5006)
                                  .values(created_at=when))
            await session.commit()
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = NS(send_message=AsyncMock())
        await h.recover_and_alert_join_requests(bot)
        assert (await db.get_candidate(c.id)).status == CandidateStatus.INTERVIEW.value
        assert (await get_event(db, 5006)).status == "flow_progressed"
        bot.send_message.assert_not_awaited()
    with_db(tmp_path, scenario)
