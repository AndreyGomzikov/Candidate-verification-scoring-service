"""Regression cases for HIDE_REQUESTER_MISSING and subsequent join attempts.

All Telegram interactions are mocked; no real candidate is affected.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from telegram.error import TelegramError

from app.handlers import GatekeeperHandlers, _join_request_missing
from app.models import Candidate, CandidateStatus
from tests.test_handlers import NoopScorer, candidate, settings


def _db_for_expiry(item):
    updates = []

    async def transition_status_if(candidate_id, status, *, reason=None, **_kwargs):
        assert candidate_id == item.id
        updates.append((status, reason))
        item.status = status.value
        item.decision_reason = reason
        return item

    return NS(list_expired_interviews=AsyncMock(return_value=[item]),
              get_candidate=AsyncMock(return_value=item),
              get_active_candidate_for_chat=AsyncMock(return_value=item),
              transition_status_if=AsyncMock(side_effect=transition_status_if),
              updates=updates)


def _expire_with(exc=None):
    item = candidate(status=CandidateStatus.INTERVIEW.value, stage=1,
                     location_answer=None, business_answer=None, goal_answer=None,
                     updated_at=datetime.now(timezone.utc) - timedelta(hours=2))
    db = _db_for_expiry(item)
    bot = NS(decline_chat_join_request=AsyncMock(side_effect=exc),
             send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    result = asyncio.run(handler.expire_stale_interviews(bot))
    return result, item, db, bot


def test_expired_missing_request_closes_without_dead_admin_card():
    count, item, db, bot = _expire_with(TelegramError('Bad Request: Hide_requester_missing'))
    assert count == 1
    assert item.status == CandidateStatus.EXPIRED.value
    assert 'Hide_requester_missing' in item.decision_reason
    assert 'Отклонение не выполнялось' in item.decision_reason
    assert len(db.updates) == 1
    assert db.updates[0][0] == CandidateStatus.EXPIRED
    bot.decline_chat_join_request.assert_awaited_once_with(
        chat_id=item.target_chat_id, user_id=item.user_id,
    )
    assert not any(c.kwargs.get("chat_id", c.args[0] if c.args else None) == settings().admin_chat_id for c in bot.send_message.await_args_list)


@pytest.mark.parametrize('error', [
    TelegramError('Bad Request: chat_admin_required'),
    TelegramError('httpx.ConnectError: Temporary failure'),
    TelegramError('RetryAfter: too many requests'),
])
def test_expiry_other_telegram_failure_still_goes_to_human(error):
    count, item, db, bot = _expire_with(error)
    assert count == 0
    assert item.status == CandidateStatus.PENDING_ADMIN.value
    assert db.updates[-1][0] == CandidateStatus.PENDING_ADMIN
    assert any(c.kwargs.get("chat_id", c.args[0] if c.args else None) == settings().admin_chat_id for c in bot.send_message.await_args_list)


def test_expiry_success_preserves_previous_behavior():
    count, item, db, bot = _expire_with()
    assert count == 1
    assert item.status == CandidateStatus.EXPIRED.value
    assert db.updates[-1][0] == CandidateStatus.EXPIRED
    assert any(c.kwargs.get("chat_id", c.args[0] if c.args else None) == item.user_chat_id for c in bot.send_message.await_args_list)


@pytest.mark.parametrize('description,expected', [
    ('Hide_requester_missing', True),
    ('Bad Request: HIDE_REQUESTER_MISSING', True),
    ('httpx.ConnectError', False),
    ('CHAT_ADMIN_REQUIRED', False),
    ('USER_ALREADY_PARTICIPANT', False),
])
def test_missing_detection_only_matches_expected_telegram_error(description,expected):
    assert _join_request_missing(TelegramError(description)) is expected


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_admin_old_card_missing_request_is_closed_without_fake_rejection(action):
    item = candidate(status=CandidateStatus.PENDING_ADMIN.value, stage=1)
    async def transition_status_if(candidate_id, status, *, reason=None, **_kw):
        assert candidate_id == item.id
        item.status, item.decision_reason = status.value, reason
        return item
    db = NS(get_candidate=AsyncMock(return_value=item),
            transition_status_if=AsyncMock(side_effect=transition_status_if))
    bot = NS(decline_chat_join_request=AsyncMock(
                 side_effect=TelegramError('Hide_requester_missing') if action == 'reject' else None),
             approve_chat_join_request=AsyncMock(
                 side_effect=TelegramError('Hide_requester_missing') if action == 'approve' else None),
             send_message=AsyncMock())
    query = NS(data=f'candidate:{action}:1', message=NS(chat=NS(id=-2002)),
               answer=AsyncMock(), edit_message_reply_markup=AsyncMock())
    update = NS(callback_query=query, effective_user=NS(id=42, full_name='Admin'))
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_admin_decision(update, NS(bot=bot)))
    assert item.status == CandidateStatus.REQUEST_MISSING.value
    assert 'Hide_requester_missing' in item.decision_reason
    db.transition_status_if.assert_awaited_once()
    assert bot.approve_chat_join_request.await_count == (1 if action == "approve" else 0)
    assert bot.decline_chat_join_request.await_count == (1 if action == "reject" else 0)
    bot.send_message.assert_not_awaited()  # no fake approval/rejection notification
    query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)
    assert query.answer.await_args.kwargs['show_alert'] is True


def _join_update(item, date):
    return NS(chat_join_request=NS(
        chat=NS(id=item.target_chat_id),
        from_user=NS(id=item.user_id, username='candidate',first_name='Test'),
        user_chat_id=item.user_chat_id + 100,
        date=date,
    ))


def _db_for_join(old):
    new = candidate(id=2, user_chat_id=old.user_chat_id + 100,
                    status=CandidateStatus.INTERVIEW.value, stage=1)
    return NS(get_active_candidate_for_chat=AsyncMock(return_value=old),
              get_latest_candidate_for_chat=AsyncMock(return_value=old),
              last_cooldown_declined_request_date=AsyncMock(return_value=None),
              set_user_chat_id=AsyncMock(return_value=old),
              get_recent_rejected_candidate=AsyncMock(return_value=None),
              create_candidate=AsyncMock(return_value=new))


def test_new_request_after_stuck_pending_admin_starts_new_interview():
    old = candidate(id=1, status=CandidateStatus.PENDING_ADMIN.value,
                    created_at=datetime.now(timezone.utc) - timedelta(hours=1),
                    request_date=datetime.now(timezone.utc) - timedelta(hours=1))
    db = _db_for_join(old)
    bot = NS(send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(old, datetime.now(timezone.utc)), NS(bot=bot)
    ))
    db.create_candidate.assert_awaited_once()
    db.set_user_chat_id.assert_not_awaited()
    assert any('1/3.' in c.kwargs.get('text','') for c in bot.send_message.await_args_list)


def test_redelivered_old_join_request_does_not_create_second_interview():
    created = datetime.now(timezone.utc) - timedelta(hours=1)
    old = candidate(id=1, status=CandidateStatus.PENDING_ADMIN.value,
                    created_at=created, request_date=created - timedelta(seconds=2))
    db = _db_for_join(old)
    bot = NS(send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(old, created - timedelta(seconds=2)), NS(bot=bot)
    ))
    db.create_candidate.assert_not_awaited()
    db.set_user_chat_id.assert_awaited_once()
    assert any('администраторам' in c.args[1] for c in bot.send_message.await_args_list)


def test_new_join_during_scoring_supersedes_old_ai_run():
    old = candidate(id=1, status=CandidateStatus.SCORING.value,
                    stage=4,
                    created_at=datetime.now(timezone.utc) - timedelta(hours=1),
                    request_date=datetime.now(timezone.utc) - timedelta(hours=1))
    db = _db_for_join(old)
    bot = NS(send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(old, datetime.now(timezone.utc)), NS(bot=bot)
    ))
    db.create_candidate.assert_awaited_once()
    db.set_user_chat_id.assert_not_awaited()


def test_late_redelivery_does_not_supersede_original_request():
    """Telegram may retry a weeks-old update after candidate was created."""
    old_date = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    old = candidate(id=1, status=CandidateStatus.PENDING_ADMIN.value,
                    created_at=old_date + timedelta(seconds=3), request_date=old_date)
    db = _db_for_join(old)
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(_join_update(old, old_date),
                                        NS(bot=NS(send_message=AsyncMock()))))
    db.create_candidate.assert_not_awaited()


def test_same_second_reapplication_is_conservatively_not_replaced():
    """Telegram does not identify attempts inside one timestamp second."""
    old_date = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    old = candidate(id=1, status=CandidateStatus.PENDING_ADMIN.value,
                    created_at=old_date + timedelta(milliseconds=150),
                    request_date=old_date)
    db = _db_for_join(old)
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(_join_update(old, old_date),
                                        NS(bot=NS(send_message=AsyncMock()))))
    db.create_candidate.assert_not_awaited()


def test_new_request_in_next_second_supersedes_original():
    old_date = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    old = candidate(id=1, status=CandidateStatus.PENDING_ADMIN.value,
                    created_at=old_date + timedelta(seconds=2), request_date=old_date)
    db = _db_for_join(old)
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(old, old_date + timedelta(seconds=1)),
        NS(bot=NS(send_message=AsyncMock()))))
    db.create_candidate.assert_awaited_once()
    assert db.create_candidate.await_args.kwargs['request_date'] == old_date + timedelta(seconds=1)


def test_old_update_after_new_application_does_not_overwrite_dm_chat():
    newer_date = datetime(2026, 11, 1, 12, 0, 2, tzinfo=timezone.utc)
    active = candidate(id=2, status=CandidateStatus.INTERVIEW.value,
                       request_date=newer_date)
    db = _db_for_join(active)
    bot = NS(send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(active, newer_date - timedelta(seconds=2)), NS(bot=bot)))
    db.set_user_chat_id.assert_not_awaited()
    db.create_candidate.assert_not_awaited()
    bot.send_message.assert_not_awaited()

@pytest.mark.parametrize('terminal_status', [
    CandidateStatus.APPROVED.value,
    CandidateStatus.REJECTED.value,
    CandidateStatus.EXPIRED.value,
    CandidateStatus.REQUEST_MISSING.value,
    CandidateStatus.SUPERSEDED.value,
])
@pytest.mark.parametrize('offset', [-10, 0])
def test_old_join_update_after_terminal_status_is_ignored(terminal_status, offset):
    """A delayed Telegram update must not recreate a completed application."""
    timestamp = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    old = candidate(id=77, status=terminal_status, request_date=timestamp)
    db = NS(
        get_active_candidate_for_chat=AsyncMock(return_value=None),
        get_latest_candidate_for_chat=AsyncMock(return_value=old),
              last_cooldown_declined_request_date=AsyncMock(return_value=None),
        get_recent_rejected_candidate=AsyncMock(return_value=None),
        create_candidate=AsyncMock(),
    )
    bot = NS(send_message=AsyncMock(), decline_chat_join_request=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(old, timestamp + timedelta(seconds=offset)), NS(bot=bot)
    ))
    db.create_candidate.assert_not_awaited()
    db.get_recent_rejected_candidate.assert_not_awaited()
    bot.send_message.assert_not_awaited()
    bot.decline_chat_join_request.assert_not_awaited()


def test_new_join_after_completed_application_creates_new_interview():
    timestamp = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    old = candidate(id=77, status=CandidateStatus.APPROVED.value,
                    request_date=timestamp)
    fresh = candidate(id=78, status=CandidateStatus.INTERVIEW.value,
                      request_date=timestamp + timedelta(seconds=1))
    db = NS(
        get_active_candidate_for_chat=AsyncMock(return_value=None),
        get_latest_candidate_for_chat=AsyncMock(return_value=old),
              last_cooldown_declined_request_date=AsyncMock(return_value=None),
        get_recent_rejected_candidate=AsyncMock(return_value=None),
        create_candidate=AsyncMock(return_value=fresh),
    )
    bot = NS(send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(old, timestamp + timedelta(seconds=1)), NS(bot=bot)
    ))
    db.create_candidate.assert_awaited_once()
    assert db.create_candidate.await_args.kwargs['request_date'] == timestamp + timedelta(seconds=1)
    bot.send_message.assert_awaited()


def test_legacy_completed_application_allows_new_request():
    """Do not mistake local DB insertion time for Telegram's request date."""
    old = candidate(id=77, status=CandidateStatus.EXPIRED.value,
                    request_date=None)
    fresh = candidate(id=78, status=CandidateStatus.INTERVIEW.value)
    db = NS(
        get_active_candidate_for_chat=AsyncMock(return_value=None),
        get_latest_candidate_for_chat=AsyncMock(return_value=old),
              last_cooldown_declined_request_date=AsyncMock(return_value=None),
        get_recent_rejected_candidate=AsyncMock(return_value=None),
        create_candidate=AsyncMock(return_value=fresh),
    )
    bot = NS(send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
    asyncio.run(handler.on_join_request(
        _join_update(old, datetime.now(timezone.utc)), NS(bot=bot)
    ))
    db.create_candidate.assert_awaited_once()


# Cooldown declines are not candidate rows; the replay ledger must survive
# expiry of the original cooldown, Telegram retries, and bot restarts.
def test_cooldown_declined_update_cannot_reopen_after_cooldown_expires():
    async def scenario():
        old_at = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)
        refused_at = old_at + timedelta(minutes=1)
        old = candidate(id=51, status=CandidateStatus.REJECTED.value,
                        request_date=old_at)
        new = candidate(id=52, status=CandidateStatus.INTERVIEW.value,
                        request_date=refused_at + timedelta(minutes=10))
        ledger = {}

        async def last_refused(user_id, chat_id):
            item = ledger.get((chat_id, user_id))
            return item[0] if item is not None else None

        async def save_intent(user_id, chat_id, request_date):
            key = (chat_id, user_id)
            if key in ledger and ledger[key][0] >= request_date:
                return False
            ledger[key] = (request_date, "pending")
            return True

        async def complete_intent(user_id, chat_id, request_date, *, confirmed):
            ledger[(chat_id, user_id)] = (
                request_date, "declined" if confirmed else "uncertain"
            )
            return True

        db = NS(
            get_active_candidate_for_chat=AsyncMock(return_value=None),
            get_latest_candidate_for_chat=AsyncMock(return_value=old),
            last_cooldown_declined_request_date=AsyncMock(side_effect=last_refused),
            begin_cooldown_decline=AsyncMock(side_effect=save_intent),
            complete_cooldown_decline=AsyncMock(side_effect=complete_intent),
            get_recent_rejected_candidate=AsyncMock(return_value=old),
            create_candidate=AsyncMock(return_value=new),
        )
        bot = NS(decline_chat_join_request=AsyncMock(), send_message=AsyncMock())
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        update = _join_update(old, refused_at)

        # First delivery is declined by Telegram and saved independently of candidates.
        await handler.on_join_request(update, NS(bot=bot))
        bot.decline_chat_join_request.assert_awaited_once_with(
            chat_id=old.target_chat_id, user_id=old.user_id,
        )
        db.begin_cooldown_decline.assert_awaited_once_with(
            old.user_id, old.target_chat_id, refused_at,
        )
        db.complete_cooldown_decline.assert_awaited_once_with(
            old.user_id, old.target_chat_id, refused_at, confirmed=True,
        )
        db.create_candidate.assert_not_awaited()
        assert ledger[(old.target_chat_id, old.user_id)] == (refused_at, "declined")

        # After cooldown expires a fresh process sees the durable ledger.
        db.get_recent_rejected_candidate.return_value = None
        second_handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await second_handler.on_join_request(update, NS(bot=bot))
        await second_handler.on_join_request(
            _join_update(old, refused_at - timedelta(seconds=1)), NS(bot=bot)
        )
        assert bot.decline_chat_join_request.await_count == 1
        assert bot.send_message.await_count == 1  # Only original cooldown notice
        db.create_candidate.assert_not_awaited()

        # Legitimate strictly newer request is accepted after cooldown.
        newer = refused_at + timedelta(minutes=10)
        await second_handler.on_join_request(_join_update(old, newer), NS(bot=bot))
        db.create_candidate.assert_awaited_once()
        assert db.create_candidate.await_args.kwargs['request_date'] == newer

    asyncio.run(scenario())


def test_failed_cooldown_decline_is_retained_as_uncertain():
    async def scenario():
        when = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)
        old = candidate(status=CandidateStatus.REJECTED.value,
                        request_date=when - timedelta(seconds=1))
        db = NS(
            get_active_candidate_for_chat=AsyncMock(return_value=None),
            get_latest_candidate_for_chat=AsyncMock(return_value=old),
            last_cooldown_declined_request_date=AsyncMock(return_value=None),
            get_recent_rejected_candidate=AsyncMock(return_value=old),
            begin_cooldown_decline=AsyncMock(return_value=True),
            complete_cooldown_decline=AsyncMock(return_value=True),
            create_candidate=AsyncMock(),
        )
        bot = NS(decline_chat_join_request=AsyncMock(
                     side_effect=TelegramError('chat_admin_required')),
                 send_message=AsyncMock())
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await handler.on_join_request(_join_update(old, when), NS(bot=bot))
        db.begin_cooldown_decline.assert_awaited_once_with(
            old.user_id, old.target_chat_id, when,
        )
        db.complete_cooldown_decline.assert_awaited_once_with(
            old.user_id, old.target_chat_id, when, confirmed=False,
        )
        db.create_candidate.assert_not_awaited()
        bot.send_message.assert_not_awaited()

    asyncio.run(scenario())


def test_cooldown_marker_saved_even_if_notification_fails():
    async def scenario():
        when = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)
        old = candidate(status=CandidateStatus.REJECTED.value,
                        request_date=when - timedelta(seconds=1))
        db = NS(
            get_active_candidate_for_chat=AsyncMock(return_value=None),
            get_latest_candidate_for_chat=AsyncMock(return_value=old),
            last_cooldown_declined_request_date=AsyncMock(return_value=None),
            get_recent_rejected_candidate=AsyncMock(return_value=old),
            begin_cooldown_decline=AsyncMock(return_value=True),
            complete_cooldown_decline=AsyncMock(return_value=True),
            create_candidate=AsyncMock(),
        )
        bot = NS(decline_chat_join_request=AsyncMock(),
                 send_message=AsyncMock(side_effect=TelegramError('DM unavailable')))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await handler.on_join_request(_join_update(old, when), NS(bot=bot))
        db.begin_cooldown_decline.assert_awaited_once_with(
            old.user_id, old.target_chat_id, when,
        )
        db.complete_cooldown_decline.assert_awaited_once_with(
            old.user_id, old.target_chat_id, when, confirmed=True,
        )
        db.create_candidate.assert_not_awaited()

    asyncio.run(scenario())


def test_cooldown_intent_is_committed_before_telegram_decline():
    """Assert the API is never called before the durable DB write."""
    async def scenario():
        when = datetime(2026, 11, 6, 11, tzinfo=timezone.utc)
        old = candidate(status=CandidateStatus.REJECTED.value,
                        request_date=when - timedelta(seconds=1))
        order = []
        async def begin(*args):
            order.append("begin")
            return True
        async def decline(**kwargs):
            order.append("telegram")
        async def complete(*args, **kwargs):
            order.append("complete")
            return True
        db = NS(
            get_active_candidate_for_chat=AsyncMock(return_value=None),
            get_latest_candidate_for_chat=AsyncMock(return_value=old),
            last_cooldown_declined_request_date=AsyncMock(return_value=None),
            get_recent_rejected_candidate=AsyncMock(return_value=old),
            begin_cooldown_decline=AsyncMock(side_effect=begin),
            complete_cooldown_decline=AsyncMock(side_effect=complete),
            create_candidate=AsyncMock(),
        )
        bot = NS(decline_chat_join_request=AsyncMock(side_effect=decline),
                 send_message=AsyncMock())
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await h.on_join_request(_join_update(old, when), NS(bot=bot))
        assert order == ["begin", "telegram", "complete"]
    asyncio.run(scenario())


def test_cooldown_crash_after_intent_does_not_reopen_after_expiry():
    """A simulated crash after the DB commit must preserve replay protection."""
    async def scenario():
        when = datetime(2026, 11, 6, 11, tzinfo=timezone.utc)
        old = candidate(status=CandidateStatus.REJECTED.value,
                        request_date=when - timedelta(seconds=1))
        attempts = set()
        async def last(_user, _chat):
            return max(attempts) if attempts else None
        async def begin(_user, _chat, date):
            assert date not in attempts
            attempts.add(date)
            return True
        db = NS(
            get_active_candidate_for_chat=AsyncMock(return_value=None),
            get_latest_candidate_for_chat=AsyncMock(return_value=old),
            last_cooldown_declined_request_date=AsyncMock(side_effect=last),
            get_recent_rejected_candidate=AsyncMock(return_value=old),
            begin_cooldown_decline=AsyncMock(side_effect=begin),
            complete_cooldown_decline=AsyncMock(),
            create_candidate=AsyncMock(),
        )
        # This is not a TelegramError; simulate process interruption before
        # the result can be written to the DB.
        bot = NS(decline_chat_join_request=AsyncMock(
            side_effect=RuntimeError("simulated process crash")),
            send_message=AsyncMock())
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        with pytest.raises(RuntimeError, match="simulated process crash"):
            await h.on_join_request(_join_update(old, when), NS(bot=bot))
        assert attempts == {when}
        db.complete_cooldown_decline.assert_not_awaited()

        db.get_recent_rejected_candidate.return_value = None
        bot.decline_chat_join_request.side_effect = None
        restarted = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await restarted.on_join_request(_join_update(old, when), NS(bot=bot))
        db.create_candidate.assert_not_awaited()
        bot.decline_chat_join_request.assert_awaited_once()
    asyncio.run(scenario())


def test_cooldown_db_intent_failure_does_not_call_telegram():
    async def scenario():
        when = datetime(2026, 11, 6, 11, tzinfo=timezone.utc)
        old = candidate(status=CandidateStatus.REJECTED.value,
                        request_date=when - timedelta(seconds=1))
        db = NS(
            get_active_candidate_for_chat=AsyncMock(return_value=None),
            get_latest_candidate_for_chat=AsyncMock(return_value=old),
            last_cooldown_declined_request_date=AsyncMock(return_value=None),
            get_recent_rejected_candidate=AsyncMock(return_value=old),
            begin_cooldown_decline=AsyncMock(side_effect=RuntimeError("db down")),
            complete_cooldown_decline=AsyncMock(),
            create_candidate=AsyncMock(),
        )
        bot = NS(decline_chat_join_request=AsyncMock(), send_message=AsyncMock())
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        with pytest.raises(RuntimeError, match="db down"):
            await h.on_join_request(_join_update(old, when), NS(bot=bot))
        bot.decline_chat_join_request.assert_not_awaited()
        db.create_candidate.assert_not_awaited()
    asyncio.run(scenario())


def test_ambiguous_cooldown_decline_is_not_automatically_retried():
    async def scenario():
        when = datetime(2026, 11, 6, 11, tzinfo=timezone.utc)
        old = candidate(status=CandidateStatus.REJECTED.value,
                        request_date=when - timedelta(seconds=1))
        recorded = []
        async def last(_user, _chat):
            return when if recorded else None
        async def intent(*_args):
            recorded.append("pending")
            return True
        async def result(*_args, **kwargs):
            recorded.append("uncertain" if not kwargs["confirmed"] else "declined")
            return True
        db = NS(
            get_active_candidate_for_chat=AsyncMock(return_value=None),
            get_latest_candidate_for_chat=AsyncMock(return_value=old),
            last_cooldown_declined_request_date=AsyncMock(side_effect=last),
            get_recent_rejected_candidate=AsyncMock(return_value=old),
            begin_cooldown_decline=AsyncMock(side_effect=intent),
            complete_cooldown_decline=AsyncMock(side_effect=result),
            create_candidate=AsyncMock(),
        )
        bot = NS(decline_chat_join_request=AsyncMock(
            side_effect=TelegramError("network timeout")), send_message=AsyncMock())
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await h.on_join_request(_join_update(old, when), NS(bot=bot))
        assert recorded == ["pending", "uncertain"]
        db.get_recent_rejected_candidate.return_value = None
        bot.decline_chat_join_request.side_effect = None
        await GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer()).on_join_request(
            _join_update(old, when), NS(bot=bot)
        )
        bot.decline_chat_join_request.assert_awaited_once()
        db.create_candidate.assert_not_awaited()
    asyncio.run(scenario())


def test_maintenance_notifies_admin_of_uncertain_intent_and_retries_notice():
    async def scenario():
        intent = NS(id=23, target_chat_id=-1001, user_id=123,
                    request_date=datetime(2026, 11, 6, 11, tzinfo=timezone.utc))
        db = NS(
            mark_stale_cooldown_intents=AsyncMock(return_value=1),
            list_unalerted_cooldown_intents=AsyncMock(return_value=[intent]),
            mark_cooldown_intent_alerted=AsyncMock(return_value=True),
        )
        h = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = NS(send_message=AsyncMock(side_effect=TelegramError("telegram offline")))
        assert await h.recover_cooldown_decline_intents(bot) == 1
        db.mark_cooldown_intent_alerted.assert_not_awaited()
        bot.send_message.side_effect = None
        assert await h.recover_cooldown_decline_intents(bot) == 1
        db.mark_cooldown_intent_alerted.assert_awaited_once_with(23)
        assert 'НЕ повторяет отклонение' in bot.send_message.await_args.kwargs['text']
    asyncio.run(scenario())
