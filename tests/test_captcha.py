"""Emoji captcha: generator rules, SQL state machine and handler flow."""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import texts
from app.captcha import CATEGORIES, make_captcha
from app.config import Settings
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus


# ---------------------------------------------------------------- generator

def test_every_emoji_belongs_to_exactly_one_category():
    seen = {}
    for name, emojis in CATEGORIES.items():
        for emoji in emojis:
            assert emoji not in seen, f"{emoji} in both {seen.get(emoji)} and {name}"
            seen[emoji] = name
    assert all("\ufe0f" not in e for e in seen), "avoid variation selectors"
    assert len(CATEGORIES) >= 10


def test_captcha_has_exactly_one_correct_option_and_random_layout():
    rng = random.Random(1234)
    positions = set()
    prompts = set()
    for _ in range(500):
        c = make_captcha(rng)
        assert len(c.options) == 4 and len(set(c.options)) == 4
        fitting = [e for e in c.options if e in CATEGORIES[c.prompt]]
        assert fitting == [c.options[c.correct_index]]
        positions.add(c.correct_index)
        prompts.add(c.prompt)
    assert positions == {0, 1, 2, 3}          # answer position is shuffled
    assert len(prompts) == len(CATEGORIES)    # every category gets used


def test_options_count_is_validated():
    with pytest.raises(ValueError):
        make_captcha(options_count=1)


# --------------------------------------------------------------- handlers

def _settings(**kw) -> Settings:
    base = dict(
        bot_token="t", target_chat_id=-100, admin_chat_id=-200, admin_user_ids={1},
        database_url="sqlite+aiosqlite:///:memory:", llm_base_url="https://x.invalid/v1",
        llm_api_key="k", llm_model="m", llm_timeout_seconds=1,
        auto_reject_spam_score=85, auto_reject_core_score_max=15,
        auto_reject_min_ai_confidence=80, max_answer_length=2500,
        captcha_enabled=True,
    )
    base.update(kw)
    return Settings(**base)


def _candidate(**kw) -> Candidate:
    values = dict(id=5, user_id=111, user_chat_id=777, target_chat_id=-100,
                  username="c", first_name="C", status=CandidateStatus.INTERVIEW.value, stage=0)
    values.update(kw)
    return Candidate(**values)


def _query(user_id=111, data="captcha:5:aaaaaaaa:1"):
    return SimpleNamespace(
        data=data, from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(), edit_message_text=AsyncMock(),
    )


def _handlers(db):
    return GatekeeperHandlers(settings=_settings(), db=db, scorer=SimpleNamespace())


def _ctx(bot):
    return SimpleNamespace(bot=bot)


def test_pass_sends_first_question():
    async def scenario():
        cand = _candidate(stage=1)
        db = SimpleNamespace(
            get_candidate=AsyncMock(return_value=_candidate()),
            answer_captcha=AsyncMock(return_value=("passed", cand)),
            confirm_captcha_delivery=AsyncMock(return_value=True),
        )
        bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=456)))
        query = _query()
        await _handlers(db).on_captcha(SimpleNamespace(callback_query=query), _ctx(bot))
        bot.send_message.assert_awaited_once()
        assert texts.QUESTION_LOCATION in bot.send_message.await_args.args[1]
        query.edit_message_text.assert_awaited_once()
        db.confirm_captcha_delivery.assert_awaited_once_with(
            cand.id, "first_question", first_message_id=456,
        )
    asyncio.run(scenario())


def test_other_user_cannot_press_applicants_button():
    async def scenario():
        db = SimpleNamespace(
            get_candidate=AsyncMock(return_value=_candidate()),
            answer_captcha=AsyncMock(),
        )
        bot = SimpleNamespace(send_message=AsyncMock())
        await _handlers(db).on_captcha(
            SimpleNamespace(callback_query=_query(user_id=999)), _ctx(bot))
        db.answer_captcha.assert_not_awaited()
        bot.send_message.assert_not_awaited()
    asyncio.run(scenario())


def test_failure_declines_request_and_expires_candidate():
    async def scenario():
        failed = _candidate(captcha_attempts=2)
        db = SimpleNamespace(
            get_candidate=AsyncMock(return_value=_candidate()),
            answer_captcha=AsyncMock(return_value=("failed", failed)),
            finish_captcha_action=AsyncMock(return_value=failed),
        )
        bot = SimpleNamespace(
            send_message=AsyncMock(), decline_chat_join_request=AsyncMock())
        await _handlers(db).on_captcha(SimpleNamespace(callback_query=_query()), _ctx(bot))
        bot.decline_chat_join_request.assert_awaited_once_with(chat_id=-100, user_id=111)
        assert db.finish_captcha_action.await_args.args[1] == "decline_failed"
        assert db.finish_captcha_action.await_args.kwargs["expired"] is True
        bot.send_message.assert_not_awaited()      # no questionnaire, no LLM
    asyncio.run(scenario())


def test_private_text_during_captcha_gets_reminder_not_scoring():
    async def scenario():
        h = _handlers(SimpleNamespace(set_user_chat_id=AsyncMock()))
        reply = AsyncMock()
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=777),
            effective_message=SimpleNamespace(text="Батуми", reply_text=reply),
        )
        result = await h._handle_private_interview_answer(update, _candidate(user_chat_id=777))
        assert result is None
        reply.assert_awaited_once_with(texts.CAPTCHA_REMINDER)
    asyncio.run(scenario())


def test_retry_delivery_failure_queues_manual_review_and_notice():
    from telegram.error import TelegramError

    async def scenario():
        cand = _candidate(captcha_attempts=1, captcha_pending_action="retry_delivery")
        reviewed = _candidate(status=CandidateStatus.PENDING_ADMIN.value,
                              decision_reason="delivery uncertain", captcha_notice_pending=True)
        db = SimpleNamespace(
            get_candidate=AsyncMock(return_value=cand),
            answer_captcha=AsyncMock(return_value=("retry", cand)),
            set_captcha=AsyncMock(return_value=cand),
            finish_captcha_action=AsyncMock(return_value=reviewed),
            confirm_captcha_delivery=AsyncMock(), mark_captcha_notice_sent=AsyncMock(),
        )
        bot = SimpleNamespace(send_message=AsyncMock(side_effect=TelegramError("offline")),
                              decline_chat_join_request=AsyncMock())
        query = _query()
        query.edit_message_text.side_effect = TelegramError("offline")
        h = _handlers(db)
        h._send_admin_card = AsyncMock(side_effect=TelegramError("admin offline"))
        await h.on_captcha(SimpleNamespace(callback_query=query), _ctx(bot))
        assert db.finish_captcha_action.await_args.args == (cand.id, "retry_delivery")
        assert db.finish_captcha_action.await_args.kwargs["error_type"] == "TelegramError"
        db.confirm_captcha_delivery.assert_not_awaited()
        db.mark_captcha_notice_sent.assert_not_awaited()
        bot.decline_chat_join_request.assert_not_awaited()
    asyncio.run(scenario())


def test_first_question_delivery_failure_uses_durable_notice():
    from telegram.error import TelegramError

    async def scenario():
        cand = _candidate(stage=1, captcha_pending_action="first_question")
        reviewed = _candidate(status=CandidateStatus.PENDING_ADMIN.value,
                              decision_reason="delivery uncertain", captcha_notice_pending=True)
        db = SimpleNamespace(get_candidate=AsyncMock(return_value=cand),
            answer_captcha=AsyncMock(return_value=("passed", cand)),
            finish_captcha_action=AsyncMock(return_value=reviewed),
            confirm_captcha_delivery=AsyncMock(), mark_captcha_notice_sent=AsyncMock())
        bot = SimpleNamespace(send_message=AsyncMock(side_effect=TelegramError("offline")))
        h = _handlers(db)
        h._send_admin_card = AsyncMock()
        await h.on_captcha(SimpleNamespace(callback_query=_query()), _ctx(bot))
        assert db.finish_captcha_action.await_args.args == (cand.id, "first_question")
        assert db.finish_captcha_action.await_args.kwargs["error_type"] == "TelegramError"
        db.mark_captcha_notice_sent.assert_awaited_once_with(cand.id)
        db.confirm_captcha_delivery.assert_not_awaited()
    asyncio.run(scenario())


def test_decline_failure_passes_telegram_error_type_to_manual_review():
    from telegram.error import TelegramError

    async def scenario():
        failed = _candidate(captcha_attempts=2, captcha_pending_action="decline_failed")
        reviewed = _candidate(
            status=CandidateStatus.PENDING_ADMIN.value,
            decision_reason="decline uncertain", captcha_notice_pending=True,
        )
        db = SimpleNamespace(
            get_candidate=AsyncMock(return_value=_candidate()),
            answer_captcha=AsyncMock(return_value=("failed", failed)),
            finish_captcha_action=AsyncMock(return_value=reviewed),
            mark_captcha_notice_sent=AsyncMock(),
        )
        bot = SimpleNamespace(
            send_message=AsyncMock(),
            decline_chat_join_request=AsyncMock(side_effect=TelegramError("decline unavailable")),
        )
        handler = _handlers(db)
        handler._send_admin_card = AsyncMock()
        await handler.on_captcha(SimpleNamespace(callback_query=_query()), _ctx(bot))
        assert db.finish_captcha_action.await_args.args == (failed.id, "decline_failed")
        assert db.finish_captcha_action.await_args.kwargs["error_type"] == "TelegramError"
        db.mark_captcha_notice_sent.assert_awaited_once_with(failed.id)
    asyncio.run(scenario())


def test_interrupted_decline_not_replayed_and_notice_retried():
    from telegram.error import TelegramError

    async def scenario():
        cand = _candidate(captcha_attempts=2, captcha_pending_action="decline_failed")
        reviewed = _candidate(status=CandidateStatus.PENDING_ADMIN.value,
                              decision_reason="uncertain", captcha_notice_pending=True)
        db = SimpleNamespace(list_captcha_recovery=AsyncMock(return_value=[cand]),
            get_candidate=AsyncMock(side_effect=[cand, reviewed, reviewed]),
            finish_captcha_action=AsyncMock(return_value=reviewed),
            list_captcha_notices=AsyncMock(return_value=[reviewed]),
            mark_captcha_notice_sent=AsyncMock())
        bot = SimpleNamespace(decline_chat_join_request=AsyncMock())
        h = _handlers(db)
        h._send_admin_card = AsyncMock(side_effect=[TelegramError("offline"), None])
        assert await h.recover_captcha_actions(bot) == 1
        db.mark_captcha_notice_sent.assert_not_awaited()
        db.list_captcha_recovery.return_value = []
        assert await h.recover_captcha_actions(bot) == 0
        db.mark_captcha_notice_sent.assert_awaited_once_with(cand.id)
        bot.decline_chat_join_request.assert_not_awaited()
    asyncio.run(scenario())


def test_expired_correct_callback_never_sends_question():
    async def scenario():
        cand = _candidate(captcha_pending_action="decline_timeout")
        db = SimpleNamespace(get_candidate=AsyncMock(return_value=cand),
            answer_captcha=AsyncMock(return_value=("expired", cand)),
            finish_captcha_action=AsyncMock(return_value=cand))
        bot = SimpleNamespace(decline_chat_join_request=AsyncMock(return_value=True), send_message=AsyncMock())
        await _handlers(db).on_captcha(SimpleNamespace(callback_query=_query()), _ctx(bot))
        assert db.finish_captcha_action.await_args.args == (cand.id, "decline_timeout")
        bot.send_message.assert_not_awaited()
    asyncio.run(scenario())
