"""Full-period /status: privacy, incomplete answers, history and Telegram size."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.db import Database
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus, CooldownDeclinedJoinRequest
from app.status_report import build_status_report, _utf16_length
from tests.test_handlers import settings, NoopScorer


def make_candidate(id, **overrides):
    values = dict(
        id=id, user_id=1000+id, target_chat_id=-1001, user_chat_id=1000+id,
        username=f"candidate{id}", first_name="Тест", stage=4,
        status="approved", created_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
        location_answer="Грузия", business_answer="Кафе", goal_answer="Партнёры",
        confidence_score=82, ai_confidence=91,
    )
    values.update(overrides)
    return Candidate(**values)


def test_report_distinguishes_decisions_and_unknown_score():
    approved = make_candidate(1, status="approved", confidence_score=0)
    expired = make_candidate(
        2, status="expired", stage=1, confidence_score=None,
        location_answer=None, business_answer=None, goal_answer=None,
    )
    pending = make_candidate(3, status="pending_admin", confidence_score=25)
    summary, pages = build_status_report([approved, expired, pending], [])
    report = "\n\n".join(pages)
    assert "ПРОШЁЛ: 1 | НЕ ПРОШЁЛ: 0" in summary
    assert "Баллы: 0/100" in report
    assert "Анкета не завершена. Баллы: не рассчитывались" in report
    assert "№2" in report
    assert "ОЖИДАЕТ РЕШЕНИЯ" in report
    assert "Ответы:\n1) Локация: —\n2) Бизнес: —\n3) Цель вступления: —" in report
    assert report.index("№1") < report.index("№2") < report.index("№3")


def test_each_answer_has_own_labeled_line_for_completed_and_partial_forms():
    completed = make_candidate(
        29,
        location_answer="Я в Батуми",
        business_answer="Я работаю менеджером в IT-компании",
        goal_answer="Общаться и знакомиться",
    )
    partial = make_candidate(
        38,
        status="expired",
        stage=2,
        confidence_score=None,
        location_answer="Москва",
        business_answer=None,
        goal_answer="",
    )
    _, pages = build_status_report([completed, partial], [])
    text = "\n\n".join(pages)
    assert "Ответы:\n1) Локация: Я в Батуми\n2) Бизнес: Я работаю менеджером в IT-компании\n3) Цель вступления: Общаться и знакомиться" in text
    assert "Ответы:\n1) Локация: Москва\n2) Бизнес: —\n3) Цель вступления: —" in text
    assert "1) —; 2) —; 3) —" not in text
    assert "Итог: Анкета не завершена. Баллы: не рассчитывались" in text


def test_report_every_attempt_including_cooldown_and_nonbmp_characters():
    moment = datetime(2026, 11, 5, tzinfo=timezone.utc)
    candidates = [make_candidate(
        index, created_at=moment + timedelta(seconds=index),
        location_answer="😀" * 700, first_name="😀" * 100,
        decision_reason="Причина " * 150,
    ) for index in range(1, 85)]
    cooldown = CooldownDeclinedJoinRequest(
        id=9, user_id=444, target_chat_id=-1001,
        request_date=moment + timedelta(days=1),
        status="declined", confirmed_at=moment + timedelta(days=1),
    )
    summary, pages = build_status_report(candidates, [cooldown])
    assert len(pages) > 2
    assert "Повторных попыток по cooldown: 1" in summary
    assert all(_utf16_length(page) <= 3800 for page in pages)
    combined = "\n\n".join(pages)
    for index in range(1, 85):
        assert combined.count(f"№{index} |")==1
    assert combined.count("Повторная попытка C9") == 1
    assert "Ответы: не запрашивались" in combined


def test_db_status_history_only_target_group_and_keeps_attempts(tmp_path):
    import pytest
    pytest.importorskip("aiosqlite")
    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'status.sqlite3'}")
        try:
            await db.init()
            first = await db.create_candidate(
                user_id=1, user_chat_id=10, target_chat_id=-1001,
                username="one", first_name="One",
            )
            await db.save_answer(first.id, field="location_answer", answer="Батуми", next_stage=2)
            await db.set_status(first.id, CandidateStatus.APPROVED)
            second = await db.create_candidate(
                user_id=1, user_chat_id=11, target_chat_id=-1001,
                username="one", first_name="One",
            )
            await db.create_candidate(
                user_id=2, user_chat_id=12, target_chat_id=-9999,
                username="other", first_name="Other",
            )
            when = datetime(2026, 11, 6, 14, 0, tzinfo=timezone.utc)
            assert await db.begin_cooldown_decline(3, -1001, when)
            assert await db.complete_cooldown_decline(3, -1001, when, confirmed=True)
            rows, cooldowns = await db.get_status_history(-1001)
            assert [c.id for c in rows] == [first.id, second.id]
            assert rows[0].location_answer == "Батуми"
            assert len(cooldowns) == 1
            assert cooldowns[0].user_id == 3
            assert cooldowns[0].status == "declined"
        finally:
            await db.close()
    asyncio.run(scenario())


def test_status_command_dm_admin_only_and_one_message():
    async def scenario():
        db = SimpleNamespace(get_status_history=AsyncMock(return_value=(
            [make_candidate(1)], [],
        )))
        admin_settings = settings()  # explicit ADMIN_USER_IDS = {42}
        handler = GatekeeperHandlers(settings=admin_settings, db=db, scorer=NoopScorer())
        message = SimpleNamespace(reply_text=AsyncMock())
        bot = SimpleNamespace(send_message=AsyncMock(), send_document=AsyncMock(),
                              get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status="administrator")
        ))
        context = SimpleNamespace(bot=bot)
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(type="private", id=42),
            effective_user=SimpleNamespace(id=42),
            effective_message=message,
        )
        await handler.on_status(update, context)
        db.get_status_history.assert_awaited_once_with(-1001)
        assert message.reply_text.await_count == 1
        delivered = message.reply_text.await_args.args[0]
        assert "SOLIDUS — история заявок Период: 05.11.2026 - 05.11.2026" in delivered
        assert "Заявок в анкете: 1" in delivered
        assert "№1" in delivered
        assert _utf16_length(delivered) <= 4096
        bot.send_message.assert_not_awaited()
        bot.send_document.assert_not_awaited()

        message.reply_text.reset_mock()
        bot.send_message.reset_mock()
        update.effective_user.id = 3000
        await handler.on_status(update, context)
        assert "только администраторам" in message.reply_text.await_args.args[0]
        bot.send_message.assert_not_awaited()

        message.reply_text.reset_mock()
        update.effective_user.id = 42
        update.effective_chat.type = "channel"
        await handler.on_status(update, context)
        assert "только в личном чате" in message.reply_text.await_args.args[0]
        bot.send_message.assert_not_awaited()

    asyncio.run(scenario())


def test_status_target_chat_creator_without_allowlist():
    async def scenario():
        s = settings()
        # Frozen dataclass; use replace to remove explicit admin allowlist.
        from dataclasses import replace
        s = replace(s, admin_user_ids=set())
        handler = GatekeeperHandlers(settings=s, db=SimpleNamespace(), scorer=NoopScorer())
        bot = SimpleNamespace(get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status="creator"),
        ))
        allowed = await handler._can_view_status(SimpleNamespace(bot=bot), 214627463)
        assert allowed is True
        bot.get_chat_member.assert_awaited_once_with(-1001, 214627463)
    asyncio.run(scenario())


def test_expired_never_shows_old_scoring_values():
    expired = make_candidate(8, status="expired", confidence_score=87,
                             ai_confidence=91, stage=1)
    _, pages = build_status_report([expired], [])
    text = "\n".join(pages)
    assert "Итог: Анкета не завершена. Баллы: не рассчитывались" in text
    assert "87/100" not in text
    assert "91/100" not in text
    assert text.count("Баллы:") == 1


def test_status_full_is_not_a_separate_command():
    async def scenario():
        db = SimpleNamespace(get_status_history=AsyncMock())
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        message = SimpleNamespace(reply_text=AsyncMock())
        bot = SimpleNamespace(send_message=AsyncMock(), send_document=AsyncMock(),
                              get_chat_member=AsyncMock(return_value=SimpleNamespace(status="administrator")))
        context = SimpleNamespace(bot=bot, args=["full"])
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(type="private", id=42),
            effective_user=SimpleNamespace(id=42), effective_message=message,
        )
        await handler.on_status(update, context)
        assert "/status —" in message.reply_text.await_args.args[0]
        db.get_status_history.assert_not_awaited()
        bot.send_message.assert_not_awaited()
        bot.send_document.assert_not_awaited()
    asyncio.run(scenario())


def test_status_many_candidates_sends_one_document_with_all_records():
    async def scenario():
        candidates = [make_candidate(i, location_answer="😀" * 150,
                       business_answer="Бизнес " * 40,
                       goal_answer="Сотрудничество " * 20)
                      for i in range(1, 150)]
        db = SimpleNamespace(get_status_history=AsyncMock(return_value=(candidates, [])))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = SimpleNamespace(send_document=AsyncMock(), send_message=AsyncMock(),
                              get_chat_member=AsyncMock(return_value=SimpleNamespace(status="administrator")))
        context = SimpleNamespace(bot=bot, args=[])
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(type="private", id=42),
            effective_user=SimpleNamespace(id=42), effective_message=message,
        )
        await handler.on_status(update, context)
        message.reply_text.assert_not_awaited()
        bot.send_message.assert_not_awaited()
        bot.send_document.assert_awaited_once()
        assert bot.send_document.await_args.kwargs["chat_id"] == 42
        doc = bot.send_document.await_args.kwargs["document"]
        assert doc.filename == "solidus-status-all.txt"
        content = doc.input_file_content.decode("utf-8")
        assert "Заявок в анкете: 149" in content
        for i in range(1, 150):
            assert content.count(f"№{i} |") == 1
        assert "Ответы:\n1) Локация:" in content

    asyncio.run(scenario())


def test_status_no_records_sends_empty_notice():
    async def scenario():
        db = SimpleNamespace(get_status_history=AsyncMock(return_value=([], [])))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = SimpleNamespace(send_message=AsyncMock(),
                              get_chat_member=AsyncMock(return_value=SimpleNamespace(status="administrator")))
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(type="private", id=42),
            effective_user=SimpleNamespace(id=42), effective_message=message,
        )
        await handler.on_status(update, SimpleNamespace(bot=bot, args=[]))
        assert message.reply_text.await_count == 1
        combined = message.reply_text.await_args.args[0]
        assert "Заявок в анкете: 0" in combined
        assert "Заявок в базе пока нет" in combined
        bot.send_message.assert_not_awaited()
    asyncio.run(scenario())


def test_overlapping_status_requests_do_not_duplicate_report():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def load(_chat_id):
            entered.set()
            await release.wait()
            return [make_candidate(7)], []

        db = SimpleNamespace(get_status_history=AsyncMock(side_effect=load))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = SimpleNamespace(send_message=AsyncMock(),
                              get_chat_member=AsyncMock(return_value=SimpleNamespace(status="administrator")))
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(type="private", id=42),
            effective_user=SimpleNamespace(id=42), effective_message=message,
        )
        ctx = SimpleNamespace(bot=bot, args=[])
        first = asyncio.create_task(handler.on_status(update, ctx))
        await asyncio.wait_for(entered.wait(), timeout=2)
        await handler.on_status(update, ctx)
        assert "уже отправляется" in message.reply_text.await_args.args[0]
        assert db.get_status_history.await_count == 1
        release.set()
        await first
        assert len(handler._status_running_users) == 0
        assert bot.send_message.await_count == 0
        assert message.reply_text.await_count == 2  # in-progress notice and one full report
        assert "№7" in message.reply_text.await_args.args[0]
    asyncio.run(scenario())


def test_legacy_expired_missing_request_is_not_labeled_incomplete():
    missing = make_candidate(31, status="expired", confidence_score=None,
                             decision_reason="Заявка отсутствует в Telegram (Hide_requester_missing)")
    incomplete = make_candidate(32, status="expired", confidence_score=None,
                                decision_reason="Анкета не завершена до истечения таймаута")
    summary, pages = build_status_report([missing, incomplete], [])
    assert "Анкета не завершена: 1 | Заявка отсутствовала: 1" in summary
    text = "\n".join(pages)
    assert "Итог: ЗАЯВКА ОТСУТСТВУЕТ В TELEGRAM | Баллы: не рассчитывались" in text
    assert "Итог: Анкета не завершена. Баллы: не рассчитывались" in text


def test_status_allowlist_does_not_bypass_revoked_telegram_admin_rights():
    """A former admin in ADMIN_USER_IDS must not see sensitive answers."""
    async def scenario():
        handler = GatekeeperHandlers(settings=settings(), db=SimpleNamespace(), scorer=NoopScorer())
        bot = SimpleNamespace(get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status="member")
        ))
        allowed = await handler._can_view_status(SimpleNamespace(bot=bot), 42)
        assert allowed is False
        bot.get_chat_member.assert_awaited_once_with(-1001, 42)
    asyncio.run(scenario())


def test_status_allowlist_is_additional_restriction_for_current_admin():
    async def scenario():
        handler = GatekeeperHandlers(settings=settings(), db=SimpleNamespace(), scorer=NoopScorer())
        bot = SimpleNamespace(get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status="administrator")
        ))
        # A current group admin not explicitly allowed is denied without API call.
        assert await handler._can_view_status(SimpleNamespace(bot=bot), 123) is False
        bot.get_chat_member.assert_not_awaited()
        # A current admin on the allowlist is granted access.
        assert await handler._can_view_status(SimpleNamespace(bot=bot), 42) is True
        bot.get_chat_member.assert_awaited_once_with(-1001, 42)
    asyncio.run(scenario())


def test_status_telegram_membership_check_fails_closed_on_api_error():
    from telegram.error import TelegramError

    async def scenario():
        handler = GatekeeperHandlers(settings=settings(), db=SimpleNamespace(), scorer=NoopScorer())
        bot = SimpleNamespace(get_chat_member=AsyncMock(
            side_effect=TelegramError("temporary network failure")
        ))
        assert await handler._can_view_status(SimpleNamespace(bot=bot), 42) is False
        bot.get_chat_member.assert_awaited_once_with(-1001, 42)
    asyncio.run(scenario())


def test_status_cooldown_rejection_summary_matches_individual_entries():
    at = datetime(2026, 11, 6, tzinfo=timezone.utc)

    def cooldown(id, status, confirmed_at=None):
        return CooldownDeclinedJoinRequest(
            id=id, user_id=2000+id, target_chat_id=-1001,
            request_date=at + timedelta(minutes=id),
            status=status, confirmed_at=confirmed_at,
        )

    items = [
        cooldown(1, "declined", at),    # Current confirmed refusal
        cooldown(2, "declined"),        # Historical confirmed refusal
        cooldown(3, "uncertain"),       # Telegram call ambiguous
        cooldown(4, "pending"),         # Still processing
    ]
    summary, pages = build_status_report(
        [make_candidate(5, status="rejected")], items,
    )
    assert "Повторных попыток по cooldown: 4" in summary
    assert "ПРОШЁЛ: 0 | НЕ ПРОШЁЛ: 3" in summary
    report = "\n".join(pages)
    assert report.count("Итог: НЕ ПРОШЁЛ (повторная заявка") == 2
    assert report.count("Итог: ИТОГ НЕ ПОДТВЕРЖДЁН") == 2


def test_status_revoked_admin_does_not_load_private_candidate_data():
    async def scenario():
        db = SimpleNamespace(get_status_history=AsyncMock(return_value=([make_candidate(1)], [])))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        message = SimpleNamespace(reply_text=AsyncMock())
        bot = SimpleNamespace(
            get_chat_member=AsyncMock(return_value=SimpleNamespace(status="member")),
            send_message=AsyncMock(),
        )
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(type="private", id=42),
            effective_user=SimpleNamespace(id=42),
            effective_message=message,
        )
        await handler.on_status(update, SimpleNamespace(bot=bot, args=[]))
        assert "только администраторам" in message.reply_text.await_args.args[0]
        db.get_status_history.assert_not_awaited()
        bot.send_message.assert_not_awaited()
    asyncio.run(scenario())


def test_status_allowlisted_group_owner_can_view_status():
    async def scenario():
        from dataclasses import replace
        s = replace(settings(), admin_user_ids={214627463})
        handler = GatekeeperHandlers(settings=s, db=SimpleNamespace(), scorer=NoopScorer())
        bot = SimpleNamespace(get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status="creator"),
        ))
        assert await handler._can_view_status(SimpleNamespace(bot=bot), 214627463)
        bot.get_chat_member.assert_awaited_once_with(-1001, 214627463)
    asyncio.run(scenario())


def test_status_admin_user_ids_absent_from_env(monkeypatch):
    """Production has no ADMIN_USER_IDS line: current Telegram admins remain eligible."""
    from app.config import Settings

    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("TARGET_CHAT_ID", "-1001")
    monkeypatch.setenv("ADMIN_CHAT_ID", "-2002")
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "test")
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    actual = Settings.from_env()
    assert actual.admin_user_ids == set()

    async def scenario():
        handler = GatekeeperHandlers(settings=actual, db=SimpleNamespace(), scorer=NoopScorer())
        bot = SimpleNamespace(get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status="creator"),
        ))
        assert await handler._can_view_status(SimpleNamespace(bot=bot), 214627463)
        bot.get_chat_member.assert_awaited_once_with(-1001, 214627463)

    asyncio.run(scenario())


def test_status_admin_user_ids_empty_value_same_as_absent(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("TARGET_CHAT_ID", "-1001")
    monkeypatch.setenv("ADMIN_CHAT_ID", "-2002")
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "test")
    monkeypatch.setenv("ADMIN_USER_IDS", "  ")
    assert Settings.from_env().admin_user_ids == set()


def test_status_combines_summary_and_twelve_candidates_into_one_text():
    async def scenario():
        candidates = [make_candidate(i, status="expired", stage=1,
                      confidence_score=None, location_answer=None,
                      business_answer=None, goal_answer=None,
                      decision_reason="Анкета не завершена до истечения таймаута")
                      for i in range(27, 39)]
        candidates[2].status = "approved"
        candidates[2].confidence_score = 85
        candidates[2].location_answer = "Я в Батуми"
        candidates[2].business_answer = "Я работаю менеджером в it компании"
        candidates[2].goal_answer = "Я хочу общаться и знакомиться"
        candidates[6].status = "approved"
        candidates[6].confidence_score = 70
        candidates[6].location_answer = "Нахожусь в Москве"
        candidates[6].business_answer = "Разработка программного обеспечения на Python и Go"
        candidates[6].goal_answer = "Расширить деловые контакты"
        candidates[4].decision_reason = "Заявка отсутствует в Telegram (Hide_requester_missing)"
        db = SimpleNamespace(get_status_history=AsyncMock(return_value=(candidates, [])))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        bot = SimpleNamespace(send_document=AsyncMock(), send_message=AsyncMock(),
                              get_chat_member=AsyncMock(return_value=SimpleNamespace(status="administrator")))
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_chat=SimpleNamespace(type="private", id=42),
                                 effective_user=SimpleNamespace(id=42), effective_message=message)
        await handler.on_status(update, SimpleNamespace(bot=bot, args=[]))
        message.reply_text.assert_awaited_once()
        text = message.reply_text.await_args.args[0]
        assert "ПРОШЁЛ: 2 | НЕ ПРОШЁЛ: 0" in text
        assert "Анкета не завершена: 9 | Заявка отсутствовала: 1" in text
        for i in range(27, 39):
            assert text.count(f"№{i} |") == 1
        assert _utf16_length(text) <= 4096
        bot.send_document.assert_not_awaited()
        bot.send_message.assert_not_awaited()
    asyncio.run(scenario())


def test_report_header_uses_first_and_last_join_dates():
    first = make_candidate(1, created_at=datetime(2026, 11, 6, tzinfo=timezone.utc),
                           request_date=datetime(2026, 11, 5, 22, 30, tzinfo=timezone.utc))
    last = make_candidate(2, created_at=datetime(2026, 11, 9, 14, 44, tzinfo=timezone.utc))
    summary, _ = build_status_report([last, first], [])
    assert summary.splitlines()[0] == (
        "SOLIDUS — история заявок Период: 05.11.2026 - 09.11.2026"
    )
    assert "(UTC)" not in summary.splitlines()[0]


def test_report_header_counts_cooldown_and_orphaned_telegram_event_dates():
    cooldown = CooldownDeclinedJoinRequest(
        id=3, user_id=345, target_chat_id=-1001,
        request_date=datetime(2026, 11, 4, 23, 30, tzinfo=timezone.utc),
        status="declined",
    )
    orphan = SimpleNamespace(
        id=9, update_id=128, user_id=999, status="failed",
        error_type="BadRequest",
        request_date=datetime(2026, 11, 10, 9, tzinfo=timezone.utc),
        created_at=datetime(2026, 11, 10, 9, tzinfo=timezone.utc),
    )
    summary, pages = build_status_report(
        [make_candidate(1, created_at=datetime(2026, 11, 7, tzinfo=timezone.utc))],
        [cooldown], [orphan],
    )
    assert summary.splitlines()[0] == (
        "SOLIDUS — история заявок Период: 04.11.2026 - 10.11.2026"
    )
    assert "Повторная попытка C3" in "\n".join(pages)
    assert "Событие Telegram U128" in "\n".join(pages)


def test_report_header_for_empty_history_and_single_date():
    summary, pages = build_status_report([], [], [])
    assert summary.splitlines()[0] == "SOLIDUS — история заявок Период: нет заявок"
    assert pages == []
    candidate = make_candidate(1, created_at=datetime(2026, 11, 9, tzinfo=timezone.utc))
    summary, _ = build_status_report([candidate], [])
    assert summary.splitlines()[0] == (
        "SOLIDUS — история заявок Период: 09.11.2026 - 09.11.2026"
    )
