"""Event flow tests independent of optional SQLite/Postgres drivers."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateTable
from telegram.error import TelegramError

from app.db import Database
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus, JoinRequestEvent
from app.status_report import build_status_report
from tests.test_handlers import NoopScorer, settings


class LedgerDB(Database):
    def __init__(self):
        self.events = {}
        self.candidates = {}
        self.next_candidate_id = 1

    @asynccontextmanager
    async def join_request_lock(self, chat_id, user_id):
        yield

    async def record_join_request_event(self, *, update_id, **data):
        self.events.setdefault(update_id, NS(update_id=update_id, **data, status="received",
            candidate_id=None, first_message_id=None, error_type=None,
            alert_required=False, admin_alerted_at=None))

    async def update_join_request_event(self, update_id, status, **fields):
        event = self.events[update_id]
        event.status = status
        for key, value in fields.items():
            setattr(event, key, value)

    async def mark_join_request_alerted(self, update_id):
        self.events[update_id].admin_alerted_at = datetime.now(timezone.utc)

    async def get_active_candidate_for_chat(self, user_id, chat_id):
        return None

    async def get_latest_candidate_for_chat(self, user_id, chat_id):
        return None

    async def last_cooldown_declined_request_date(self, user_id, chat_id):
        return None

    async def get_recent_rejected_candidate(self, user_id, chat_id, cutoff):
        return None

    async def create_candidate(self, **values):
        c = Candidate(id=self.next_candidate_id, status="interview", stage=1,
                      created_at=datetime.now(timezone.utc), **values)
        self.candidates[c.id] = c
        self.next_candidate_id += 1
        return c

    async def set_status(self, candidate_id, status, *, reason=None, **_):
        c = self.candidates[candidate_id]
        c.status = status.value
        c.decision_reason = reason
        return c

    async def get_candidate(self, candidate_id):
        return self.candidates[candidate_id]


def join_update(number):
    return NS(update_id=number, chat_join_request=NS(
        chat=NS(id=-1001),
        from_user=NS(id=123, username="demo", first_name="Demo"),
        user_chat_id=9988, date=datetime.now(timezone.utc),
    ))


def test_first_question_success_audits_message_id():
    async def run():
        db = LedgerDB()
        bot = NS(send_message=AsyncMock(return_value=NS(message_id=703)))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await handler.on_join_request(join_update(42), NS(bot=bot))
        event = db.events[42]
        assert event.status == "first_question_sent"
        assert event.first_message_id == 703
        assert db.candidates[event.candidate_id].status == "interview"
        bot.send_message.assert_awaited_once()
    asyncio.run(run())


def test_first_question_delivery_error_does_not_auto_decline():
    async def run():
        db = LedgerDB()
        bot = NS(send_message=AsyncMock(side_effect=[TelegramError("cannot initiate"), NS(message_id=10)]))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await handler.on_join_request(join_update(43), NS(bot=bot))
        event = db.events[43]
        assert event.status == "first_question_failed"
        assert event.alert_required
        assert event.admin_alerted_at is not None
        candidate = db.candidates[event.candidate_id]
        assert candidate.status == "pending_admin"
        assert "Не удалось начать ЛС" in candidate.decision_reason
        _, pages = build_status_report([candidate], [])
        assert "ТЕХНИЧЕСКАЯ ПРОБЛЕМА" in "\n".join(pages)
    asyncio.run(run())


def test_admin_alert_failure_remains_pending():
    async def run():
        db = LedgerDB()
        bot = NS(send_message=AsyncMock(side_effect=TelegramError("send failed")))
        handler = GatekeeperHandlers(settings=settings(), db=db, scorer=NoopScorer())
        await handler.on_join_request(join_update(44), NS(bot=bot))
        assert db.events[44].status == "first_question_failed"
        assert db.events[44].alert_required is True
        assert db.events[44].admin_alerted_at is None
        assert db.candidates[1].status == "pending_admin"
    asyncio.run(run())


def test_event_table_compiles_for_postgres_and_sqlite():
    pg = str(CreateTable(JoinRequestEvent.__table__).compile(dialect=postgresql.dialect()))
    sq = str(CreateTable(JoinRequestEvent.__table__).compile(dialect=sqlite.dialect()))
    for ddl in (pg, sq):
        assert "join_request_events" in ddl
        assert "UNIQUE (update_id)" in ddl
        assert "first_message_id" in ddl
        assert "admin_alerted_at" in ddl
        assert "FOREIGN KEY(candidate_id)" in ddl


def test_sqlite_schema_enforces_unique_telegram_update_id():
    """Sync SQLite smoke: the new table and uniqueness work without aiosqlite."""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import IntegrityError
    from app.models import Base
    engine = create_engine("sqlite://")
    try:
        Base.metadata.create_all(engine)
        now = datetime.now(timezone.utc)
        with engine.begin() as conn:
            conn.execute(JoinRequestEvent.__table__.insert().values(
                update_id=801, target_chat_id=-1001, user_id=91,
                status="received", created_at=now, updated_at=now,
            ))
        try:
            with engine.begin() as conn:
                conn.execute(JoinRequestEvent.__table__.insert().values(
                    update_id=801, target_chat_id=-1001, user_id=91,
                    status="received", created_at=now, updated_at=now,
                ))
        except IntegrityError:
            pass
        else:
            raise AssertionError("Update ID unique constraint missing")
    finally:
        engine.dispose()


def test_status_exposes_event_without_created_candidate():
    event = JoinRequestEvent(
        id=7, update_id=801, user_id=91, target_chat_id=-1001,
        status="delivery_unconfirmed", error_type="InterruptedBeforeDeliveryConfirmation",
        request_date=datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    summary, pages = build_status_report([], [], [event])
    assert "Технические события без анкеты: 1" in summary
    assert "U801" in pages[0]
    assert "АНКЕТА НЕ СОЗДАНА" in pages[0]
