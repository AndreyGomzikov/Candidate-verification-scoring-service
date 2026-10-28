"""Regression: blocked close notices must not starve due support messages."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import texts
from app.db import Database
from app.models import Base, SupportMessage, SupportTicket


class _SQLiteSessionAdapter:
    """Run the production SQL query on an in-memory SQLite database."""

    def __init__(self, engine):
        self.engine = engine
        self.session = None

    async def __aenter__(self):
        self.session = Session(self.engine, expire_on_commit=False)
        return self

    async def __aexit__(self, *_args):
        self.session.close()

    async def scalars(self, statement):
        return self.session.scalars(statement)

    async def commit(self):
        self.session.commit()


def test_blocked_close_notices_do_not_fill_batch_before_limit():
    """Twenty-five blocked notices must not hide a later, ready message."""
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            for index in range(25):
                ticket = SupportTicket(
                    candidate_id=100 + index,
                    user_id=1000 + index,
                    user_chat_id=2000 + index,
                    status="closed",
                )
                session.add(ticket)
                session.flush()
                session.add_all([
                    SupportMessage(
                        ticket_id=ticket.id,
                        direction="admin_to_candidate",
                        text="Undelivered reply",
                        delivery_status="failed",
                        delivery_attempts=8,
                    ),
                    SupportMessage(
                        ticket_id=ticket.id,
                        direction="ticket_closed",
                        text=texts.SUPPORT_CLOSED,
                        delivery_status="pending",
                    ),
                ])
            ready_ticket = SupportTicket(
                candidate_id=999,
                user_id=9999,
                user_chat_id=8888,
                status="open",
            )
            session.add(ready_ticket)
            session.flush()
            ready = SupportMessage(
                ticket_id=ready_ticket.id,
                direction="candidate_to_admin",
                text="Later message that must not starve",
                delivery_status="pending",
            )
            session.add(ready)
            session.commit()
            ready_id = ready.id

        db = object.__new__(Database)
        db.sessions = lambda: _SQLiteSessionAdapter(engine)
        ids = asyncio.run(db.list_due_support_message_ids(datetime.now(timezone.utc), limit=20))
        assert ids == [ready_id]
    finally:
        engine.dispose()


def test_due_close_notice_becomes_eligible_after_earlier_reply_delivered():
    """The filter must not permanently suppress eligible close notifications."""
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            ticket = SupportTicket(
                candidate_id=100,
                user_id=200,
                user_chat_id=300,
                status="closed",
            )
            session.add(ticket)
            session.flush()
            earlier = SupportMessage(
                ticket_id=ticket.id,
                direction="admin_to_candidate",
                text="Answer",
                delivery_status="failed",
            )
            notice = SupportMessage(
                ticket_id=ticket.id,
                direction="ticket_closed",
                text=texts.SUPPORT_CLOSED,
                delivery_status="pending",
            )
            session.add_all([earlier, notice])
            session.commit()
            earlier_id, notice_id = earlier.id, notice.id

        db = object.__new__(Database)
        db.sessions = lambda: _SQLiteSessionAdapter(engine)
        now = datetime.now(timezone.utc)
        assert asyncio.run(db.list_due_support_message_ids(now)) == []
        with Session(engine) as session:
            session.get(SupportMessage, earlier_id).delivery_status = "delivered"
            session.commit()
        assert asyncio.run(db.list_due_support_message_ids(now)) == [notice_id]
    finally:
        engine.dispose()
