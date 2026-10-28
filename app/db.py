from __future__ import annotations

import json
import logging
from datetime import timedelta
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import and_, exists, func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from . import texts
from .models import Base, Candidate, CandidateStatus, SupportMessage, SupportTicket
from .scoring import ScoreResult


logger = logging.getLogger(__name__)


class Database:
    def __init__(self, url: str):
        self.engine = create_async_engine(url, future=True, hide_parameters=True)
        self.sessions = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # create_all() does NOT add columns/indexes to existing databases.
            # These idempotent, additive changes preserve all historical messages.
            if conn.dialect.name == "postgresql":
                for name, definition in (
                    ("delivery_status", "VARCHAR(16) NOT NULL DEFAULT 'pending'"),
                    ("delivery_attempts", "INTEGER NOT NULL DEFAULT 0"),
                    ("next_attempt_at", "TIMESTAMP WITH TIME ZONE"),
                    ("related_message_id", "INTEGER REFERENCES support_messages(id)"),
                ):
                    await conn.execute(text(
                        f"ALTER TABLE support_messages ADD COLUMN IF NOT EXISTS "
                        f"{name} {definition}"
                    ))
            elif conn.dialect.name == "sqlite":
                info = await conn.execute(text("PRAGMA table_info(support_messages)"))
                existing = {row[1] for row in info.all()}
                for name, definition in (
                    ("delivery_status", "VARCHAR(16) NOT NULL DEFAULT 'pending'"),
                    ("delivery_attempts", "INTEGER NOT NULL DEFAULT 0"),
                    ("next_attempt_at", "DATETIME"),
                    ("related_message_id", "INTEGER REFERENCES support_messages(id)"),
                ):
                    if name not in existing:
                        await conn.execute(text(
                            f"ALTER TABLE support_messages ADD COLUMN {name} {definition}"
                        ))
            else:
                raise RuntimeError("Support migration requires PostgreSQL or SQLite")

            await conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_support_delivery_alert_per_message
                ON support_messages(related_message_id)
                WHERE direction = 'delivery_alert'
            """))

            # Backfill the queue for messages written by older versions.
            await conn.execute(text("""
                UPDATE support_messages SET delivery_status = 'delivered'
                WHERE delivery_status = 'pending' AND (
                    (direction = 'candidate_to_admin' AND bot_admin_message_id IS NOT NULL)
                    OR (direction = 'admin_to_candidate' AND delivered_private_message_id IS NOT NULL)
                )
            """))

            # Backfill terminal failures from an older release. The unique
            # index and NOT EXISTS make the operation idempotent on restart.
            await conn.execute(text("""
                INSERT INTO support_messages (
                    ticket_id, direction, text, related_message_id,
                    delivery_status, delivery_attempts, created_at
                )
                SELECT m.ticket_id, 'delivery_alert', '', m.id,
                       'pending', 0, CURRENT_TIMESTAMP
                FROM support_messages m
                WHERE m.delivery_status = 'failed'
                  AND m.direction != 'delivery_alert'
                  AND NOT EXISTS (
                      SELECT 1 FROM support_messages a
                      WHERE a.direction = 'delivery_alert'
                        AND a.related_message_id = m.id
                  )
            """))

            # Rare pre-existing duplicate open tickets: keep newest open and
            # preserve older tickets and their full histories as closed.
            duplicates = await conn.execute(text("""
                SELECT COUNT(*) FROM (
                    SELECT user_id FROM support_tickets WHERE status = 'open'
                    GROUP BY user_id HAVING COUNT(*) > 1
                ) duplicate_users
            """))
            if duplicates.scalar_one() > 0:
                logger.warning("Closing legacy duplicate open support tickets")
                await conn.execute(text("""
                    UPDATE support_tickets SET status = 'closed',
                        updated_at = CURRENT_TIMESTAMP, closed_at = CURRENT_TIMESTAMP
                    WHERE id IN (
                        SELECT id FROM (
                            SELECT id, ROW_NUMBER() OVER (
                                PARTITION BY user_id ORDER BY id DESC
                            ) AS rn
                            FROM support_tickets WHERE status = 'open'
                        ) ranked WHERE rn > 1
                    )
                """))
            await conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_support_one_open_ticket_per_user
                ON support_tickets(user_id) WHERE status = 'open'
            """))

    async def close(self) -> None:
        await self.engine.dispose()

    async def create_candidate(
        self,
        *,
        user_id: int,
        user_chat_id: int,
        target_chat_id: int,
        username: str | None,
        first_name: str | None,
    ) -> Candidate:
        async with self.sessions() as session:
            await session.execute(
                update(Candidate)
                .where(
                    Candidate.user_id == user_id,
                    Candidate.target_chat_id == target_chat_id,
                    Candidate.status.in_(
                        [
                            CandidateStatus.INTERVIEW.value,
                            CandidateStatus.SCORING.value,
                            CandidateStatus.PENDING_ADMIN.value,
                        ]
                    ),
                )
                .values(status=CandidateStatus.SUPERSEDED.value)
            )
            candidate = Candidate(
                user_id=user_id,
                user_chat_id=user_chat_id,
                target_chat_id=target_chat_id,
                username=username,
                first_name=first_name,
                status=CandidateStatus.INTERVIEW.value,
                stage=1,
            )
            session.add(candidate)
            await session.commit()
            await session.refresh(candidate)
            return candidate

    async def get_active_interview(self, user_id: int) -> Candidate | None:
        async with self.sessions() as session:
            stmt = (
                select(Candidate)
                .where(
                    Candidate.user_id == user_id,
                    Candidate.status == CandidateStatus.INTERVIEW.value,
                )
                .order_by(Candidate.id.desc())
                .limit(1)
            )
            return await session.scalar(stmt)

    async def get_active_candidate_for_chat(
        self, user_id: int, target_chat_id: int
    ) -> Candidate | None:
        async with self.sessions() as session:
            stmt = (
                select(Candidate)
                .where(
                    Candidate.user_id == user_id,
                    Candidate.target_chat_id == target_chat_id,
                    Candidate.status.in_(
                        [
                            CandidateStatus.INTERVIEW.value,
                            CandidateStatus.SCORING.value,
                            CandidateStatus.PENDING_ADMIN.value,
                        ]
                    ),
                )
                .order_by(Candidate.id.desc())
                .limit(1)
            )
            return await session.scalar(stmt)

    async def get_recent_rejected_candidate(
        self, user_id: int, target_chat_id: int, cutoff: datetime
    ) -> Candidate | None:
        async with self.sessions() as session:
            stmt = (
                select(Candidate)
                .where(
                    Candidate.user_id == user_id,
                    Candidate.target_chat_id == target_chat_id,
                    Candidate.status.in_(
                        [
                            CandidateStatus.REJECTED.value,
                            CandidateStatus.AUTO_REJECTED.value,
                            CandidateStatus.EXPIRED.value,
                        ]
                    ),
                    Candidate.updated_at >= cutoff,
                )
                .order_by(Candidate.updated_at.desc())
                .limit(1)
            )
            return await session.scalar(stmt)

    async def list_expired_interviews(self, cutoff: datetime) -> list[Candidate]:
        async with self.sessions() as session:
            stmt = (
                select(Candidate)
                .where(
                    Candidate.status == CandidateStatus.INTERVIEW.value,
                    Candidate.updated_at < cutoff,
                )
                .order_by(Candidate.updated_at.asc())
            )
            result = await session.scalars(stmt)
            return list(result.all())

    async def get_candidate(self, candidate_id: int) -> Candidate | None:
        async with self.sessions() as session:
            return await session.get(Candidate, candidate_id)

    async def set_user_chat_id(self, candidate_id: int, user_chat_id: int) -> Candidate:
        async with self.sessions() as session:
            candidate = await session.get(Candidate, candidate_id)
            if candidate is None:
                raise LookupError(candidate_id)
            candidate.user_chat_id = user_chat_id
            await session.commit()
            await session.refresh(candidate)
            return candidate

    async def save_answer(
        self, candidate_id: int, *, field: str, answer: str, next_stage: int
    ) -> Candidate:
        allowed = {"location_answer", "business_answer", "goal_answer"}
        if field not in allowed:
            raise ValueError(f"Unsupported answer field: {field}")
        async with self.sessions() as session:
            candidate = await session.get(Candidate, candidate_id)
            if candidate is None:
                raise LookupError(candidate_id)
            setattr(candidate, field, answer)
            candidate.stage = next_stage
            await session.commit()
            await session.refresh(candidate)
            return candidate

    async def set_status(
        self,
        candidate_id: int,
        status: CandidateStatus,
        *,
        reason: str | None = None,
        decided_by_user_id: int | None = None,
    ) -> Candidate:
        async with self.sessions() as session:
            candidate = await session.get(Candidate, candidate_id)
            if candidate is None:
                raise LookupError(candidate_id)
            candidate.status = status.value
            if reason is not None:
                candidate.decision_reason = reason
            if decided_by_user_id is not None:
                candidate.decided_by_user_id = decided_by_user_id
            await session.commit()
            await session.refresh(candidate)
            return candidate

    async def save_score(
        self, candidate_id: int, score: ScoreResult, raw_ai_json: str
    ) -> Candidate:
        async with self.sessions() as session:
            candidate = await session.get(Candidate, candidate_id)
            if candidate is None:
                raise LookupError(candidate_id)
            candidate.confidence_score = score.confidence_score
            candidate.location_score = score.location_score
            candidate.business_score = score.business_score
            candidate.intent_score = score.intent_score
            candidate.spam_score = score.spam_score
            candidate.ai_confidence = score.ai_confidence
            candidate.summary = score.summary
            candidate.risk_flags_json = json.dumps(
                score.risk_flags, ensure_ascii=False
            )
            candidate.raw_ai_json = raw_ai_json
            await session.commit()
            await session.refresh(candidate)
            return candidate

    async def count_by_status(self) -> dict[str, int]:
        async with self.sessions() as session:
            result = await session.scalars(select(Candidate.status))
            return dict(Counter(result.all()))


    # Support data is independent of candidate statuses: no second scoring/decision.
    async def get_latest_candidate_for_user(self, user_id: int) -> Candidate | None:
        async with self.sessions() as session:
            return await session.scalar(
                select(Candidate)
                .where(Candidate.user_id == user_id)
                .order_by(Candidate.id.desc())
                .limit(1)
            )

    async def count_recent_support_messages(self, user_id: int, since: datetime) -> int:
        async with self.sessions() as session:
            return int(await session.scalar(
                select(func.count(SupportMessage.id))
                .join(SupportTicket, SupportTicket.id == SupportMessage.ticket_id)
                .where(
                    SupportTicket.user_id == user_id,
                    SupportMessage.direction == "candidate_to_admin",
                    SupportMessage.created_at >= since,
                )
            ) or 0)

    async def get_or_create_open_support_ticket(
        self, candidate: Candidate, user_chat_id: int
    ) -> SupportTicket:
        """The partial unique index serializes competing creations per user."""
        async with self.sessions() as session:
            ticket = await session.scalar(
                select(SupportTicket)
                .where(
                    SupportTicket.user_id == candidate.user_id,
                    SupportTicket.status == "open",
                )
                .limit(1)
            )
            if ticket is None:
                ticket = SupportTicket(
                    user_id=candidate.user_id, candidate_id=candidate.id,
                    user_chat_id=user_chat_id, status="open",
                )
                session.add(ticket)
                try:
                    await session.commit()
                    await session.refresh(ticket)
                    return ticket
                except IntegrityError:
                    # Another update won the race; retry in a clean transaction.
                    await session.rollback()
                    ticket = await session.scalar(
                        select(SupportTicket).where(
                            SupportTicket.user_id == candidate.user_id,
                            SupportTicket.status == "open",
                        ).limit(1)
                    )
                    if ticket is None:
                        raise
            ticket.user_chat_id = user_chat_id
            ticket.candidate_id = candidate.id
            ticket.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(ticket)
            return ticket

    async def get_support_ticket(self, ticket_id: int) -> SupportTicket | None:
        async with self.sessions() as session:
            return await session.get(SupportTicket, ticket_id)

    async def get_ticket_for_admin_reply(self, bot_admin_message_id: int) -> SupportTicket | None:
        async with self.sessions() as session:
            return await session.scalar(
                select(SupportTicket)
                .join(SupportMessage, SupportTicket.id == SupportMessage.ticket_id)
                .where(
                    SupportMessage.bot_admin_message_id == bot_admin_message_id,
                    SupportMessage.direction == "candidate_to_admin",
                )
                .limit(1)
            )

    async def store_candidate_support_message(
        self, ticket_id: int, text: str, private_chat_id: int, message_id: int
    ) -> SupportMessage | None:
        """None means Telegram redelivered a previously processed update."""
        async with self.sessions() as session:
            exists = await session.scalar(
                select(SupportMessage.id).where(
                    SupportMessage.source_private_chat_id == private_chat_id,
                    SupportMessage.source_private_message_id == message_id,
                )
            )
            if exists is not None:
                return None
            item = SupportMessage(
                ticket_id=ticket_id, text=text, direction="candidate_to_admin",
                source_private_chat_id=private_chat_id,
                source_private_message_id=message_id,
            )
            session.add(item)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                # A redelivered Telegram update may race with another worker.
                duplicate = await session.scalar(select(SupportMessage.id).where(
                    (SupportMessage.source_private_chat_id == private_chat_id)
                    & (SupportMessage.source_private_message_id == message_id)
                ))
                if duplicate is not None:
                    return None
                raise
            await session.refresh(item)
            return item

    async def store_admin_support_message(
        self, ticket_id: int, text: str, admin_user_id: int, message_id: int
    ) -> SupportMessage | None:
        async with self.sessions() as session:
            exists = await session.scalar(
                select(SupportMessage.id).where(
                    SupportMessage.source_admin_message_id == message_id
                )
            )
            if exists is not None:
                return None
            item = SupportMessage(
                ticket_id=ticket_id, text=text, direction="admin_to_candidate",
                admin_user_id=admin_user_id, source_admin_message_id=message_id,
            )
            session.add(item)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                duplicate = await session.scalar(select(SupportMessage.id).where(
                    SupportMessage.source_admin_message_id == message_id
                ))
                if duplicate is not None:
                    return None
                raise
            await session.refresh(item)
            return item

    async def list_due_support_message_ids(
        self, now: datetime, *, limit: int = 20
    ) -> list[int]:
        """Find messages ready to send, including abandoned in-flight leases."""
        async with self.sessions() as session:
            # A lease can expire after the final attempt (process crash). Record
            # terminal failure and queue one operator alert in the SAME transaction.
            expired = (await session.scalars(
                select(SupportMessage).where(
                    SupportMessage.delivery_status == "sending",
                    SupportMessage.delivery_attempts >= 8,
                    SupportMessage.next_attempt_at <= now,
                )
            )).all()
            for item in expired:
                item.delivery_status = "failed"
                item.delivery_error = "DeliveryLeaseExpired"
                item.next_attempt_at = None
                if item.direction != "delivery_alert":
                    await self._enqueue_failure_alert(session, item)
                else:
                    logger.critical("Support failure alert exhausted: message=%s", item.id)
            await session.commit()
            # Exclude blocked close notices before LIMIT. Otherwise a backlog
            # of notices waiting for an earlier failed admin reply can fill the
            # whole batch and indefinitely starve unrelated, deliverable mail.
            earlier_reply = SupportMessage.__table__.alias("earlier_reply")
            blocked_close_notice = exists(
                select(1).select_from(earlier_reply).where(
                    earlier_reply.c.ticket_id == SupportMessage.ticket_id,
                    earlier_reply.c.id < SupportMessage.id,
                    earlier_reply.c.direction == "admin_to_candidate",
                    earlier_reply.c.delivery_status.in_(
                        ("pending", "retry", "sending", "failed")
                    ),
                )
            )
            result = await session.scalars(
                select(SupportMessage.id)
                .where(
                    SupportMessage.delivery_status.in_(("pending", "retry", "sending")),
                    or_(SupportMessage.next_attempt_at.is_(None),
                        SupportMessage.next_attempt_at <= now),
                    or_(
                        SupportMessage.direction != "ticket_closed",
                        ~blocked_close_notice,
                    ),
                )
                .order_by(SupportMessage.id.asc())
                .limit(limit)
            )
            return list(result.all())

    async def claim_support_delivery(
        self, message_id: int, now: datetime, *, max_attempts: int = 8,
        lease_seconds: int = 300,
    ) -> SupportMessage | None:
        """Atomic claim prevents two bot workers retrying the same message."""
        async with self.sessions() as session:
            previous = SupportMessage.__table__.alias("previous")
            result = await session.execute(
                update(SupportMessage)
                .where(
                    SupportMessage.id == message_id,
                    SupportMessage.delivery_status.in_(("pending", "retry", "sending")),
                    SupportMessage.delivery_attempts < max_attempts,
                    or_(
                        SupportMessage.direction != "ticket_closed",
                        ~exists(select(1).select_from(previous).where(
                            previous.c.ticket_id == SupportMessage.ticket_id,
                            previous.c.id < SupportMessage.id,
                            previous.c.direction == "admin_to_candidate",
                            # A terminal failure still blocks the close notice:
                            # the moderator may manually retry that earlier reply.
                            previous.c.delivery_status.in_(("pending", "retry", "sending", "failed")),
                        )),
                    ),
                    or_(SupportMessage.next_attempt_at.is_(None),
                        SupportMessage.next_attempt_at <= now),
                )
                .values(
                    delivery_status="sending",
                    delivery_attempts=SupportMessage.delivery_attempts + 1,
                    next_attempt_at=now + timedelta(seconds=lease_seconds),
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                return None
            await session.commit()
            return await session.get(SupportMessage, message_id)

    async def mark_support_delivery(
        self, message_id: int, *, attempt: int,
        admin_message_id: int | None = None,
        private_message_id: int | None = None,
    ) -> bool:
        async with self.sessions() as session:
            values: dict = {
                "delivery_status": "delivered", "delivery_error": None,
                "next_attempt_at": None,
            }
            if admin_message_id is not None:
                values["bot_admin_message_id"] = admin_message_id
            if private_message_id is not None:
                values["delivered_private_message_id"] = private_message_id
            previous = SupportMessage.__table__.alias("previous")
            result = await session.execute(
                update(SupportMessage)
                .where(
                    SupportMessage.id == message_id,
                    SupportMessage.delivery_status == "sending",
                    SupportMessage.delivery_attempts == attempt,
                )
                .values(**values)
            )
            await session.commit()
            return result.rowcount == 1

    @staticmethod
    async def _enqueue_failure_alert(session: AsyncSession, failed: SupportMessage) -> None:
        """Durable operator alert, deduplicated for each failed delivery.

        If the same message fails again after manual retry, re-arm its existing
        alert instead of creating unbounded duplicates.
        """
        alert = await session.scalar(select(SupportMessage).where(
            SupportMessage.direction == "delivery_alert",
            SupportMessage.related_message_id == failed.id,
        ))
        if alert is None:
            session.add(SupportMessage(
                ticket_id=failed.ticket_id,
                direction="delivery_alert", text="",
                related_message_id=failed.id,
            ))
        else:
            alert.delivery_status = "pending"
            alert.delivery_attempts = 0
            alert.delivery_error = None
            alert.next_attempt_at = None

    async def fail_support_delivery(
        self, message_id: int, *, attempt: int, error: str,
        now: datetime, max_attempts: int = 8,
    ) -> str:
        """Terminal failures enqueue an admin alert in the same DB transaction."""
        delay = min(3600, 30 * 2 ** min(attempt - 1, 8))
        status = "failed" if attempt >= max_attempts else "retry"
        async with self.sessions() as session:
            result = await session.execute(
                update(SupportMessage)
                .where(
                    SupportMessage.id == message_id,
                    SupportMessage.delivery_status == "sending",
                    SupportMessage.delivery_attempts == attempt,
                )
                .values(
                    delivery_status=status,
                    delivery_error=error[:80],
                    next_attempt_at=(now + timedelta(seconds=delay)) if status == "retry" else None,
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                return "stale"
            if status == "failed":
                entry = await session.get(SupportMessage, message_id)
                if entry.direction != "delivery_alert":
                    await self._enqueue_failure_alert(session, entry)
                else:
                    logger.critical("Support delivery alert failed permanently: message=%s", message_id)
            await session.commit()
            return status

    async def retry_failed_support_message(self, message_id: int) -> bool:
        """Administrator-requested replay; other message states are untouched."""
        async with self.sessions() as session:
            result = await session.execute(
                update(SupportMessage)
                .where(SupportMessage.id == message_id,
                       SupportMessage.delivery_status == "failed",
                       SupportMessage.direction != "delivery_alert")
                .values(delivery_status="pending", delivery_attempts=0,
                        delivery_error=None, next_attempt_at=None)
            )
            if result.rowcount == 1:
                # Cancel any alert which has not been sent yet. A new terminal
                # failure can re-arm this exact alert after a later retry cycle.
                await session.execute(update(SupportMessage).where(
                    SupportMessage.direction == "delivery_alert",
                    SupportMessage.related_message_id == message_id,
                    SupportMessage.delivery_status.in_(("pending", "retry")),
                ).values(delivery_status="cancelled", next_attempt_at=None))
            await session.commit()
            return result.rowcount == 1

    async def get_support_message(self, message_id: int) -> SupportMessage | None:
        async with self.sessions() as session:
            return await session.get(SupportMessage, message_id)

    async def list_failed_support_messages(self, *, limit: int = 15) -> list[SupportMessage]:
        async with self.sessions() as session:
            rows = await session.scalars(
                select(SupportMessage)
                .where(SupportMessage.delivery_status == "failed")
                .order_by(SupportMessage.id.desc())
                .limit(limit)
            )
            return list(rows.all())

    async def cancel_support_alert(self, message_id: int, *, attempt: int) -> bool:
        async with self.sessions() as session:
            result = await session.execute(update(SupportMessage).where(
                SupportMessage.id == message_id,
                SupportMessage.direction == "delivery_alert",
                SupportMessage.delivery_status == "sending",
                SupportMessage.delivery_attempts == attempt,
            ).values(delivery_status="cancelled", next_attempt_at=None,
                     delivery_error=None))
            await session.commit()
            return result.rowcount == 1

    async def recover_support_card_mapping(
        self, ticket_id: int, message_id: int, admin_message_id: int
    ) -> SupportTicket | None:
        """Recover when Telegram sent a card before its message ID was committed.

        Caller MUST first confirm message is a reply to this bot's message in
        the configured admin group and parse the bot-authored identity marker.
        """
        async with self.sessions() as session:
            item = await session.get(SupportMessage, message_id)
            if (
                item is None or item.ticket_id != ticket_id
                or item.direction != "candidate_to_admin"
            ):
                return None
            ticket = await session.get(SupportTicket, ticket_id)
            if ticket is None:
                return None
            if item.bot_admin_message_id is None:
                item.bot_admin_message_id = admin_message_id
                item.delivery_status = "delivered"
                item.next_attempt_at = None
                item.delivery_error = None
                await session.commit()
            # An earlier retry might have posted a second card; both still
            # legitimately identify this same message/ticket.
            return ticket

    async def close_support_ticket(self, ticket_id: int, admin_user_id: int) -> bool:
        """Atomic closure + durable DM notification, once per ticket."""
        async with self.sessions() as session:
            now = datetime.now(timezone.utc)
            result = await session.execute(
                update(SupportTicket)
                .where(SupportTicket.id == ticket_id, SupportTicket.status == "open")
                .values(status="closed", updated_at=now, closed_at=now,
                        closed_by_user_id=admin_user_id)
            )
            if result.rowcount != 1:
                await session.rollback()
                return False
            session.add(SupportMessage(
                ticket_id=ticket_id,
                direction="ticket_closed",
                text=texts.SUPPORT_CLOSED,
                admin_user_id=admin_user_id,
            ))
            await session.commit()
            return True

    async def get_ticket_close_notification_id(self, ticket_id: int) -> int | None:
        async with self.sessions() as session:
            return await session.scalar(
                select(SupportMessage.id)
                .where(SupportMessage.ticket_id == ticket_id,
                       SupportMessage.direction == "ticket_closed")
                .order_by(SupportMessage.id.desc()).limit(1)
            )
