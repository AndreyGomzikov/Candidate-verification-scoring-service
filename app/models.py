from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CandidateStatus(StrEnum):
    INTERVIEW = "interview"
    SCORING = "scoring"
    PENDING_ADMIN = "pending_admin"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_REJECTED = "auto_rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    ERROR = "error"


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    user_chat_id: Mapped[int] = mapped_column(BigInteger)
    target_chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), default=CandidateStatus.INTERVIEW.value, index=True
    )
    stage: Mapped[int] = mapped_column(Integer, default=1)

    location_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    business_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    goal_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    confidence_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    location_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    business_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    intent_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spam_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_confidence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_flags_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_ai_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SupportTicket(Base):
    """One conversation until an administrator closes it."""

    __tablename__ = "support_tickets"

    __table_args__ = (
        Index(
            "uq_support_one_open_ticket_per_user", "user_id", unique=True,
            postgresql_where=text("status = 'open'"),
            sqlite_where=text("status = 'open'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    user_chat_id: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_by_user_id: Mapped[int | None] = mapped_column(BigInteger)


class SupportMessage(Base):
    """Persistent source + delivery information for each support message."""

    __tablename__ = "support_messages"
    __table_args__ = (
        UniqueConstraint(
            "source_private_chat_id", "source_private_message_id",
            name="uq_support_candidate_source",
        ),
        Index(
            "uq_support_delivery_alert_per_message", "related_message_id", unique=True,
            postgresql_where=text("direction = 'delivery_alert'"),
            sqlite_where=text("direction = 'delivery_alert'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("support_tickets.id"), index=True)
    direction: Mapped[str] = mapped_column(String(24))
    text: Mapped[str] = mapped_column(Text)
    admin_user_id: Mapped[int | None] = mapped_column(BigInteger)
    source_private_chat_id: Mapped[int | None] = mapped_column(BigInteger)
    source_private_message_id: Mapped[int | None] = mapped_column(Integer)
    source_admin_message_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    bot_admin_message_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    delivered_private_message_id: Mapped[int | None] = mapped_column(Integer)
    delivery_error: Mapped[str | None] = mapped_column(String(80))
    # A delivery_alert refers to the original failed message; NULL otherwise.
    related_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("support_messages.id"), nullable=True
    )
    # Persistent retry queue. A stale "sending" lease is reclaimed after a crash.
    delivery_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    delivery_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
