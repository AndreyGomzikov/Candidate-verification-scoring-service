from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import BigInteger, DateTime, Integer, String, Text
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
