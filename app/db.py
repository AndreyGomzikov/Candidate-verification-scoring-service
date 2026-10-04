from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base, Candidate, CandidateStatus
from .scoring import ScoreResult


class Database:
    def __init__(self, url: str):
        self.engine = create_async_engine(url, future=True)
        self.sessions = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

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
