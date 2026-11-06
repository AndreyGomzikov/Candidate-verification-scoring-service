"""Independent spam review must fail closed, never classify mere poor fit as abuse."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus
from app.scoring import (
    AIScorer, ScoreResult, auto_reject_reason, calculate_confidence_score,
    parse_score_json, with_spam_verification,
)
from app.core_review import CoreVerification
from app.spam_review import (
    SpamEvidence, SpamVerification, parse_primary_evidence, validate_second_review,
)

LOCATION = "Живу в Батуми, Грузия"
BUSINESS = "Владелец IT-компании с 12 сотрудниками"
GOAL = "Буду массово рассылать рекламу каждому участнику клуба"
BAD_GOAL = "Тестирую работу гейткипера"


def score_payload(goal=GOAL, spam=95, evidence=True):
    return json.dumps({
        "location_score": 90,
        "business_score": 85,
        "intent_score": 0,
        "spam_score": spam,
        "summary": "Кандидат из Батуми.",
        "risk_flags": [],
        "confidence_evidence": {
            "location": "Живу в Батуми",
            "business": "Владелец IT-компании",
            "goal": goal,
        },
        "contradiction_quotes": [], "uncertainty_quotes": [],
        "spam_evidence": ([{
            "category": "mass_advertising",
            "source_field": "goal",
            "quote": goal,
            "explanation": "Кандидат прямо объявляет о массовой рассылке рекламы",
        }] if evidence else []),
    }, ensure_ascii=False)


def verdict(*, decision="confirmed", quote=GOAL, category="mass_advertising", field="goal"):
    return json.dumps({
        "verdict": decision,
        "category": category,
        "source_field": field,
        "quote": quote,
        "reason": "Кандидат явно заявил о намерении массовой рассылки рекламы",
    }, ensure_ascii=False)


def check(score):
    return auto_reject_reason(
        score, spam_threshold=85, core_score_max=15, min_ai_confidence=80,
    )


def new_scorer():
    return AIScorer(
        base_url="https://example.invalid/v1", api_key="test", model="test", timeout_seconds=2,
    )


def test_primary_evidence_requires_literal_quote_and_allowed_category():
    assert not parse_primary_evidence([{
        "category": "mass_advertising", "source_field": "goal",
        "quote": "Рассылка рекламы всем", "explanation": "Это подтверждение массового спама",
    }], answers={"location": LOCATION, "business": BUSINESS, "goal": BAD_GOAL})
    assert not parse_primary_evidence([{
        "category": "weak_business_fit", "source_field": "goal",
        "quote": BAD_GOAL, "explanation": "Не указан бизнес-запрос кандидата",
    }], answers={"location": LOCATION, "business": BUSINESS, "goal": BAD_GOAL})
    assert not parse_primary_evidence([{
        "category": "mass_advertising", "source_field": "location",
        "quote": GOAL, "explanation": "Массовая рассылка рекламы участникам",
    }], answers={"location": LOCATION, "business": BUSINESS, "goal": GOAL})


def test_high_spam_risk_without_evidence_is_not_an_auto_reject():
    result = parse_score_json(
        score_payload(goal=BAD_GOAL, evidence=False),
        location=LOCATION, business=BUSINESS, goal=BAD_GOAL,
    )
    assert result.ai_confidence == 100
    assert result.confidence_score == calculate_confidence_score(90, 85, 0, 95)
    assert check(result) is None
    updated, raw = asyncio.run(new_scorer().confirm_high_spam(
        score=result, raw="{}", location=LOCATION, business=BUSINESS, goal=BAD_GOAL,
    ))
    assert updated.spam_verification.status == "insufficient_evidence"
    assert check(updated) is None
    assert json.loads(raw)["_gatekeeper_spam_review"]["status"] == "insufficient_evidence"


def test_made_up_quote_does_not_authorize_rejection():
    payload = json.loads(score_payload(goal=BAD_GOAL))
    payload["spam_evidence"][0]["quote"] = GOAL
    result = parse_score_json(json.dumps(payload, ensure_ascii=False),
                              location=LOCATION, business=BUSINESS, goal=BAD_GOAL)
    assert not result.spam_evidence
    assert check(result) is None


def test_second_pass_can_explicitly_disagree_with_primary_suspicion():
    original = parse_score_json(score_payload(), location=LOCATION, business=BUSINESS, goal=GOAL)
    result = validate_second_review(
        verdict(decision="not_confirmed"),
        answers={"location": LOCATION, "business": BUSINESS, "goal": GOAL},
        primary=original.spam_evidence,
    )
    assert result.status == "not_confirmed"
    assert check(replace(original, spam_verification=result)) is None


def test_second_pass_cannot_hallucinate_or_change_category():
    primary = parse_score_json(score_payload(), location=LOCATION, business=BUSINESS, goal=GOAL)
    for result in (
        verdict(quote="Здесь нет такой цитаты"),
        verdict(category="fraudulent_scheme"),
        verdict(field="business"),
        verdict(quote="каждому участнику", field="goal", category="phishing"),
        '{"verdict":"confirmed", "category":"mass_advertising", "quote":42}',
        "not json",
    ):
        second = validate_second_review(
            result, answers={"location": LOCATION, "business": BUSINESS, "goal": GOAL},
            primary=primary.spam_evidence,
        )
        assert not second.permits_auto_reject
        assert check(replace(primary, spam_verification=second)) is None


def test_clear_spam_can_be_auto_rejected_only_after_two_grounded_passes():
    result = parse_score_json(score_payload(), location=LOCATION, business=BUSINESS, goal=GOAL)
    assert result.ai_confidence == 100
    assert check(result) is None
    decision = validate_second_review(
        verdict(), answers={"location": LOCATION, "business": BUSINESS, "goal": GOAL},
        primary=result.spam_evidence,
    )
    assert decision.status == "confirmed"
    assert "спама" in check(replace(result, spam_verification=decision))


def test_review_failure_cannot_auto_reject_even_for_other_low_scores():
    score = ScoreResult(
        location_score=0, business_score=0, intent_score=0,
        spam_score=100, ai_confidence=100, confidence_score=0,
        summary="Высокий риск", risk_flags=[],
        spam_verification=SpamVerification("error", "Вторая модель недоступна"),
    )
    assert check(score) is None


def test_core_location_and_business_thresholds_stay_unchanged():
    base = ScoreResult(
        location_score=0, business_score=95, intent_score=90,
        spam_score=0, ai_confidence=90, confidence_score=0,
        summary="Не в Грузии", risk_flags=[],
    )
    assert check(base) is None  # An unsupported low number is never enough.
    confirmed_location = replace(base, core_verifications=(CoreVerification(
        "location", "confirmed", "Явное отрицательное подтверждение",
    ),))
    assert "локации" in check(confirmed_location)
    base = replace(base, location_score=90, business_score=5)
    assert check(base) is None
    confirmed_business = replace(base, core_verifications=(CoreVerification(
        "business", "confirmed", "Явное отрицательное подтверждение",
    ),))
    assert "бизнес" in check(confirmed_business)


def test_two_separate_llm_requests_and_auditable_result(monkeypatch):
    answers = [score_payload(), verdict()]
    calls = []

    class FakeHTTPResponse:
        status_code = 200
        def __init__(self, content): self.content = content
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": self.content}}]}

    class FakeHTTPClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            return FakeHTTPResponse(answers.pop(0))

    monkeypatch.setattr("app.scoring.httpx.AsyncClient", FakeHTTPClient)
    scorer = new_scorer()

    async def run():
        result, raw = await scorer.score(location=LOCATION, business=BUSINESS, goal=GOAL)
        assert check(result) is None
        return await scorer.confirm_high_spam(
            score=result, raw=raw, location=LOCATION, business=BUSINESS, goal=GOAL,
        )
    result, raw = asyncio.run(run())
    assert check(result)
    assert len(calls) == 2
    assert "spam_score" not in calls[1]["messages"][1]["content"]
    assert "mass_advertising" not in calls[1]["messages"][1]["content"]
    assert "evidence" not in calls[1]["messages"][1]["content"]
    assert "Оцени анкету" in calls[0]["messages"][1]["content"]
    assert "ДАННЫЕ" in calls[1]["messages"][1]["content"]
    audit = json.loads(raw)
    assert audit["_gatekeeper_confidence"]["effective_ai_confidence"] == 100
    assert audit["_gatekeeper_spam_review"]["status"] == "confirmed"
    assert audit["_gatekeeper_spam_review"]["validated_primary_evidence"]


def test_reviewer_network_failure_routes_to_moderator():
    class BrokenReviewScorer:
        async def score(self, **kwargs):
            return parse_score_json(score_payload(), location=LOCATION,
                                    business=BUSINESS, goal=GOAL), "{}"
        async def review_rejection(self, **kwargs):
            raise httpx.ReadTimeout("Second check unavailable")

    candidate = Candidate(
        id=3, user_id=33, user_chat_id=33, target_chat_id=-100,
        status=CandidateStatus.SCORING.value, location_answer=LOCATION,
        business_answer=BUSINESS, goal_answer=GOAL,
    )
    class DB:
        statuses = []
        saved_score = None
        async def get_candidate(self, candidate_id):
            return candidate

        async def save_score(self, ident, score, raw):
            self.saved_score = score
            audit = json.loads(raw)["_gatekeeper_decision"]
            assert audit["stage3"]["allow_auto_reject"] is False
            # A proposal may be absent if the strict Python spam evidence
            # guard rejects the original model's sales claim.
            if audit["stage1_proposal"] is not None:
                assert audit["stage2_error_type"] == "ReadTimeout"
            return candidate
        async def set_status(self, ident, status, **kwargs):
            self.statuses.append(status)
            candidate.status = status.value
            return candidate
    db = DB()
    bot = SimpleNamespace(send_message=AsyncMock())
    handler = GatekeeperHandlers(settings=SimpleNamespace(
        auto_reject_spam_score=85, auto_reject_min_ai_confidence=80,
        auto_reject_core_score_max=15, admin_chat_id=-200,
    ), db=db, scorer=BrokenReviewScorer())
    asyncio.run(handler._score_and_route(bot, candidate))
    assert db.statuses == [CandidateStatus.PENDING_ADMIN]
    assert db.saved_score is not None
    assert db.saved_score.spam_verification is None or not db.saved_score.spam_verification.permits_auto_reject
    assert not hasattr(bot, "decline_chat_join_request")
    assert bot.send_message.await_count >= 1
