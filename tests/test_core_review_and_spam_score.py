"""Regression tests: grounded core rejections and unconfirmed spam penalties."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.core_review import (
    CoreEvidence, CoreVerification, parse_core_evidence, validate_core_review,
)
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus
from app.scoring import (
    AIScorer, ScoreResult, auto_reject_reason, calculate_confidence_score,
    finalize_candidate_score, parse_score_json,
)
from app.spam_review import SpamVerification


LOCATION = "Живу в Батуми, Грузия"
BUSINESS = "Владелец IT-компании с 12 сотрудниками"
GOAL = "Ищу деловых партнёров"
ABSENT = "Живу в Москве и не планирую приезжать в Грузию"
NO_BUSINESS = "Не занимаюсь бизнесом, не управляю и не веду проектов"


def make_payload(*, location_score=0, business_score=90, spam_score=5, core=None):
    return json.dumps({
        "location_score": location_score, "business_score": business_score,
        "intent_score": 90, "spam_score": spam_score,
        "summary": "Проверка кандидата", "risk_flags": [],
        "confidence_evidence": {
            "location": ABSENT if location_score == 0 else LOCATION,
            "business": BUSINESS if business_score > 15 else NO_BUSINESS,
            "goal": GOAL,
        },
        "contradiction_quotes": [], "uncertainty_quotes": [],
        "core_rejection_evidence": core or {},
        "spam_evidence": [],
    }, ensure_ascii=False)


def reason(score):
    return auto_reject_reason(
        score, spam_threshold=85, core_score_max=15, min_ai_confidence=80
    )


def scorer():
    return AIScorer(
        base_url="https://example.invalid/v1", api_key="test", model="test",
        timeout_seconds=1,
    )


def test_wrong_low_location_from_model_does_not_auto_reject():
    # The answer explicitly says Batumi, but first model wrongly says 0.
    model = json.loads(make_payload())
    model["core_rejection_evidence"] = {
        "location": {"quote": LOCATION, "explanation": "Это не Грузия, по мнению модели"}
    }
    model["confidence_evidence"]["location"] = LOCATION
    score = parse_score_json(json.dumps(model, ensure_ascii=False),
                             location=LOCATION, business=BUSINESS, goal=GOAL)
    assert score.ai_confidence == 100
    assert reason(score) is None
    primary = score.core_evidence[0]
    second = validate_core_review(json.dumps({
        "verdict": "not_confirmed", "field": "location", "quote": LOCATION,
        "reason": "Батуми явно находится на территории Грузии",
    }, ensure_ascii=False), field="location", answer=LOCATION, primary=primary)
    assert not second.permits_auto_reject
    assert reason(replace(score, core_verifications=(second,))) is None


def test_low_business_without_verified_negative_evidence_is_human_review():
    first = json.loads(make_payload(location_score=90, business_score=0))
    first["confidence_evidence"]["business"] = BUSINESS
    first["core_rejection_evidence"] = {
        "business": {"quote": "не занимаюсь ничем", "explanation": "Нет бизнеса"}
    }
    score = parse_score_json(json.dumps(first, ensure_ascii=False),
                             location=LOCATION, business=BUSINESS, goal=GOAL)
    assert score.ai_confidence == 100
    assert not score.core_evidence
    assert reason(score) is None


def test_valid_core_evidence_requires_matching_independent_confirmation():
    evidence = parse_core_evidence({"location": {
        "quote": ABSENT, "explanation": "Кандидат прямо отказался от поездки в Грузию",
    }}, answers={"location": ABSENT, "business": BUSINESS})
    assert len(evidence) == 1
    first = evidence[0]
    negative = validate_core_review(json.dumps({
        "verdict": "confirmed", "field": "location", "quote": ABSENT,
        "reason": "Кандидат явно не планирует поездку в Грузию",
    }, ensure_ascii=False), field="location", answer=ABSENT, primary=first)
    assert negative.permits_auto_reject
    assert "локации" in reason(ScoreResult(
        0, 90, 90, 5, 95, 44, "Отсутствие планов", [],
        core_verifications=(negative,),
    ))
    for fake_quote in ("живу в Тбилиси", "Не занимаюсь бизнесом"):
        bad = validate_core_review(json.dumps({
            "verdict": "confirmed", "field": "location", "quote": fake_quote,
            "reason": "Кандидат не планирует поездки в Грузию",
        }, ensure_ascii=False), field="location", answer=ABSENT, primary=first)
        assert not bad.permits_auto_reject
    mismatch = validate_core_review(json.dumps({
        "verdict": "confirmed", "field": "business", "quote": ABSENT,
        "reason": "Кандидат не планирует поездки в Грузию",
    }, ensure_ascii=False), field="location", answer=ABSENT, primary=first)
    assert not mismatch.permits_auto_reject


def test_second_core_request_is_blind_and_auditable():
    first = make_payload(core={"location": {
        "quote": ABSENT, "explanation": "Прямой отказ от посещения Грузии",
    }})
    original = parse_score_json(first, location=ABSENT, business=BUSINESS, goal=GOAL)
    calls = []
    async def fake_complete(messages):
        calls.append(messages)
        return json.dumps({
            "verdict": "confirmed", "field": "location", "quote": ABSENT,
            "reason": "Кандидат прямо не планирует приезжать в Грузию",
        }, ensure_ascii=False)
    engine = scorer()
    engine._complete_json = fake_complete
    score, raw = asyncio.run(engine.confirm_low_core(
        score=original, raw=first, location=ABSENT,
        business=BUSINESS, goal=GOAL, core_score_max=15,
    ))
    assert reason(score) is not None
    assert len(calls) == 1
    assert "location_score" not in calls[0][1]["content"]
    assert "Прямой отказ от посещения" not in calls[0][1]["content"]
    assert json.loads(raw)["_gatekeeper_core_review"]["reviews"][0]["status"] == "confirmed"


def test_second_core_unavailable_never_automatically_rejects():
    original = parse_score_json(make_payload(core={"location": {
        "quote": ABSENT, "explanation": "Прямой отказ от приезда в Грузию",
    }}), location=ABSENT, business=BUSINESS, goal=GOAL)
    async def broken(messages):
        raise TimeoutError("LLM timeout")
    engine = scorer()
    engine._complete_json = broken
    score, raw = asyncio.run(engine.confirm_low_core(
        score=original, raw=make_payload(), location=ABSENT,
        business=BUSINESS, goal=GOAL, core_score_max=15,
    ))
    assert score.core_verifications[0].status == "error"
    assert reason(score) is None
    assert json.loads(raw)["_gatekeeper_core_review"]["reviews"][0]["status"] == "error"


def test_unconfirmed_high_spam_does_not_depress_displayed_score_and_raw_retained():
    initial = ScoreResult(90, 85, 0, 90, 95,
        calculate_confidence_score(90, 85, 0, 90), "Кандидат", [],
        spam_verification=SpamVerification("not_confirmed", "Нет убедительных оснований"),
    )
    new, raw = finalize_candidate_score(initial, "{}", spam_threshold=85)
    assert new.spam_score == 90  # Do not overwrite the model assessment.
    assert new.preliminary_confidence_score == calculate_confidence_score(90, 85, 0, 90)
    assert new.confidence_score == calculate_confidence_score(90, 85, 0, 0)
    assert new.spam_penalty_withheld
    report = json.loads(raw)["_gatekeeper_score_adjustment"]
    assert report["original_confidence_score"] == new.preliminary_confidence_score
    assert report["displayed_confidence_score"] == new.confidence_score
    assert report["model_spam_score"] == 90
    assert reason(new) is None


def test_confirmed_high_spam_keeps_original_penalty_and_can_auto_reject():
    initial = ScoreResult(90, 85, 0, 90, 95, 0, "Спам", [],
                          spam_verification=SpamVerification("confirmed", "Массовая рассылка"))
    new, raw = finalize_candidate_score(initial, "{}", spam_threshold=85)
    assert not new.spam_penalty_withheld
    assert new.confidence_score == calculate_confidence_score(90, 85, 0, 90)
    assert json.loads(raw)["_gatekeeper_score_adjustment"]["spam_penalty_withheld"] is False
    assert "спама" in reason(new)


def test_moderator_card_explicitly_distinguishes_both_scores():
    from app.config import Settings
    s = Settings(
        bot_token="test", target_chat_id=-100, admin_chat_id=-200,
        admin_user_ids={7}, database_url="sqlite+aiosqlite:///:memory:",
        llm_base_url="https://example.invalid", llm_api_key="test", llm_model="test",
        llm_timeout_seconds=1, auto_reject_spam_score=85,
        auto_reject_core_score_max=15, auto_reject_min_ai_confidence=80,
        max_answer_length=2500,
    )
    score = ScoreResult(90, 85, 0, 90, 95, 67, "Требует проверки", [],
                        preliminary_confidence_score=20, spam_penalty_withheld=True)
    candidate = Candidate(id=1, user_id=17, user_chat_id=17, target_chat_id=-100,
        location_answer=LOCATION, business_answer=BUSINESS, goal_answer=GOAL)
    card = GatekeeperHandlers(settings=s, db=object(), scorer=object())._build_admin_card(
        candidate, score=score,
    )
    assert "без неподтверждённого спам-штрафа" in card
    assert "Исходный Score" in card
    assert "Спам-риск: 90" in card


def test_false_low_location_routes_to_moderator_without_decline():
    """The complete handler must not decline a Batumi applicant for an AI error."""
    from app.config import Settings
    s = Settings(
        bot_token="test", target_chat_id=-100, admin_chat_id=-200,
        admin_user_ids={7}, database_url="sqlite+aiosqlite:///:memory:",
        llm_base_url="https://example.invalid", llm_api_key="test", llm_model="test",
        llm_timeout_seconds=1, auto_reject_spam_score=85,
        auto_reject_core_score_max=15, auto_reject_min_ai_confidence=80,
        max_answer_length=2500,
    )
    primary_raw = json.loads(make_payload())
    primary_raw["confidence_evidence"]["location"] = LOCATION
    primary_raw["core_rejection_evidence"] = {"location": {
        "quote": LOCATION, "explanation": "Ошибочное утверждение первой модели о локации",
    }}
    primary_raw = json.dumps(primary_raw, ensure_ascii=False)
    item = Candidate(
        id=8, user_id=21, user_chat_id=21, target_chat_id=-100,
        status=CandidateStatus.SCORING.value, stage=4,
        location_answer=LOCATION, business_answer=BUSINESS, goal_answer=GOAL,
    )

    class FakeScorer:
        async def score(self, **kwargs):
            return parse_score_json(primary_raw, **kwargs), primary_raw
        async def confirm_low_core(self, **kwargs):
            data = kwargs["score"]
            review = CoreVerification(
                "location", "not_confirmed", "Кандидат живёт в Батуми"
            )
            return replace(data, core_verifications=(review,)), kwargs["raw"]

    class FakeDB:
        stored = None
        statuses = []
        async def save_score(self, candidate_id, score, raw):
            self.stored = score
            assert json.loads(raw)["_gatekeeper_score_adjustment"]["version"] == 1
            return item
        async def set_status(self, candidate_id, status, **kwargs):
            self.statuses.append(status)
            item.status = status.value
            return item

    bot = SimpleNamespace(send_message=AsyncMock(), decline_chat_join_request=AsyncMock())
    db = FakeDB()
    handler = GatekeeperHandlers(settings=s, db=db, scorer=FakeScorer())
    asyncio.run(handler._score_and_route(bot, item))
    assert db.statuses[-1] == CandidateStatus.PENDING_ADMIN
    # Batumi is a city of Georgia: no strict stage-1 proposal, so stage 2 is not called.
    assert not db.stored.core_verifications
    assert db.stored.decision_confidences == () or all(
        not x.evidence_verified for x in db.stored.decision_confidences
    )
    bot.decline_chat_join_request.assert_not_awaited()


def test_low_business_confirmed_with_grounded_citation():
    first = CoreEvidence("business", NO_BUSINESS, "Кандидат отрицает деловую роль")
    review = validate_core_review(json.dumps({
        "verdict": "confirmed", "field": "business", "quote": NO_BUSINESS,
        "reason": "Кандидат прямо говорит об отсутствии бизнеса и проектов",
    }, ensure_ascii=False), field="business", answer=NO_BUSINESS, primary=first)
    score = ScoreResult(90, 0, 50, 5, 95, 33, "Не занимается бизнесом", [],
                        core_verifications=(review,))
    assert "бизнес" in reason(score)


def test_high_spam_without_second_confirmation_stays_preliminary():
    base = ScoreResult(90, 90, 80, 90, 75, 15, "Нужна ручная проверка", [])
    result, audit = finalize_candidate_score(base, "{}", spam_threshold=85)
    assert result.spam_penalty_withheld
    assert reason(result) is None
    assert result.confidence_score == calculate_confidence_score(90, 90, 80, 0)
    assert json.loads(audit)["_gatekeeper_score_adjustment"]["model_spam_score"] == 90
