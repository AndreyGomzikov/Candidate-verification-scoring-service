"""Regression tests for auditable confidence and fail-closed auto-rejection."""
from __future__ import annotations

import asyncio
import json

from app.handlers import GatekeeperHandlers
from app.scoring import (
    AIScorer,
    ScoreResult,
    auto_reject_reason,
    parse_score_json,
)

LOCATION = "Сейчас живу в Тбилиси, Грузия."
BUSINESS = "Основатель и директор IT-студии, штат 12 сотрудников."
GOAL = "Хочу найти партнёров для совместных технологических проектов."


def response(**overrides):
    result = {
        "location_score": 95,
        "business_score": 88,
        "intent_score": 86,
        "spam_score": 2,
        "ai_confidence": 5,  # Legacy field must NOT control auto-rejection.
        "summary": "Основатель компании из Тбилиси.",
        "risk_flags": [],
        "confidence_evidence": {
            "location": "живу в Тбилиси",
            "business": "Основатель и директор IT-студии",
            "goal": "найти партнёров",
        },
        "contradiction_quotes": [],
        "uncertainty_quotes": [],
    }
    result.update(overrides)
    return json.dumps(result, ensure_ascii=False)


def evaluate(content=None, *, location=LOCATION, business=BUSINESS, goal=GOAL):
    return parse_score_json(
        content or response(), location=location, business=business, goal=goal
    )


def check_rejection(score):
    return auto_reject_reason(
        score,
        spam_threshold=85,
        core_score_max=15,
        min_ai_confidence=80,
    )


def test_complete_traceable_application_has_deterministic_max_confidence():
    score = evaluate()
    assert score.ai_confidence == 100
    assert score.confidence_details is not None
    detail = score.confidence_details
    assert (detail.completeness, detail.specificity, detail.consistency, detail.certainty) == (
        100, 100, 100, 100
    )
    assert detail.evidence_verified
    assert detail.formula_score == 100
    assert set(detail.evidence_quotes) == {"location", "business", "goal"}


def test_model_ai_confidence_is_ignored_in_both_directions():
    confident = evaluate(response(ai_confidence=0))
    unconfident = evaluate(response(ai_confidence=100))
    assert confident.ai_confidence == unconfident.ai_confidence == 100


def test_missing_or_false_evidence_fails_closed_even_for_high_spam_score():
    invalid = response(
        spam_score=99,
        ai_confidence=100,
        confidence_evidence={
            "location": "живу в Нью-Йорке",  # Not in candidate's answer.
            "business": "Основатель и директор IT-студии",
            "goal": "найти партнёров",
        },
    )
    score = evaluate(invalid)
    assert score.confidence_details.formula_score == 90
    assert score.ai_confidence == 79
    assert not score.confidence_details.evidence_verified
    assert check_rejection(score) is None


def test_legacy_response_with_high_model_confidence_cannot_auto_reject():
    legacy = json.loads(response(spam_score=100, ai_confidence=100))
    del legacy["confidence_evidence"]
    del legacy["contradiction_quotes"]
    del legacy["uncertainty_quotes"]
    score = evaluate(json.dumps(legacy, ensure_ascii=False))
    assert score.ai_confidence < 80
    assert check_rejection(score) is None


def test_inconsistent_answers_never_auto_reject_even_with_quotes():
    contradictory = response(
        spam_score=99,
        contradiction_quotes=["живу в Тбилиси", "Основатель и директор IT-студии"],
    )
    result = evaluate(contradictory)
    assert result.confidence_details.consistency == 0
    assert result.confidence_details.formula_score == 75
    assert result.ai_confidence == 75
    assert check_rejection(result) is None


def test_malformed_contradiction_quotes_fails_closed():
    result = evaluate(response(spam_score=99, contradiction_quotes=["неизвестное утверждение"]))
    assert not result.confidence_details.evidence_verified
    assert result.confidence_details.consistency == 0
    assert result.ai_confidence < 80
    assert check_rejection(result) is None


def test_uncertainty_is_found_from_text_even_if_model_omits_it():
    vague_goal = "Возможно, хочу найти партнёров для совместных проектов."
    result = evaluate(
        response(
            spam_score=99,
            confidence_evidence={
                "location": "живу в Тбилиси",
                "business": "Основатель и директор IT-студии",
                "goal": "найти партнёров",
            },
        ),
        goal=vague_goal,
    )
    assert result.confidence_details.certainty == 65
    assert result.confidence_details.formula_score == 95
    assert result.ai_confidence == 79
    assert check_rejection(result) is None


def test_missing_location_answer_prevents_auto_rejection():
    result = evaluate(response(location_score=0, spam_score=99), location="?")
    assert result.confidence_details.completeness == 67
    assert result.ai_confidence <= 79
    assert check_rejection(result) is None


def test_high_spam_requires_independent_confirmation_even_with_high_confidence():
    result = evaluate(response(spam_score=98))
    assert result.ai_confidence == 100
    assert check_rejection(result) is None


def test_admin_card_explains_computed_factors_and_failed_provenance():
    from types import SimpleNamespace
    from app.models import Candidate
    item = Candidate(
        id=2, user_id=10, user_chat_id=10, target_chat_id=-100,
        first_name="Тест", location_answer=LOCATION,
        business_answer=BUSINESS, goal_answer=GOAL,
    )
    handler = GatekeeperHandlers(
        settings=SimpleNamespace(), db=SimpleNamespace(), scorer=SimpleNamespace()
    )
    score = evaluate(response(confidence_evidence={"location": "чужая цитата"}))
    card = handler._build_admin_card(item, score=score)
    assert "Основания уверенности" in card
    assert "автоматический отказ запрещён" in card
    assert "Уверенность оценки" in card
    assert len(card) <= 3900


def test_raw_json_has_explicit_formula_and_quoted_evidence(monkeypatch):
    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": response(spam_score=90)}}]}

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("app.scoring.httpx.AsyncClient", FakeClient)
    scorer = AIScorer(
        base_url="https://example.invalid", api_key="fake",
        model="test", timeout_seconds=1,
    )
    score, stored_raw = asyncio.run(scorer.score(
        location=LOCATION, business=BUSINESS, goal=GOAL
    ))
    audit = json.loads(stored_raw)["_gatekeeper_confidence"]
    assert score.ai_confidence == audit["effective_ai_confidence"] == 100
    assert audit["rubric_version"] == 1
    assert audit["formula_weights"] == {
        "completeness": 0.3, "specificity": 0.3,
        "consistency": 0.25, "certainty": 0.15,
    }
    assert audit["evidence_quotes"]["location"] == "живу в Тбилиси"
