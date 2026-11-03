"""Regression tests for the LLM confidence unit mismatch seen in live Telegram tests."""

import asyncio
import json

import pytest

from app.scoring import AIScorer, parse_score_json, propose_auto_rejection


def raw_score(confidence):
    return json.dumps(
        {
            "location_score": 0,
            "business_score": 20,
            "intent_score": 10,
            "spam_score": 15,
            "spam_type": "low_effort",
            "spam_evidence": "",
            "location_mismatch_evidence": (
                "Я сейчас не в Грузии и не планирую приезжать ближайшие годы."
            ),
            "business_mismatch_evidence": "",
            "ai_confidence": confidence,
            "summary": "Тестовая анкета",
            "risk_flags": [],
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0),
        (0.7, 70),
        (0.85, 85),
        (0.799, 79),
        (0.8, 80),
        (0.29, 29),
        (0.999, 99),
        (0.001, 0),
        (1, 1),
        (1.0, 1),
        (70, 70),
        (70.0, 70),
        (100, 100),
        ("0.7", 70),
        ("85", 85),
        ("70.0", 70),
    ],
)
def test_ai_confidence_is_normalized_without_upward_rounding(raw, expected):
    result = parse_score_json(raw_score(raw))
    assert result.ai_confidence == expected
    # No collateral change to other scoring fields or the weighted formula.
    assert (result.location_score, result.business_score,
            result.intent_score, result.spam_score) == (0, 20, 10, 15)
    assert result.confidence_score == 3


@pytest.mark.parametrize(
    "bad",
    [
        None, True, False, -1, -0.2, 101, 1.01, 70.5,
        float("nan"), float("inf"), -float("inf"),
        "", "70%", "0,7", "70,5", "nan", "inf", "1e2", "01", " 0.7% ",
        {}, [],
    ],
)
def test_invalid_ai_confidence_cannot_be_silently_converted(bad):
    with pytest.raises(ValueError, match="ai_confidence"):
        parse_score_json(raw_score(bad))


def test_confidence_threshold_is_not_rounded_up_to_auto_reject():
    answers = (
        "Я сейчас не в Грузии и не планирую приезжать ближайшие годы.",
        "Нет дополнительных сведений.",
        "Нет дополнительных сведений.",
    )
    opts = dict(spam_threshold=85, core_score_max=15, min_ai_confidence=80,
                candidate_answers=answers)
    assert propose_auto_rejection(parse_score_json(raw_score(0.7)), **opts) is None
    assert propose_auto_rejection(parse_score_json(raw_score(0.799)), **opts) is None
    proposed = propose_auto_rejection(parse_score_json(raw_score(0.8)), **opts)
    assert proposed is not None
    assert proposed.criterion == "location"
    # A proposal is only stage 1; the independent review and decision gate
    # must still be successful before calling Telegram's decline endpoint.


def test_ai_scorer_returns_normalized_score_and_original_json(monkeypatch):
    raw = raw_score(0.7)

    async def fake_request(self, messages):
        assert messages[0]["role"] == "system"
        return raw

    monkeypatch.setattr(AIScorer, "_request_json", fake_request)
    scorer = AIScorer(
        base_url="https://invalid.example", api_key="test", model="test",
        timeout_seconds=10,
    )
    score, returned_raw = asyncio.run(
        scorer.score(location="Test", business="Test", goal="Test")
    )
    assert score.ai_confidence == 70
    assert returned_raw == raw
