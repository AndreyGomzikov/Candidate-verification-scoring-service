"""Regression: two LLMs can agree on a *literal* but semantically false quote.

The gate must never trust `supports_rejection` in unrelated answer fields
for location/business: a supporting quote can actually prove eligibility.
"""
import json

import pytest

from app.decision import (
    decide_auto_rejection, parse_semantic_review, review_contract_failure,
)
from app.scoring import ScoreResult, propose_auto_rejection, semantic_review_confirms_rejection

LOCATION_NO = "Я сейчас не в Грузии и не планирую приезжать ближайшие годы"
BUSINESS_NO = "У меня нет бизнеса, нет проектов, нигде не работаю"
NEUTRAL = "Нет дополнительных сведений"
SPAM = "Переходите по ссылке и покупайте обучение, сейчас скидка 50%"
PARAMS = dict(spam_threshold=85, core_score_max=15, min_ai_confidence=80)
FIELDS = ("location", "business", "goal")


def score(**kwargs):
    defaults = dict(location_score=90, business_score=90, intent_score=90,
                    spam_score=0, ai_confidence=99, confidence_score=90,
                    summary="review", risk_flags=[])
    defaults.update(kwargs)
    return ScoreResult(**defaults)


def review_json(criterion, criterion_quote, *, additionally_supporting=None, spam_type="none"):
    additionally_supporting = additionally_supporting or {}
    checks = {field: "no_relevant_information" for field in FIELDS}
    evidence = {field: {"supporting_quotes": [], "contradicting_quotes": [],
                        "uncertainty_reason": ""} for field in FIELDS}
    main_field = "goal" if criterion == "spam" else criterion
    checks[main_field] = "supports_rejection"
    evidence[main_field]["supporting_quotes"] = [criterion_quote]
    for field, quote in additionally_supporting.items():
        checks[field] = "supports_rejection"
        evidence[field]["supporting_quotes"] = [quote]
    return json.dumps({
        "schema_version": 3,
        "verdict": "reject", "criterion": criterion, "spam_type": spam_type,
        "field_checks": checks,
        "field_evidence": evidence,
        "no_relevance_basis": {field: ("explicit_neutral" if checks[field] == "no_relevant_information"
                                       else "not_applicable") for field in FIELDS},
        "overall_consistency": "consistent", "eligibility_signal": "absent",
        "explanation_code": f"explicit_{criterion}_mismatch" if criterion != "spam" else "explicit_spam",
        "evidence_quote": criterion_quote,
    }, ensure_ascii=False)


@pytest.mark.parametrize("criterion,other_field,clarification", [
    ("location", "goal", "В декабре у меня конференция в Тбилиси, уже зарегистрировался"),
    ("location", "goal", "Забронировал гостиницу в Батуми на следующую неделю"),
    ("location", "goal", "Встреча с партнёрами в Грузии назначена на пятницу"),
    ("location", "business", "В следующем месяце командировка в Батуми"),
    ("location", "business", "Мне подтвердили бронь гостиницы в Тбилиси"),
    ("business", "goal", "Я фотограф, снимаю свадьбы по заказу"),
    ("business", "goal", "Занимаюсь репетиторством и даю частные уроки"),
    ("business", "goal", "У меня небольшое кафе, обслуживаю клиентов"),
    ("business", "location", "Я инженер-строитель, выполняю проектирование"),
    ("business", "location", "Оказываю услуги разработки на заказ"),
])
def test_matching_llm_mistakes_and_literally_valid_quote_cannot_decline(
    criterion, other_field, clarification,
):
    answers = dict.fromkeys(FIELDS, NEUTRAL)
    main_quote = LOCATION_NO if criterion == "location" else BUSINESS_NO
    answers[criterion] = main_quote
    answers[other_field] = clarification
    ordered_answers = tuple(answers[field] for field in FIELDS)
    s = (score(location_score=0, location_mismatch_evidence=LOCATION_NO)
         if criterion == "location" else
         score(business_score=0, business_mismatch_evidence=BUSINESS_NO))
    proposal = propose_auto_rejection(s, candidate_answers=ordered_answers, **PARAMS)
    # Some clarifications are already caught by stage 1. The remaining
    # scenarios reproduce the bug where both LLMs agree and only gate 3
    # could stop the erroneous rejection.
    raw = review_json(criterion, main_quote,
                      additionally_supporting={other_field: clarification})
    review = parse_semantic_review(raw)
    # Literal quote is present, all AI flags say 'reject' and no contradictions.
    # Despite formally consistent JSON, the other answer must force human review.
    assert review_contract_failure(review, ordered_answers) == "noncriterion_answer_requires_review"
    result = decide_auto_rejection(
        proposal=proposal, review=review, score=s,
        candidate_answers=ordered_answers, **PARAMS,
    )
    assert not result.allow_auto_reject
    assert result.code == (
        "noncriterion_answer_requires_review" if proposal is not None
        else "missing_proposal_or_review"
    )
    assert result.reason is None
    assert not semantic_review_confirms_rejection(
        raw, criterion=criterion, candidate_answers=ordered_answers,
    )


@pytest.mark.parametrize("criterion", ["location", "business"])
def test_unsupported_extra_field_blocked_even_if_claims_support(criterion):
    main_quote = LOCATION_NO if criterion == "location" else BUSINESS_NO
    answers = dict.fromkeys(FIELDS, NEUTRAL)
    answers[criterion] = main_quote
    other_field = "goal"
    raw = review_json(criterion, main_quote, additionally_supporting={other_field: NEUTRAL})
    s = (score(location_score=0, location_mismatch_evidence=LOCATION_NO)
         if criterion == "location" else
         score(business_score=0, business_mismatch_evidence=BUSINESS_NO))
    ordered = tuple(answers[field] for field in FIELDS)
    decision = decide_auto_rejection(
        proposal=propose_auto_rejection(s, candidate_answers=ordered, **PARAMS),
        review=parse_semantic_review(raw), score=s,
        candidate_answers=ordered, **PARAMS,
    )
    assert decision.code == "noncriterion_answer_requires_review"


@pytest.mark.parametrize("criterion", ["location", "business"])
def test_clear_core_violation_and_only_neutral_other_answers_still_declines(criterion):
    main_quote = LOCATION_NO if criterion == "location" else BUSINESS_NO
    answers = dict.fromkeys(FIELDS, NEUTRAL)
    answers[criterion] = main_quote
    ordered = tuple(answers[field] for field in FIELDS)
    s = (score(location_score=0, location_mismatch_evidence=LOCATION_NO)
         if criterion == "location" else
         score(business_score=0, business_mismatch_evidence=BUSINESS_NO))
    result = decide_auto_rejection(
        proposal=propose_auto_rejection(s, candidate_answers=ordered, **PARAMS),
        review=parse_semantic_review(review_json(criterion, main_quote)),
        score=s, candidate_answers=ordered, **PARAMS,
    )
    assert result.allow_auto_reject and result.code == "confirmed"


def test_spam_evidence_in_goal_remains_eligible_for_auto_reject():
    answers = (NEUTRAL, NEUTRAL, SPAM)
    s = score(spam_score=95, spam_type="advertising", spam_evidence=SPAM)
    result = decide_auto_rejection(
        proposal=propose_auto_rejection(s, candidate_answers=answers, **PARAMS),
        review=parse_semantic_review(review_json("spam", SPAM, spam_type="advertising")),
        score=s, candidate_answers=answers, **PARAMS,
    )
    assert result.allow_auto_reject and result.code == "confirmed"
