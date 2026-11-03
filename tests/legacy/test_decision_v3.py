from ._adapter import score_as_current as ScoreResult
"""Contract v3 regression: default to a human when model claims absence."""
import json

import pytest

from app.decision import decide_auto_rejection, parse_semantic_review
from app.scoring import propose_auto_rejection

LOCATION_NO = "Я сейчас не в Грузии и не планирую приезжать ближайшие годы"
BUSINESS_NO = "У меня нет бизнеса, нет проектов, нигде не работаю"
PARAMS = dict(spam_threshold=85, core_score_max=15, min_ai_confidence=80)
NEUTRAL = "Нет дополнительных сведений"


def score(**kwargs):
    data = dict(location_score=90, business_score=90, intent_score=90,
                spam_score=0, ai_confidence=99, confidence_score=85,
                summary="Проверка", risk_flags=[])
    data.update(kwargs)
    return ScoreResult(**data)


def review(criterion, quote, *, extra=None):
    statuses = {f: "no_relevant_information" for f in ("location", "business", "goal")}
    statuses[criterion] = "supports_rejection"
    data = {
        "schema_version": 3, "verdict": "reject", "criterion": criterion,
        "spam_type": "none", "field_checks": statuses,
        "no_relevance_basis": {f: ("explicit_neutral" if statuses[f] == "no_relevant_information"
                                   else "not_applicable") for f in statuses},
        "field_evidence": {f: {"supporting_quotes": [quote] if f == criterion else [],
                               "contradicting_quotes": [], "uncertainty_reason": ""}
                           for f in statuses},
        "overall_consistency": "consistent", "eligibility_signal": "absent",
        "explanation_code": f"explicit_{criterion}_mismatch",
        "evidence_quote": quote,
    }
    if extra:
        data.update(extra)
    return data


def gate(criterion, answers, score_value, data):
    proposal = propose_auto_rejection(score_value, candidate_answers=answers, **PARAMS)
    return decide_auto_rejection(proposal=proposal, review=parse_semantic_review(
        json.dumps(data, ensure_ascii=False)), score=score_value,
        candidate_answers=answers, **PARAMS)


@pytest.mark.parametrize("goal", [
    "В декабре у меня конференция в Тбилиси, уже зарегистрировался",
    "Забронирован номер в гостинице Батуми на следующую неделю",
    "Встреча с партнёрами в Грузии назначена на пятницу",
    "Мне подтвердили бронь гостиницы в Тбилиси",
    "Уже забронировал билет на самолёт до Тбилиси",
    "В следующем месяце командировка в Батуми",
])
def test_both_ai_stages_claim_irrelevant_but_goal_contains_trip(goal):
    answers = (LOCATION_NO, NEUTRAL, goal)
    result = gate("location", answers, score(location_score=0,
                  location_mismatch_evidence=LOCATION_NO), review("location", LOCATION_NO))
    assert not result.allow_auto_reject
    assert result.reason is None


@pytest.mark.parametrize("goal", [
    "Я фотограф, снимаю свадьбы по заказу",
    "Занимаюсь репетиторством и даю частные уроки",
    "У меня небольшое кафе, обслуживаю клиентов",
    "Я инженер-строитель, выполняю проектирование",
    "Оказываю услуги разработки на заказ",
    "Руковожу небольшим агентством",
])
def test_both_ai_stages_claim_irrelevant_but_goal_contains_work(goal):
    answers = (NEUTRAL, BUSINESS_NO, goal)
    result = gate("business", answers, score(business_score=0,
                  business_mismatch_evidence=BUSINESS_NO), review("business", BUSINESS_NO))
    assert not result.allow_auto_reject
    assert result.reason is None


@pytest.mark.parametrize("explanation", [
    "Кандидат уже живёт в Тбилиси. Отказывать нельзя.",
    "Кандидат полностью подходит сообществу. Оснований для отказа нет.",
    "Кандидат отвечает критериям, следует одобрить заявку.",
    "Однозначное несоответствие критерию локации.",
])
def test_free_text_explanation_never_enters_decision_contract(explanation):
    data = review("location", LOCATION_NO)
    data["explanation"] = explanation
    with pytest.raises(ValueError, match="incomplete"):
        parse_semantic_review(json.dumps(data, ensure_ascii=False))


def test_old_v2_contract_abstains():
    data = review("location", LOCATION_NO)
    data.pop("schema_version")
    data.pop("no_relevance_basis")
    data["explanation"] = "Несоответствие"
    with pytest.raises(ValueError):
        parse_semantic_review(json.dumps(data, ensure_ascii=False))


@pytest.mark.parametrize("invalid", [None, "not_applicable", "explicit_neutral"])
def test_invalid_absence_basis_never_authorizes(invalid):
    data = review("location", LOCATION_NO)
    data["no_relevance_basis"]["goal"] = invalid
    answers = (LOCATION_NO, NEUTRAL, NEUTRAL)
    s = score(location_score=0, location_mismatch_evidence=LOCATION_NO)
    if invalid is None:
        with pytest.raises(ValueError):
            parse_semantic_review(json.dumps(data, ensure_ascii=False))
    else:
        expected = invalid == "explicit_neutral"
        assert gate("location", answers, s, data).allow_auto_reject is expected


def test_generated_reason_not_from_ai_and_uses_verified_quote():
    s = score(location_score=0, location_mismatch_evidence=LOCATION_NO)
    result = gate("location", (LOCATION_NO, NEUTRAL, NEUTRAL), s,
                  review("location", LOCATION_NO))
    assert result.allow_auto_reject
    assert result.code == "confirmed"
    assert "два AI-этапа и Decision Gate" in result.reason
    assert LOCATION_NO in result.reason


def test_model_cannot_invent_neutrality_with_full_nonneutral_answer():
    s = score(location_score=0, location_mismatch_evidence=LOCATION_NO)
    result = gate("location", (LOCATION_NO, NEUTRAL, "Иногда сотрудничаю с партнёрами"),
                  s, review("location", LOCATION_NO))
    assert result.code == "unverified_absence_of_relevance"
