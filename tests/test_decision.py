"""Tests of the third-stage decision gate, the *only* decline authorization."""
import json
from dataclasses import replace

import pytest

from app.decision import (
    RejectionProposal, decide_auto_rejection, parse_semantic_review,
)
from app.scoring import ScoreResult, propose_auto_rejection

LOCATION_NO = "Я сейчас не в Грузии и не планирую приезжать ближайшие годы"
BUSINESS_NO = "У меня нет бизнеса, нет проектов, нигде не работаю"
AD = "Переходите по ссылке и покупайте обучение, сейчас скидка 50%"
OTHER = "Нет дополнительных сведений"
PARAMS = dict(spam_threshold=85, core_score_max=15, min_ai_confidence=80)


def scoring(**changes):
    data = dict(location_score=90, business_score=90, intent_score=90,
                spam_score=0, ai_confidence=97, confidence_score=90,
                summary="Заявка", risk_flags=[])
    data.update(changes)
    return ScoreResult(**data)


def raw_review(criterion, evidence, *, verdict="reject", spam_type="none",
               field_checks=None, explanation=None):
    checks = {"location": "no_relevant_information",
              "business": "no_relevant_information",
              "goal": "no_relevant_information"}
    checks["goal" if criterion == "spam" else criterion] = "supports_rejection"
    if field_checks:
        checks.update(field_checks)
    evidence_map = {
        field: {"supporting_quotes": [evidence] if status == "supports_rejection" else [],
                "contradicting_quotes": [], "uncertainty_reason": ""}
        for field, status in checks.items() if field in ("location", "business", "goal")
    }
    payload = {
        "schema_version": 3, "verdict": verdict, "criterion": criterion, "spam_type": spam_type,
        "field_checks": checks, "field_evidence": evidence_map,
        "no_relevance_basis": {f: "explicit_neutral" if status == "no_relevant_information"
                               else "not_applicable" for f, status in checks.items()},
        "overall_consistency": "consistent", "eligibility_signal": "absent",
        "explanation_code": {"location": "explicit_location_mismatch",
                             "business": "explicit_business_mismatch",
                             "spam": "explicit_spam"}[criterion],
        "evidence_quote": evidence,
    }
    if explanation is not None:
        payload["explanation"] = explanation  # forbidden unstructured field
    return json.dumps(payload, ensure_ascii=False)


def scenario(criterion):
    if criterion == "location":
        return (LOCATION_NO, OTHER, OTHER), scoring(
            location_score=0, location_mismatch_evidence=LOCATION_NO
        ), raw_review("location", LOCATION_NO)
    if criterion == "business":
        return (OTHER, BUSINESS_NO, OTHER), scoring(
            business_score=0, business_mismatch_evidence=BUSINESS_NO
        ), raw_review("business", BUSINESS_NO)
    if criterion == "spam":
        return (OTHER, OTHER, AD), scoring(
            spam_score=95, spam_type="advertising", spam_evidence=AD
        ), raw_review("spam", AD, spam_type="advertising")
    raise AssertionError(criterion)


def evaluate(criterion, *, answers=None, score=None, review=None, proposal=None,
             proposal_override=False, **params):
    original_answers, original_score, raw = scenario(criterion)
    answers = answers if answers is not None else original_answers
    score = score if score is not None else original_score
    config = dict(PARAMS, **params)
    if proposal_override:
        p = proposal
    else:
        p = propose_auto_rejection(score, candidate_answers=answers, **config)
    r = parse_semantic_review(review if review is not None else raw)
    return decide_auto_rejection(
        proposal=p, review=r, score=score, candidate_answers=answers, **config
    )


@pytest.mark.parametrize("criterion", ["location", "business", "spam"])
def test_both_models_and_gate_agree_on_obvious_violation(criterion):
    result = evaluate(criterion)
    assert result.allow_auto_reject is True
    assert result.code == "confirmed"


@pytest.mark.parametrize("criterion", ["location", "business", "spam"])
def test_missing_or_malformed_second_review_abstains(criterion):
    answers, score, _ = scenario(criterion)
    proposal = propose_auto_rejection(score, candidate_answers=answers, **PARAMS)
    for review in (None, True, "reject", {}):
        result = decide_auto_rejection(
            proposal=proposal, review=review, score=score,
            candidate_answers=answers, **PARAMS
        )
        assert not result.allow_auto_reject


@pytest.mark.parametrize("criterion", ["location", "business", "spam"])
@pytest.mark.parametrize("status", ["contradicts", "uncertain"])
@pytest.mark.parametrize("field", ["location", "business", "goal"])
def test_any_ambiguity_in_any_answer_abstains(criterion, status, field):
    answers, score, raw = scenario(criterion)
    payload = json.loads(raw)
    payload["field_checks"][field] = status
    decision = evaluate(criterion, review=json.dumps(payload, ensure_ascii=False))
    assert not decision.allow_auto_reject


@pytest.mark.parametrize("criterion", ["location", "business", "spam"])
def test_manual_review_verdict_abstains(criterion):
    answers, score, raw = scenario(criterion)
    payload = json.loads(raw)
    payload["verdict"] = "review"
    assert not evaluate(criterion, review=json.dumps(payload)).allow_auto_reject


@pytest.mark.parametrize("criterion, opposite", [
    ("location", "business"), ("business", "location"), ("spam", "business")
])
def test_different_criterion_between_models_abstains(criterion, opposite):
    _, _, raw = scenario(criterion)
    payload = json.loads(raw)
    payload["criterion"] = opposite
    assert not evaluate(criterion, review=json.dumps(payload)).allow_auto_reject


def test_spam_subcategory_mismatch_abstains():
    _, _, raw = scenario("spam")
    payload = json.loads(raw)
    payload["spam_type"] = "fraud"
    assert evaluate("spam", review=json.dumps(payload)).code == "spam_type_mismatch"


@pytest.mark.parametrize("field", ["location", "business", "goal"])
def test_incomplete_questionnaire_always_goes_to_admin(field):
    answers, score, review = scenario("spam")
    altered = list(answers)
    altered[("location", "business", "goal").index(field)] = ""
    assert not evaluate("spam", answers=altered).allow_auto_reject


@pytest.mark.parametrize("criterion", ["location", "business", "spam"])
def test_fake_quote_from_second_model_abstains(criterion):
    _, _, raw = scenario(criterion)
    payload = json.loads(raw)
    payload["evidence_quote"] = "Кандидат дословно этого нигде не говорил"
    assert not evaluate(criterion, review=json.dumps(payload, ensure_ascii=False)).allow_auto_reject


def test_spam_second_quote_must_prove_same_violation():
    _, _, raw = scenario("spam")
    payload = json.loads(raw)
    payload["evidence_quote"] = OTHER
    assert evaluate("spam", review=json.dumps(payload, ensure_ascii=False)).code == "quote_not_in_supporting_evidence"


@pytest.mark.parametrize("criterion", ["location", "business", "spam"])
def test_forged_first_stage_proposal_abstains(criterion):
    answers, score, raw = scenario(criterion)
    expected = propose_auto_rejection(score, candidate_answers=answers, **PARAMS)
    assert expected is not None
    forged = replace(expected, reason="Иная причина")
    assert not evaluate(criterion, proposal=forged, proposal_override=True).allow_auto_reject


def test_higher_threshold_disables_spam_decline():
    _, score, _ = scenario("spam")
    assert not evaluate("spam", score=score, spam_threshold=101).allow_auto_reject


def test_low_confidence_disables_auto_decline():
    _, score, _ = scenario("location")
    assert not evaluate("location", score=replace(score, ai_confidence=79)).allow_auto_reject


@pytest.mark.parametrize("payload", [
    '{}', '[]', '{invalid',
    raw_review("location", LOCATION_NO, explanation=""),
    raw_review("location", LOCATION_NO, explanation=" "),
    raw_review("location", LOCATION_NO).replace('"evidence_quote":', '"unexpected": 123, "evidence_quote":'),
    raw_review("spam", AD, spam_type="nonexistent"),
])
def test_strict_parser_rejects_invalid_review(payload):
    with pytest.raises((ValueError, TypeError)):
        parse_semantic_review(payload)


def test_quote_implicitly_truncating_context_does_not_authorize():
    answers = (OTHER, OTHER, "Не покупайте этот курс! " + AD)
    score = scoring(spam_score=99, spam_type="advertising", spam_evidence=AD)
    proposal = propose_auto_rejection(score, candidate_answers=answers, **PARAMS)
    assert proposal is None  # advertising is explicitly *warned against*
    result = decide_auto_rejection(
        proposal=proposal, review=parse_semantic_review(raw_review(
            "spam", AD, spam_type="advertising")), score=score,
        candidate_answers=answers, **PARAMS,
    )
    assert not result.allow_auto_reject


def test_cross_field_contradiction_vetoes_even_when_models_claim_reject():
    answers = (LOCATION_NO, OTHER, "Завтра лечу в Тбилиси")
    score = scoring(location_score=0, location_mismatch_evidence=LOCATION_NO)
    proposal = propose_auto_rejection(score, candidate_answers=answers, **PARAMS)
    # A recognized travel signal also cancels the stage-one proposal.
    assert proposal is None
    result = decide_auto_rejection(
        proposal=proposal, review=parse_semantic_review(raw_review(
            "location", LOCATION_NO)), score=score, candidate_answers=answers,
        **PARAMS,
    )
    assert not result.allow_auto_reject


# Version 2 of the independent semantic contract. Unknown/old/incoherent model
# outputs must never authorize an automatic rejection.
@pytest.mark.parametrize("field", ["location", "business", "goal"])
def test_missing_per_field_evidence_abstains(field):
    _, _, raw = scenario("location")
    payload = json.loads(raw)
    del payload["field_evidence"][field]
    with pytest.raises(ValueError):
        parse_semantic_review(json.dumps(payload, ensure_ascii=False))


def test_old_semantic_contract_is_rejected():
    _, _, raw = scenario("location")
    payload = json.loads(raw)
    for field in ("field_evidence", "overall_consistency", "eligibility_signal", "explanation_code"):
        del payload[field]
    with pytest.raises(ValueError, match="incomplete"):
        parse_semantic_review(json.dumps(payload, ensure_ascii=False))


@pytest.mark.parametrize("field", ["location", "business", "goal"])
def test_wrong_field_quote_is_not_accepted(field):
    answers, score, raw = scenario("location")
    payload = json.loads(raw)
    payload["field_checks"][field] = "supports_rejection"
    payload["no_relevance_basis"][field] = "not_applicable"
    payload["field_evidence"][field]["supporting_quotes"] = [LOCATION_NO]
    # The location quote exists only in field location.
    if field != "location":
        assert evaluate("location", review=json.dumps(payload, ensure_ascii=False)).code == "unverifiable_field_quote"
    else:
        assert evaluate("location", review=json.dumps(payload, ensure_ascii=False)).allow_auto_reject


def test_no_relevant_status_cannot_hide_a_counter_quote():
    _, _, raw = scenario("location")
    payload = json.loads(raw)
    payload["field_evidence"]["goal"]["contradicting_quotes"] = [OTHER]
    assert evaluate("location", review=json.dumps(payload, ensure_ascii=False)).code == "unreported_relevant_evidence"


def test_support_status_without_supporting_quote_is_not_accepted():
    _, _, raw = scenario("business")
    payload = json.loads(raw)
    payload["field_evidence"]["business"]["supporting_quotes"] = []
    assert evaluate("business", review=json.dumps(payload, ensure_ascii=False)).code == "inconsistent_support_evidence"


@pytest.mark.parametrize("flag,value", [
    ("overall_consistency", "contradicted"),
    ("overall_consistency", "uncertain"),
    ("eligibility_signal", "present"),
    ("eligibility_signal", "uncertain"),
])
def test_global_conflict_and_eligibility_flags_veto(flag, value):
    _, _, raw = scenario("location")
    payload = json.loads(raw)
    payload[flag] = value
    assert evaluate("location", review=json.dumps(payload, ensure_ascii=False)).code == "global_conflict_or_uncertainty"


@pytest.mark.parametrize("explanation", [
    "В ответе о цели указаны планы приезда в Грузию; есть противоречие с локацией.",
    "Кандидат инвестор и может соответствовать критерию бизнес-сообщества.",
    "Неоднозначная формулировка — следует передать администратору.",
    "Есть командировка в Тбилиси, но формальные поля сообщают об отсутствии поездок.",
])
def test_free_text_explanation_with_conflicting_information_vetoes(explanation):
    _, _, raw = scenario("location")
    payload = json.loads(raw)
    payload["explanation"] = explanation
    with pytest.raises(ValueError, match="incomplete"):
        parse_semantic_review(json.dumps(payload, ensure_ascii=False))


def test_wrong_structured_explanation_code_vetoes():
    _, _, raw = scenario("business")
    payload = json.loads(raw)
    payload["explanation_code"] = "explicit_location_mismatch"
    assert evaluate("business", review=json.dumps(payload, ensure_ascii=False)).code == "explanation_code_mismatch"


def test_manually_constructed_nonmatching_quotes_are_vetoed():
    _, _, raw = scenario("location")
    payload = json.loads(raw)
    payload["evidence_quote"] = "Не та цитата из анкеты"
    assert evaluate("location", review=json.dumps(payload, ensure_ascii=False)).code == "quote_not_in_supporting_evidence"


def test_reported_contradiction_with_exact_quote_requires_admin():
    answers, score, raw = scenario("location")
    answers = (answers[0], "Работаю в другом городе", "Завтра лечу в Тбилиси")
    payload = json.loads(raw)
    payload["field_checks"]["goal"] = "contradicts"
    payload["no_relevance_basis"]["goal"] = "not_applicable"
    payload["field_evidence"]["goal"]["contradicting_quotes"] = [answers[2]]
    payload["overall_consistency"] = "contradicted"
    payload["eligibility_signal"] = "present"
    payload["explanation_code"] = "needs_review"
    payload["verdict"] = "review"
    proposal = propose_auto_rejection(score, candidate_answers=answers, **PARAMS)
    parsed = parse_semantic_review(json.dumps(payload, ensure_ascii=False))
    decision = decide_auto_rejection(proposal=proposal, review=parsed, score=score,
                                     candidate_answers=answers, **PARAMS)
    assert not decision.allow_auto_reject


@pytest.mark.parametrize("bad_quotes", [
    ["Анкета содержит несуществующее доказательство"],
    ["сейчас не в Грузии... не планирую приезжать"],
    [1],
    "Не массив",
])
def test_invalid_support_evidence_is_not_authorized(bad_quotes):
    _, _, raw = scenario("location")
    payload = json.loads(raw)
    payload["field_evidence"]["location"]["supporting_quotes"] = bad_quotes
    raw_payload = json.dumps(payload, ensure_ascii=False)
    try:
        review = parse_semantic_review(raw_payload)
    except (ValueError, TypeError):
        return
    assert not evaluate("location", review=raw_payload).allow_auto_reject


@pytest.mark.parametrize("criterion", ["location", "business", "spam"])
def test_full_version_two_contract_still_authorizes_obvious_violation(criterion):
    assert evaluate(criterion).code == "confirmed"
