"""Regression: contradictory qualifications inside the PRIMARY answer are not
made safe by two LLMs agreeing on an incomplete negative excerpt.

For core rejections, only complete, recognized, explicit negative statements
are eligible for an automatic decline. Everything else requires manual review.
"""
import json

import pytest

from app.decision import (
    RejectionProposal, decide_auto_rejection, parse_semantic_review,
    review_contract_failure,
)
from app.scoring import ScoreResult, propose_auto_rejection

LOCATION_NEGATIVE = 'Я сейчас не в Грузии и не планирую приезжать ближайшие годы'
BUSINESS_NEGATIVE = 'У меня нет бизнеса, нет проектов, нигде не работаю'
NEUTRAL = 'Нет дополнительных сведений'
PARAMS = dict(spam_threshold=85, core_score_max=15, min_ai_confidence=80)


def score(criterion, evidence):
    fields = dict(location_score=90, business_score=90, intent_score=80,
                  spam_score=0, ai_confidence=99, confidence_score=80,
                  summary='Test', risk_flags=[])
    fields['location_score' if criterion == 'location' else 'business_score'] = 0
    fields[criterion + '_mismatch_evidence'] = evidence
    return ScoreResult(**fields)


def second_ai_review(criterion, evidence):
    checks = {f: ('supports_rejection' if f == criterion else 'no_relevant_information')
              for f in ('location', 'business', 'goal')}
    return parse_semantic_review(json.dumps({
        'schema_version': 3, 'verdict': 'reject', 'criterion': criterion,
        'spam_type': 'none', 'field_checks': checks,
        'no_relevance_basis': {f: ('not_applicable' if f == criterion else 'explicit_neutral')
                               for f in checks},
        'field_evidence': {
            f: {'supporting_quotes': [evidence] if f == criterion else [],
                'contradicting_quotes': [], 'uncertainty_reason': ''} for f in checks
        },
        'overall_consistency': 'consistent', 'eligibility_signal': 'absent',
        'explanation_code': f'explicit_{criterion}_mismatch',
        'evidence_quote': evidence,
    }, ensure_ascii=False))


@pytest.mark.parametrize('criterion,negative,qualifying', [
    ('location', LOCATION_NEGATIVE, 'Уже в Тбилиси'),
    ('location', LOCATION_NEGATIVE, 'Уже обосновался в Тбилиси'),
    ('location', LOCATION_NEGATIVE, 'Только что прилетел в Батуми'),
    ('business', BUSINESS_NEGATIVE, 'Я фотограф, снимаю свадьбы по заказу'),
    ('business', BUSINESS_NEGATIVE, 'У меня небольшое кафе, обслуживаю клиентов'),
    ('business', BUSINESS_NEGATIVE, 'Занимаюсь репетиторством и даю частные уроки'),
    ('location', LOCATION_NEGATIVE, 'Между тем у меня забронирован отель'),
    ('business', BUSINESS_NEGATIVE, 'Выполняю заказы клиентов'),
])
@pytest.mark.parametrize('separator', ['. ', '; ', '\n', ' — ', '. И ещё: '])
def test_both_ai_stages_cannot_hide_conflict_inside_primary_answer(
    criterion, negative, qualifying, separator,
):
    full = negative + separator + qualifying
    answers = (full, NEUTRAL, NEUTRAL) if criterion == 'location' else (
        NEUTRAL, full, NEUTRAL,
    )
    s = score(criterion, negative)
    proposal = propose_auto_rejection(s, candidate_answers=answers, **PARAMS)
    # Proposal must be absent even when model provides a true substring.
    assert proposal is None
    bad_review = second_ai_review(criterion, negative)
    # The stage-3 contract is also independently fail-closed.
    assert review_contract_failure(bad_review, answers) == 'ambiguous_criterion_answer'
    final = decide_auto_rejection(proposal=proposal, review=bad_review,
                                  score=s, candidate_answers=answers, **PARAMS)
    assert not final.allow_auto_reject
    assert final.reason is None


@pytest.mark.parametrize('criterion,full_answer', [
    ('location', LOCATION_NEGATIVE),
    ('location', 'Я не в Грузии и не планирую приезжать'),
    ('location', 'Я не в Грузии, приезжать не собираюсь, пока и не хочу'),
    ('business', BUSINESS_NEGATIVE),
    ('business', 'Нет бизнеса, нет проектов, нигде не работаю'),
])
def test_unambiguous_whole_core_response_is_still_auto_rejectable(criterion, full_answer):
    answers = (full_answer, NEUTRAL, NEUTRAL) if criterion == 'location' else (
        NEUTRAL, full_answer, NEUTRAL,
    )
    s = score(criterion, full_answer)
    proposal = propose_auto_rejection(s, candidate_answers=answers, **PARAMS)
    assert proposal is not None
    review = second_ai_review(criterion, full_answer)
    assert review_contract_failure(review, answers) is None
    final = decide_auto_rejection(proposal=proposal, review=review, score=s,
                                  candidate_answers=answers, **PARAMS)
    assert final.allow_auto_reject and final.code == 'confirmed'
    assert full_answer in final.reason


@pytest.mark.parametrize('criterion,source', [
    ('location', LOCATION_NEGATIVE), ('business', BUSINESS_NEGATIVE)
])
def test_second_ai_cannot_use_excerpt_even_if_primary_answer_is_clear(criterion, source):
    excerpt = 'не в Грузии' if criterion == 'location' else 'нет бизнеса'
    answers = (source, NEUTRAL, NEUTRAL) if criterion == 'location' else (
        NEUTRAL, source, NEUTRAL,
    )
    assert review_contract_failure(second_ai_review(criterion, excerpt), answers) \
        == 'unverifiable_field_quote'


def test_unknown_novel_negative_phrase_is_reviewed_not_guessed():
    phrase = 'Нахожусь вне Грузии и никаких поездок туда не планирую'
    s = score('location', phrase)
    assert propose_auto_rejection(s, candidate_answers=(phrase, NEUTRAL, NEUTRAL),
                                  **PARAMS) is None


# A punctuation-stripping word comparison alone is not evidence that a
# candidate made an unambiguous declarative statement. Both AI passes may
# return an otherwise valid schema for an interrogative / quoted answer.
@pytest.mark.parametrize('criterion,answer', [
    ('location', LOCATION_NEGATIVE + '?'),
    ('business', BUSINESS_NEGATIVE + '?'),
    ('location', LOCATION_NEGATIVE + '？'),  # full-width question mark
    ('business', BUSINESS_NEGATIVE + '⁇'),  # Unicode double question mark
    ('location', LOCATION_NEGATIVE + '?!'),
    ('business', BUSINESS_NEGATIVE + '؟'),  # Arabic question mark
    ('location', 'Я не в Грузии? И не планирую приезжать'),
    ('business', 'Нет бизнеса? Нет проектов, нигде не работаю'),
    ('location', LOCATION_NEGATIVE + '…'),
    ('business', BUSINESS_NEGATIVE + '...'),
    ('location', '«' + LOCATION_NEGATIVE + '»'),
    ('business', '"' + BUSINESS_NEGATIVE + '"'),
    ('location', 'Я не в Грузии. Не планирую приезжать'),
    ('business', 'Нет бизнеса; нет проектов, нигде не работаю'),
    ('location', LOCATION_NEGATIVE + '!'),
])
@pytest.mark.parametrize('evidence_with_punctuation', [False, True])
def test_core_interrogative_or_ambiguous_punctuation_never_auto_rejects(
    criterion, answer, evidence_with_punctuation,
):
    # The model might quote the words only OR include the question/quote marks.
    evidence = answer if evidence_with_punctuation else (
        LOCATION_NEGATIVE if criterion == 'location'
        else BUSINESS_NEGATIVE
    )
    if criterion == 'location' and answer == 'Я не в Грузии? И не планирую приезжать':
        evidence = answer
    if criterion == 'business' and answer == 'Нет бизнеса? Нет проектов, нигде не работаю':
        evidence = answer
    if criterion == 'location' and answer == 'Я не в Грузии. Не планирую приезжать':
        evidence = answer
    if criterion == 'business' and answer == 'Нет бизнеса; нет проектов, нигде не работаю':
        evidence = answer
    answers = (answer, NEUTRAL, NEUTRAL) if criterion == 'location' else (
        NEUTRAL, answer, NEUTRAL,
    )
    s = score(criterion, evidence)
    proposal = propose_auto_rejection(s, candidate_answers=answers, **PARAMS)
    assert proposal is None
    review = second_ai_review(criterion, evidence)
    # The independent third stage must also fail closed if the earlier
    # classifier was buggy or an attacker forged its decision.
    assert review_contract_failure(review, answers) is not None
    # Force the old model's invalid proposal through the final gate too.
    # This must not authorize a decline even when both models agree.
    forged_proposal = RejectionProposal(
        criterion=criterion, evidence_quote=evidence, spam_type=None,
        reason='AI suggested core mismatch',
    )
    decision = decide_auto_rejection(
        proposal=forged_proposal, review=review, score=s,
        candidate_answers=answers, **PARAMS,
    )
    assert not decision.allow_auto_reject
    assert decision.reason is None


@pytest.mark.parametrize('criterion,negative', [
    ('location', LOCATION_NEGATIVE), ('business', BUSINESS_NEGATIVE),
])
def test_simple_final_period_is_still_an_unambiguous_core_answer(criterion, negative):
    # A single sentence-final full stop is not a question or qualification.
    answer = negative + '.'
    answers = (answer, NEUTRAL, NEUTRAL) if criterion == 'location' else (
        NEUTRAL, answer, NEUTRAL,
    )
    s = score(criterion, negative)
    proposal = propose_auto_rejection(s, candidate_answers=answers, **PARAMS)
    assert proposal is not None
    review = second_ai_review(criterion, negative)
    assert review_contract_failure(review, answers) is None
    decision = decide_auto_rejection(
        proposal=proposal, review=review, score=s,
        candidate_answers=answers, **PARAMS,
    )
    assert decision.allow_auto_reject
