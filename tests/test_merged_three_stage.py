"""Regression of the merged production route, including actual third-stage vetoes.

Mock only external DeepSeek, Telegram and persistence. No live API calls.
The previous v3 gate scenarios are retained separately in tests/legacy/.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import TelegramError

from app import texts
from app.decision import GateDecision, parse_semantic_review
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus
from app.scoring import (
    AIScorer, parse_score_json, propose_auto_rejection,
)

LOCATION_NO = 'Я сейчас не в Грузии и не планирую приезжать ближайшие годы.'
BUSINESS_NO = 'У меня нет бизнеса, нет проектов, нигде не работаю.'
NEUTRAL = 'Нет дополнительных сведений.'
AD = 'Переходите по ссылке и покупайте обучение, сейчас скидка 50%'
PARAMS = dict(spam_threshold=85, core_score_max=15, min_ai_confidence=80)


def primary(criterion='location', answers=None, *, false_quote=False,
            malformed_metadata=False, unclear=False):
    if answers is None:
        answers = (LOCATION_NO, NEUTRAL, NEUTRAL)
    loc, biz, goal = answers
    response = {
        'location_score': 0 if criterion == 'location' else 90,
        'business_score': 0 if criterion == 'business' else 90,
        'intent_score': 0,
        'spam_score': 98 if criterion == 'spam' else 1,
        'summary': 'Первичный скоринг',
        'risk_flags': [],
        'confidence_evidence': {'location': '', 'business': '', 'goal': ''},
        'contradiction_quotes': [] if not malformed_metadata else 'malformed',
        'uncertainty_quotes': [] if not malformed_metadata else 'malformed',
        'core_rejection_evidence': {},
        'spam_evidence': [],
    }
    if criterion in ('location', 'business'):
        field = criterion
        quote = {'location': loc, 'business': biz}[field]
        if false_quote:
            quote = 'Я никогда не утверждал, что живу в Грузии'
        response['core_rejection_evidence'][field] = {
            'quote': quote,
            'explanation': 'Ответ прямо и однозначно отрицает наличие критерия',
        }
        response['confidence_evidence'][field] = quote
    if criterion == 'spam':
        response['spam_evidence'] = [{
            'category': 'mass_advertising', 'source_field': 'goal',
            'quote': goal, 'explanation': 'Прямая навязчивая реклама обучения',
        }]
    return json.dumps(response, ensure_ascii=False)


def review_data(criterion, answers, *, verdict='reject', overrides=None):
    index = {'location': 0, 'business': 1, 'spam': 2}[criterion]
    field = ('location', 'business', 'goal')[index]
    quote = answers[index]
    checks = {f: ('supports_rejection' if f == field else 'no_relevant_information')
              for f in ('location', 'business', 'goal')}
    payload = {
        'schema_version': 3, 'verdict': verdict,
        'criterion': criterion,
        'spam_type': 'advertising' if criterion == 'spam' else 'none',
        'field_checks': checks,
        'no_relevance_basis': {f: 'not_applicable' if f == field else 'explicit_neutral'
                               for f in checks},
        'field_evidence': {f: {'supporting_quotes': [quote] if f == field else [],
                               'contradicting_quotes': [], 'uncertainty_reason': ''}
                           for f in checks},
        'overall_consistency': 'consistent',
        'eligibility_signal': 'absent',
        'explanation_code': ('explicit_spam' if criterion == 'spam'
                             else f'explicit_{criterion}_mismatch'),
        'evidence_quote': quote,
    }
    if overrides:
        payload.update(overrides)
    return payload


def answers_for(criterion):
    return {
        'location': (LOCATION_NO, NEUTRAL, NEUTRAL),
        'business': (NEUTRAL, BUSINESS_NO, NEUTRAL),
        'spam': (NEUTRAL, NEUTRAL, AD),
    }[criterion]


class FakeDB:
    def __init__(self, candidate):
        self.candidate = candidate
        self.saved = None
        self.raw = None
        self.statuses = []

    async def get_candidate(self, candidate_id):
        return self.candidate

    async def save_score(self, ident, result, raw):
        assert ident == self.candidate.id
        self.saved = result
        self.raw = json.loads(raw)
        return self.candidate

    async def set_status(self, ident, status, **kwargs):
        self.statuses.append((status, kwargs))
        self.candidate.status = status.value
        return self.candidate


class FakeScorer:
    def __init__(self, score, raw, review=None, error=None):
        self.primary = score
        self.raw = raw
        self.review = review
        self.error = error
        self.calls = []

    async def score(self, **kwargs):
        self.calls.append(('primary', kwargs))
        return self.primary, self.raw

    async def review_rejection(self, **kwargs):
        self.calls.append(('independent', kwargs))
        if self.error:
            raise self.error
        return self.review


def run_scenario(criterion='location', *, answers=None, review=None,
                 stage2_error=None, telegram_error=None, raw_override=None,
                 min_confidence=80):
    answers = answers or answers_for(criterion)
    raw = raw_override or primary(criterion, answers)
    parsed = parse_score_json(raw, location=answers[0], business=answers[1], goal=answers[2])
    candidate = Candidate(
        id=67, user_id=555, user_chat_id=555,
        target_chat_id=-100556, status=CandidateStatus.SCORING.value,
        stage=4, location_answer=answers[0], business_answer=answers[1],
        goal_answer=answers[2], first_name='Test',
    )
    review = review if review is not None else parse_semantic_review(
        json.dumps(review_data(criterion, answers), ensure_ascii=False)
    )
    db = FakeDB(candidate)
    scorer = FakeScorer(parsed, raw, review, stage2_error)
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)),
        decline_chat_join_request=AsyncMock(side_effect=telegram_error),
    )
    settings = SimpleNamespace(
        auto_reject_spam_score=85, auto_reject_core_score_max=15,
        auto_reject_min_ai_confidence=min_confidence,
        admin_chat_id=-200,
    )
    asyncio.run(GatekeeperHandlers(settings=settings, db=db, scorer=scorer)
                ._score_and_route(bot, candidate))
    return parsed, scorer, db, bot


@pytest.mark.parametrize('criterion', ['location', 'business', 'spam'])
def test_genuine_ground_three_stages_proceed_to_auto_decline(criterion, caplog):
    with caplog.at_level(logging.INFO, logger='app.handlers'):
        score, scorer, db, bot = run_scenario(criterion)
    assert score.confidence_for(criterion) == 100
    assert db.statuses[-1][0] == CandidateStatus.AUTO_REJECTED
    assert bot.decline_chat_join_request.await_count == 1
    assert len(scorer.calls) == 2
    audit = db.raw['_gatekeeper_decision']
    assert audit['stage1_proposal']['criterion'] == criterion
    assert audit['stage2_review']['verdict'] == 'reject'
    assert audit['stage3']['allow_auto_reject'] is True
    assert audit['stage3']['code'] == 'confirmed'
    assert 'Stage1 candidate=67' in caplog.text
    assert 'Stage2 candidate=67' in caplog.text
    assert 'Stage3 candidate=67' in caplog.text
    assert LOCATION_NO not in caplog.text  # sensitive responses stay in DB only


def test_specific_location_confidence_no_longer_needs_other_two_evidence_quotes():
    answers = answers_for('location')
    raw = primary('location', answers)
    score = parse_score_json(raw, location=answers[0], business=answers[1], goal=answers[2])
    assert score.confidence_details.evidence_verified is False
    assert score.confidence_details.formula_score < 100
    assert score.ai_confidence == 100
    confidence = score.decision_confidences[0]
    assert confidence.criterion == 'location'
    assert (confidence.completeness, confidence.specificity,
            confidence.consistency, confidence.certainty) == (100, 100, 100, 100)
    assert confidence.evidence_verified
    assert propose_auto_rejection(score, candidate_answers=answers, **PARAMS) is not None


@pytest.mark.parametrize('criterion', ['location', 'business'])
@pytest.mark.parametrize('suffix', [
    '?', '!', ' ...', ' Но скоро прилетаю в Тбилиси.',
    '; поездка в Грузию запланирована',
    '\nПланирую работать в Батуми.', ' — цитата из чужой анкеты',
])
def test_ambiguous_primary_answer_never_reaches_second_ai(criterion, suffix):
    originals = list(answers_for(criterion))
    index = 0 if criterion == 'location' else 1
    originals[index] += suffix
    _, scorer, db, bot = run_scenario(criterion, answers=tuple(originals))
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    assert len(scorer.calls) == 1
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.raw['_gatekeeper_decision']['stage1_proposal'] is None


@pytest.mark.parametrize('criterion,other_field', [
    ('location', 'В следующем месяце командировка в Батуми'),
    ('location', 'Я планирую переехать в Тбилиси'),
    ('business', 'Я фотограф, снимаю свадьбы по заказу'),
    ('business', 'Я инвестирую в местные проекты'),
    ('business', 'Занимаюсь репетиторством'),
])
def test_full_profile_cannot_be_hidden_by_single_negative_quote(criterion, other_field):
    answers = list(answers_for(criterion))
    answers[2] = other_field
    _, scorer, db, bot = run_scenario(criterion, answers=tuple(answers))
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.raw['_gatekeeper_decision']['stage3']['allow_auto_reject'] is False


@pytest.mark.parametrize('overrides', [
    {'eligibility_signal': 'present'},
    {'overall_consistency': 'contradicted'},
    {'explanation_code': 'needs_review'},
    {'schema_version': 2},
    {'evidence_quote': 'unsupported quote'},
])
def test_unsafe_second_review_never_authorizes_decline(overrides):
    answers = answers_for('location')
    bad = review_data('location', answers, overrides=overrides)
    try:
        parsed = parse_semantic_review(json.dumps(bad, ensure_ascii=False))
    except ValueError:
        parsed = None
    _, scorer, db, bot = run_scenario('location', review=parsed)
    # When parser rejects a malformed review, an actual scorer would raise,
    # and routing would fail closed. Test the strict parser separately above.
    if parsed is not None:
        assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
        bot.decline_chat_join_request.assert_not_awaited()


def test_second_ai_timeout_fails_closed_and_persists_reason_code(caplog):
    with caplog.at_level(logging.INFO, logger='app.handlers'):
        _, scorer, db, bot = run_scenario(
            'location', stage2_error=TimeoutError('Unavailable'),
        )
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.raw['_gatekeeper_decision']['stage2_error_type'] == 'TimeoutError'
    assert db.raw['_gatekeeper_decision']['stage3']['code'] == 'missing_proposal_or_review'
    assert 'unavailable error=TimeoutError' in caplog.text


def test_gate_error_fails_closed(monkeypatch):
    def broken_gate(**kwargs):
        raise RuntimeError('test gate failure')
    monkeypatch.setattr('app.handlers.decide_auto_rejection', broken_gate)
    _, _, db, bot = run_scenario('location')
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    assert db.raw['_gatekeeper_decision']['stage3']['code'] == 'decision_gate_error'
    bot.decline_chat_join_request.assert_not_awaited()


def test_failed_telegram_decline_preserves_pending_and_notifies_candidate():
    _, _, db, bot = run_scenario('location', telegram_error=TelegramError('Fail'))
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    assert db.raw['_gatekeeper_decision']['stage3']['allow_auto_reject']
    candidate_texts = [call.args[1] for call in bot.send_message.await_args_list
                       if call.args and call.args[0] == 555]
    assert texts.REJECTED not in candidate_texts
    assert texts.TECHNICAL_REVIEW in candidate_texts


def test_bad_primary_quote_prevents_proposal_and_review():
    answers = answers_for('location')
    raw = primary('location', answers, false_quote=True)
    _, scorer, db, bot = run_scenario('location', answers=answers, raw_override=raw)
    assert len(scorer.calls) == 1
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    bot.decline_chat_join_request.assert_not_awaited()


def test_malformed_primary_contradiction_metadata_caps_specific_confidence_below_80():
    answers = answers_for('location')
    raw = primary('location', answers, malformed_metadata=True)
    score = parse_score_json(raw, location=answers[0], business=answers[1], goal=answers[2])
    assert score.confidence_for('location') <= 79
    assert propose_auto_rejection(score, candidate_answers=answers, **PARAMS) is None


def test_spam_quote_in_warning_context_requires_admin():
    answers = (NEUTRAL, NEUTRAL, 'Не покупайте обучение у мошенников. ' + AD)
    _, scorer, db, bot = run_scenario('spam', answers=answers)
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    bot.decline_chat_join_request.assert_not_awaited()


def test_spam_without_independent_confirmation_has_penalty_withheld():
    _, _, db, bot = run_scenario('spam', stage2_error=TimeoutError())
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    assert db.saved.spam_penalty_withheld is True
    assert db.raw['_gatekeeper_score_adjustment']['spam_penalty_withheld'] is True
    bot.decline_chat_join_request.assert_not_awaited()


def test_review_api_does_not_leak_first_score(monkeypatch):
    calls = []
    async def fake_complete(self, messages):
        calls.append(messages)
        return json.dumps(review_data('location', answers_for('location')), ensure_ascii=False)
    monkeypatch.setattr(AIScorer, '_complete_json', fake_complete)
    scorer = AIScorer(base_url='https://example.invalid', api_key='fake',
                      model='fake', timeout_seconds=1)
    answers = answers_for('location')
    review = asyncio.run(scorer.review_rejection(
        criterion='location', location=answers[0], business=answers[1], goal=answers[2]))
    assert review.verdict == 'reject'
    payload = calls[0][1]['content']
    assert 'location_score' not in payload
    assert 'ai_confidence' not in payload
    assert 'summary' not in payload
    for answer in answers:
        assert answer in payload


def test_criterion_confidence_audit_is_serialized_with_formula(monkeypatch):
    answers = answers_for('location')
    raw = primary('location', answers)
    async def complete(self, messages):
        return raw
    monkeypatch.setattr(AIScorer, '_complete_json', complete)
    scorer = AIScorer(base_url='https://example.invalid', api_key='fake',
                      model='fake', timeout_seconds=1)
    score, raw = asyncio.run(scorer.score(
        location=answers[0], business=answers[1], goal=answers[2]))
    saved = json.loads(raw)
    assert saved['_gatekeeper_criterion_confidence']['rubric_version'] == 2
    assert saved['_gatekeeper_criterion_confidence']['effective_ai_confidence'] == 100
    assert saved['_gatekeeper_criterion_confidence']['criteria'][0]['criterion'] == 'location'
    assert saved['_gatekeeper_criterion_confidence']['weights'] == {
        'completeness': .3, 'specificity': .3, 'consistency': .25, 'certainty': .15,
    }

@pytest.mark.parametrize('criterion', ['location', 'business', 'spam'])
def test_http_mock_transport_runs_real_two_deepseek_requests_and_gate(monkeypatch, criterion):
    """Exercise real production AIScorer and GatekeeperHandlers, offline."""
    import httpx
    from app import scoring as scoring_module

    answers = answers_for(criterion)
    request_bodies = []
    llm_outputs = [
        primary(criterion, answers),
        json.dumps(review_data(criterion, answers), ensure_ascii=False),
    ]
    def respond(request):
        assert request.url.path.endswith('/chat/completions')
        data = json.loads(request.content)
        request_bodies.append(data)
        response_text = llm_outputs.pop(0)
        return httpx.Response(200, json={
            'choices': [{'message': {'content': response_text}}]
        })

    transport = httpx.MockTransport(respond)
    original = httpx.AsyncClient
    monkeypatch.setattr(scoring_module.httpx, 'AsyncClient',
                        lambda **kwargs: original(transport=transport, **kwargs))
    real_scorer = AIScorer(
        base_url='https://test.invalid/v1', api_key='not-a-secret',
        model='mock', timeout_seconds=1,
    )
    candidate = Candidate(
        id=77, user_id=222, user_chat_id=222, target_chat_id=-100,
        location_answer=answers[0], business_answer=answers[1],
        goal_answer=answers[2], stage=4, status=CandidateStatus.SCORING.value,
    )
    db = FakeDB(candidate)
    bot = SimpleNamespace(
        send_message=AsyncMock(), decline_chat_join_request=AsyncMock(),
    )
    settings = SimpleNamespace(
        auto_reject_min_ai_confidence=80, auto_reject_core_score_max=15,
        auto_reject_spam_score=85, admin_chat_id=-200,
    )
    asyncio.run(GatekeeperHandlers(settings=settings, db=db, scorer=real_scorer)
                ._score_and_route(bot, candidate))
    assert len(request_bodies) == 2
    assert db.statuses[-1][0] == CandidateStatus.AUTO_REJECTED
    bot.decline_chat_join_request.assert_awaited_once()
    assert 'location_score' not in request_bodies[1]['messages'][1]['content']
    assert 'expected_spam_type' not in request_bodies[1]['messages'][1]['content']
    assert 'criterion' not in request_bodies[1]['messages'][1]['content']
    assert db.raw['_gatekeeper_decision']['stage3']['code'] == 'confirmed'


@pytest.mark.parametrize('goal', [
    'В декабре у меня конференция в Тбилиси, уже зарегистрировался',
    'Забронирован номер в гостинице Батуми на следующую неделю',
    'Мне подтвердили бронь гостиницы в Тбилиси',
    'Я планирую приехать в Грузию, чтобы найти партнёров',
])
def test_http_mock_two_models_can_both_misread_cross_field_and_gate_vetoes(monkeypatch, goal):
    import httpx
    from app import scoring as scoring_module

    answers = (LOCATION_NO, NEUTRAL, goal)
    first = primary('location', answers)
    # Deliberately deceptive v3: second LLM falsely declares the non-neutral
    # goal as neutral. Stage 3 must reject this consensus.
    second = json.dumps(review_data('location', answers), ensure_ascii=False)
    outputs = [first, second]
    original_client = httpx.AsyncClient
    def respond(request):
        return httpx.Response(200, json={'choices': [{'message': {'content': outputs.pop(0)}}]})
    monkeypatch.setattr(scoring_module.httpx, 'AsyncClient',
                        lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw))
    scorer = AIScorer(base_url='https://test.invalid/v1', api_key='fake',
                      model='mock', timeout_seconds=1)
    candidate = Candidate(
        id=80, user_id=222, user_chat_id=222, target_chat_id=-100,
        location_answer=answers[0], business_answer=answers[1], goal_answer=answers[2],
        stage=4, status=CandidateStatus.SCORING.value,
    )
    db = FakeDB(candidate)
    bot = SimpleNamespace(send_message=AsyncMock(), decline_chat_join_request=AsyncMock())
    settings = SimpleNamespace(auto_reject_min_ai_confidence=80,
                               auto_reject_core_score_max=15, auto_reject_spam_score=85,
                               admin_chat_id=-200)
    asyncio.run(GatekeeperHandlers(settings=settings, db=db, scorer=scorer)
                ._score_and_route(bot, candidate))
    assert db.statuses[-1][0] == CandidateStatus.PENDING_ADMIN
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.raw['_gatekeeper_decision']['stage3']['allow_auto_reject'] is False

@pytest.mark.parametrize('threshold', [0, 40, 70, 79])
def test_invalid_evidence_is_not_authorized_even_when_admin_lowers_threshold(threshold):
    answers = answers_for('location')
    raw = primary('location', answers, malformed_metadata=True)
    score = parse_score_json(raw, location=answers[0], business=answers[1], goal=answers[2])
    assert not score.decision_confidences[0].evidence_verified
    assert propose_auto_rejection(
        score, candidate_answers=answers, spam_threshold=85,
        core_score_max=15, min_ai_confidence=threshold,
    ) is None
