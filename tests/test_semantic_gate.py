"""Safe second-pass rejection: the first model must never decline alone."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.decision import parse_semantic_review
from app.config import Settings
from app.handlers import GatekeeperHandlers
from app.models import Candidate, CandidateStatus
from app.scoring import (
    AIScorer,
    ScoreResult,
    auto_reject_reason,
    semantic_review_confirms_rejection,
    proposed_rejection_criterion,
    SEMANTIC_REVIEW_PROMPT,
)

LOCATION_NO = "Я сейчас не в Грузии и не планирую приезжать ближайшие годы"
BUSINESS_NO = "У меня нет бизнеса, нет проектов, нигде не работаю"
SPAM = "Переходите по ссылке и покупайте обучение, сейчас скидка 50%"
REJECT_SETTINGS = dict(spam_threshold=85, core_score_max=15, min_ai_confidence=80)


def score(**kw):
    data = dict(
        location_score=90, business_score=90, intent_score=70, spam_score=0,
        ai_confidence=99, confidence_score=70, summary="review", risk_flags=[],
    )
    data.update(kw)
    return ScoreResult(**data)


def review(criterion, evidence, *, verdict="reject", checks=None, spam_type="none"):
    checks = checks or {"location": "no_relevant_information",
                         "business": "no_relevant_information",
                         "goal": "no_relevant_information"}
    evidence_map = {
        field: {"supporting_quotes": [evidence] if status == "supports_rejection" else [],
                "contradicting_quotes": [], "uncertainty_reason": ""}
        for field, status in checks.items() if field in ("location", "business", "goal")
    }
    return json.dumps({
        "schema_version": 3, "verdict": verdict, "criterion": criterion, "spam_type": spam_type,
        "field_checks": checks, "field_evidence": evidence_map,
        "no_relevance_basis": {f: "explicit_neutral" if status == "no_relevant_information"
                               else "not_applicable" for f, status in checks.items()},
        "overall_consistency": "consistent", "eligibility_signal": "absent",
        "explanation_code": {"location": "explicit_location_mismatch",
                             "business": "explicit_business_mismatch",
                             "spam": "explicit_spam"}[criterion],
        "evidence_quote": evidence,
    }, ensure_ascii=False)


def loc_checks(**kw):
    return {"location": "supports_rejection", "business": "no_relevant_information",
            "goal": "no_relevant_information", **kw}


def biz_checks(**kw):
    return {"location": "no_relevant_information", "business": "supports_rejection",
            "goal": "no_relevant_information", **kw}


def test_proposal_is_not_final_verdict():
    reason = auto_reject_reason(
        score(location_score=0, location_mismatch_evidence=LOCATION_NO),
        **REJECT_SETTINGS, candidate_answers=(LOCATION_NO, "", ""),
    )
    assert reason is not None
    assert proposed_rejection_criterion(reason) == "location"
    assert proposed_rejection_criterion("неизвестная причина") is None


@pytest.mark.parametrize("criterion, quote, answers, checks", [
    ("location", LOCATION_NO, (LOCATION_NO, "Не знаю", "Нет дополнительных сведений"), loc_checks()),
    ("business", BUSINESS_NO, ("Не знаю", BUSINESS_NO, "Нет дополнительных сведений"), biz_checks()),
    ("spam", SPAM, ("Не знаю", "Нет дополнительных сведений", SPAM), {"location": "no_relevant_information",
             "business": "no_relevant_information", "goal": "supports_rejection"}),
])
def test_structurally_valid_second_review_can_confirm(criterion, quote, answers, checks):
    assert semantic_review_confirms_rejection(
        review(criterion, quote, checks=checks,
               spam_type="advertising" if criterion == "spam" else "none"),
        criterion=criterion, candidate_answers=answers,
        expected_spam_type="advertising" if criterion == "spam" else None,
    )


@pytest.mark.parametrize("status", ["contradicts", "uncertain"])
@pytest.mark.parametrize("field", ["location", "business", "goal"])
def test_review_cannot_reject_with_ambiguity_in_any_field(status, field):
    checks = loc_checks(**{field: status})
    assert not semantic_review_confirms_rejection(
        review("location", LOCATION_NO, checks=checks),
        criterion="location", candidate_answers=(LOCATION_NO, "", ""),
    )


@pytest.mark.parametrize("bad", [
    '{"verdict":"reject"}',
    "garbage not JSON",
    json.dumps({"verdict": "reject", "criterion": "location", "evidence_quote": LOCATION_NO,
                "field_checks": {"location": "supports_rejection", "business": "no_relevant_information"}}),
    review("business", LOCATION_NO, checks=biz_checks()),
    review("location", "Я нахожусь в Грузии и приезжать планирую", checks=loc_checks()),
    review("location", "я сейчас не в грузии... не планирую приезжать", checks=loc_checks()),
    review("location", LOCATION_NO, checks=loc_checks(goal=["no_relevant_information"])),
    review("location", LOCATION_NO, checks=loc_checks(foo="neutral")),
])
def test_malformed_or_unverifiable_second_review_never_rejects(bad):
    assert not semantic_review_confirms_rejection(
        bad, criterion="location", candidate_answers=(LOCATION_NO, "", "")
    )


def test_wrong_verdict_or_score_does_not_authorize_decline():
    for criterion, verdict in (("business", "reject"), ("location", "review"),
                               ("spam", "reject")):
        assert not semantic_review_confirms_rejection(
            review(criterion, LOCATION_NO, verdict=verdict, checks=loc_checks()),
            criterion="location", candidate_answers=(LOCATION_NO, "", ""),
        )


@pytest.mark.parametrize("goal", [
    "В декабре у меня конференция в Тбилиси, уже зарегистрировался",
    "Уже забронировал билет на самолёт до Тбилиси",
    "В следующем месяце командировка в Батуми",
])
def test_cross_field_travel_requires_semantic_abstention(goal):
    # The old keyword gate misses these. The second pass sees whole answers.
    answers = (LOCATION_NO, "", goal)
    assert auto_reject_reason(
        score(location_score=0, location_mismatch_evidence=LOCATION_NO),
        **REJECT_SETTINGS, candidate_answers=answers,
    )
    assert not semantic_review_confirms_rejection(
        review("location", LOCATION_NO,
               checks=loc_checks(goal="contradicts")),
        criterion="location", candidate_answers=answers,
    )


@pytest.mark.parametrize("goal", [
    "Я фотограф, снимаю свадьбы по заказу",
    "Занимаюсь репетиторством и даю частные уроки",
    "У меня небольшое кафе, обслуживаю клиентов",
])
def test_cross_field_professional_activity_requires_semantic_abstention(goal):
    answers = ("", BUSINESS_NO, goal)
    assert auto_reject_reason(
        score(business_score=0, business_mismatch_evidence=BUSINESS_NO),
        **REJECT_SETTINGS, candidate_answers=answers,
    )
    assert not semantic_review_confirms_rejection(
        review("business", BUSINESS_NO, checks=biz_checks(goal="contradicts")),
        criterion="business", candidate_answers=answers,
    )


@pytest.mark.parametrize("criterion, answers, raw", [
    ("location", (LOCATION_NO, "Не знаю", "Нет дополнительных сведений"), review("location", LOCATION_NO, checks=loc_checks())),
    ("business", ("Не знаю", BUSINESS_NO, "Нет дополнительных сведений"), review("business", BUSINESS_NO, checks=biz_checks())),
])
def test_actual_scorer_second_request_sends_every_answer(criterion, answers, raw):
    scorer = AIScorer(base_url="https://llm.invalid/v1", api_key="test", model="x", timeout_seconds=1)
    scorer._request_json = AsyncMock(return_value=raw)
    assert asyncio.run(scorer.verify_rejection(
        criterion=criterion, location=answers[0], business=answers[1], goal=answers[2]
    ))
    messages = scorer._request_json.await_args.args[0]
    assert messages[0]["content"] == SEMANTIC_REVIEW_PROMPT
    for answer in answers:
        assert answer in messages[1]["content"]
    assert '"criterion": "' + criterion + '"' in messages[1]["content"]


class Scorer:
    def __init__(self, result, reviewer=True, failure=False):
        self.result = result
        self.review_result = reviewer
        self.failure = failure
        self.review_calls = []

    async def score(self, **kwargs):
        return self.result, "{}"

    async def review_rejection(self, **kwargs):
        self.review_calls.append(kwargs)
        if self.failure:
            raise TimeoutError("verification timeout")
        criterion = kwargs["criterion"]
        evidence = kwargs["location"] if criterion == "location" else kwargs["business"]
        checks = loc_checks() if criterion == "location" else biz_checks()
        return parse_semantic_review(review(
            criterion, evidence,
            verdict="reject" if self.review_result else "review", checks=checks,
        ))


class DB:
    def __init__(self, candidate):
        self.candidate = candidate
        self.states = []

    async def save_score(self, *args):
        return self.candidate

    async def set_status(self, _id, status, **kwargs):
        self.candidate.status = status.value
        self.states.append(status)
        return self.candidate


def fake_settings():
    return Settings(
        bot_token="test", target_chat_id=-1001, admin_chat_id=-2002,
        admin_user_ids={42}, database_url="sqlite+aiosqlite:///:memory:",
        llm_base_url="https://example.invalid/v1", llm_api_key="test", llm_model="test",
        llm_timeout_seconds=1, auto_reject_spam_score=85,
        auto_reject_core_score_max=15, auto_reject_min_ai_confidence=80,
        max_answer_length=2500,
    )


def fake_candidate(*, location=LOCATION_NO, business="Сведения отсутствуют", goal="Хочу вступить в сообщество"):
    return Candidate(
        id=1, user_id=111, user_chat_id=777, target_chat_id=-1001,
        username="sample", first_name="Sample", status=CandidateStatus.SCORING.value,
        stage=4, location_answer=location, business_answer=business, goal_answer=goal,
    )


def route(candidate, scorer):
    db = DB(candidate)
    bot = SimpleNamespace(decline_chat_join_request=AsyncMock())
    handler = GatekeeperHandlers(settings=fake_settings(), db=db, scorer=scorer)
    handler._send_admin_card = AsyncMock()
    handler._send_candidate_message = AsyncMock()
    asyncio.run(handler._score_and_route(bot, candidate))
    return db, bot, handler


@pytest.mark.parametrize("goal", [
    "В декабре у меня конференция в Тбилиси, уже зарегистрировался",
    "Уже забронировал билет на самолёт до Тбилиси",
    "В следующем месяце командировка в Батуми",
])
def test_handler_cross_field_travel_never_declines_when_review_disagrees(goal):
    item = fake_candidate(goal=goal)
    scorer = Scorer(score(location_score=0, location_mismatch_evidence=LOCATION_NO), False)
    db, bot, handler = route(item, scorer)
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN
    assert scorer.review_calls[0]["goal"] == goal
    assert handler._send_admin_card.await_args.kwargs["status_note"]


@pytest.mark.parametrize("goal", [
    "Я фотограф, снимаю свадьбы по заказу",
    "Занимаюсь репетиторством и даю частные уроки",
    "У меня небольшое кафе, обслуживаю клиентов",
])
def test_handler_cross_field_work_never_declines_when_review_disagrees(goal):
    item = fake_candidate(location="Тбилиси", business=BUSINESS_NO, goal=goal)
    scorer = Scorer(score(business_score=0, business_mismatch_evidence=BUSINESS_NO), False)
    db, bot, _ = route(item, scorer)
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN


@pytest.mark.parametrize("reviewer,failure", [(False, False), (True, True)])
def test_handler_abstains_on_negative_or_failed_second_review(reviewer, failure):
    item = fake_candidate()
    scorer = Scorer(score(location_score=0, location_mismatch_evidence=LOCATION_NO),
                    reviewer, failure)
    db, bot, _ = route(item, scorer)
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN


def test_handler_can_auto_decline_with_second_confirmation():
    item = fake_candidate()
    scorer = Scorer(score(location_score=0, location_mismatch_evidence=LOCATION_NO), True)
    db, bot, _ = route(item, scorer)
    bot.decline_chat_join_request.assert_awaited_once()
    assert db.states[-1] is CandidateStatus.AUTO_REJECTED
    assert scorer.review_calls[0]["criterion"] == "location"


def test_handler_without_a_proposal_does_not_call_extra_llm():
    item = fake_candidate(location="Я в Тбилиси")
    scorer = Scorer(score(), True)
    db, bot, _ = route(item, scorer)
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN
    assert not scorer.review_calls


def test_handler_without_reviewer_defaults_to_admin():
    item = fake_candidate()
    scorer = SimpleNamespace(score=AsyncMock(return_value=(
        score(location_score=0, location_mismatch_evidence=LOCATION_NO), "{}"
    )))
    db, bot, _ = route(item, scorer)
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN


def test_spam_confirmation_must_match_initial_spam_type():
    answers = ("", "", SPAM)
    checks = {"location": "no_relevant_information", "business": "no_relevant_information",
              "goal": "supports_rejection"}
    assert not semantic_review_confirms_rejection(
        review("spam", SPAM, checks=checks, spam_type="fraud"),
        criterion="spam", expected_spam_type="advertising", candidate_answers=answers,
    )
    assert not semantic_review_confirms_rejection(
        review("spam", SPAM, checks=checks, spam_type="advertising"),
        criterion="spam", expected_spam_type=None, candidate_answers=answers,
    )


def test_http_client_retries_json_mode_once_for_legacy_provider(monkeypatch):
    import httpx
    from app import scoring as scoring_module

    seen = []
    raw = review("location", LOCATION_NO, checks=loc_checks())

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(400, json={"error": "response_format unsupported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})

    transport = httpx.MockTransport(respond)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(scoring_module.httpx, "AsyncClient", lambda **kwargs:
                        original_client(transport=transport, **kwargs))
    scorer = AIScorer(base_url="https://test.invalid/v1", api_key="fake", model="test", timeout_seconds=1)
    confirmed = asyncio.run(scorer.verify_rejection(
        criterion="location", location=LOCATION_NO, business="Не знаю", goal="Нет дополнительных сведений"
    ))
    assert confirmed is True
    assert len(seen) == 2
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in seen[1]
    assert seen[0]["messages"][0]["content"] == SEMANTIC_REVIEW_PROMPT


def test_http_second_review_failure_requires_admin(monkeypatch):
    import httpx
    from app import scoring as scoring_module

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": "service unavailable"})
    )
    monkeypatch.setattr(scoring_module.httpx, "AsyncClient", lambda **kwargs:
                        original_client(transport=transport, **kwargs))
    scorer = AIScorer(base_url="https://test.invalid/v1", api_key="fake", model="test", timeout_seconds=1)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(scorer.verify_rejection(
            criterion="location", location=LOCATION_NO, business="", goal=""
        ))


def test_legacy_boolean_confirmation_cannot_bypass_third_stage():
    item = fake_candidate()
    scorer = SimpleNamespace(
        score=AsyncMock(return_value=(
            score(location_score=0, location_mismatch_evidence=LOCATION_NO), "{}"
        )),
        review_rejection=AsyncMock(return_value=True),
    )
    db, bot, _ = route(item, scorer)
    scorer.review_rejection.assert_awaited_once()
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN


def test_real_scorer_returns_structured_review_not_boolean():
    scorer = AIScorer(
        base_url="https://llm.invalid/v1", api_key="test", model="x", timeout_seconds=1,
    )
    scorer._request_json = AsyncMock(return_value=review(
        "location", LOCATION_NO, checks=loc_checks()
    ))
    parsed = asyncio.run(scorer.review_rejection(
        criterion="location", location=LOCATION_NO,
        business="Дополнительных сведений нет", goal="Интересуюсь сообществом",
    ))
    assert parsed.criterion == "location"
    assert parsed.verdict == "reject"
    assert parsed.evidence_quote == LOCATION_NO
    assert parsed.checks()["location"] == "supports_rejection"
    assert parsed.checks()["goal"] == "no_relevant_information"


def test_unexpected_decision_gate_error_keeps_candidate_pending(monkeypatch):
    from app import handlers as handlers_module
    def broken_gate(**_kwargs):
        raise RuntimeError("broken decision gate")
    monkeypatch.setattr(handlers_module, "decide_auto_rejection", broken_gate)
    item = fake_candidate()
    scorer = Scorer(score(location_score=0, location_mismatch_evidence=LOCATION_NO), True)
    db, bot, _ = route(item, scorer)
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN


def test_handler_conflicting_review_explanation_goes_to_admin():
    item = fake_candidate()

    class ContradictoryReviewer(Scorer):
        async def review_rejection(self, **kwargs):
            self.review_calls.append(kwargs)
            payload = json.loads(review("location", LOCATION_NO, checks=loc_checks()))
            payload["explanation"] = "Отказывать нельзя: кандидат живёт в Тбилиси"
            # A rejected old-format response is treated as an AI parse failure.
            return parse_semantic_review(json.dumps(payload, ensure_ascii=False))

    scorer = ContradictoryReviewer(
        score(location_score=0, location_mismatch_evidence=LOCATION_NO)
    )
    db, bot, _ = route(item, scorer)
    assert scorer.review_calls
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN

# Offline integration: actual scorer HTTP flow + production routing, with both
# external APIs replaced by test doubles. No network traffic or real decline.
@pytest.mark.parametrize("goal, expected_decline", [
    ("Нет дополнительных сведений", True),
    ("В декабре у меня конференция в Тбилиси, уже зарегистрировался", False),
    ("Забронирован номер в гостинице Батуми на следующую неделю", False),
    ("Мне подтвердили бронь гостиницы в Тбилиси", False),
])
def test_real_scorer_http_two_stage_to_bot_route_is_safe(monkeypatch, goal, expected_decline):
    import httpx
    from app import scoring as scoring_module

    calls = []
    first_reply = json.dumps({
        "location_score": 0, "business_score": 90, "intent_score": 70,
        "spam_score": 0, "spam_type": "none", "ai_confidence": 99,
        "summary": "Тестовый ответ", "risk_flags": [],
        "spam_evidence": "", "location_mismatch_evidence": LOCATION_NO,
        "business_mismatch_evidence": "",
    }, ensure_ascii=False)
    # Both AI stages can be wrong: second pass asserts no relevant information
    # even when the goal contains travel arrangements.
    second_reply = review("location", LOCATION_NO, checks=loc_checks())

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["messages"][1]["role"] == "user"
        assert goal in payload["messages"][1]["content"]
        content = first_reply if len(calls) == 1 else second_reply
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    transport = httpx.MockTransport(respond)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(scoring_module.httpx, "AsyncClient", lambda **kwargs:
                        original_client(transport=transport, **kwargs))
    scorer = AIScorer(base_url="https://test.invalid/v1", api_key="test", model="x",
                      timeout_seconds=1)
    db, bot, _handler = route(fake_candidate(goal=goal), scorer)
    assert len(calls) == 2
    if expected_decline:
        bot.decline_chat_join_request.assert_awaited_once()
        assert db.states[-1] is CandidateStatus.AUTO_REJECTED
    else:
        bot.decline_chat_join_request.assert_not_awaited()
        assert db.states[-1] is CandidateStatus.PENDING_ADMIN


@pytest.mark.parametrize("criterion, clarification", [
    ("location", "В декабре у меня конференция в Тбилиси, уже зарегистрировался"),
    ("business", "Я фотограф, снимаю свадьбы по заказу"),
])
def test_handler_two_models_wrongly_support_extra_field_never_declines(criterion, clarification):
    # Both stages agree on a false rejection; the quote exists verbatim in
    # goal, but its meaning cannot be determined by literal matching alone.
    if criterion == "location":
        item = fake_candidate(location=LOCATION_NO, business="Не знаю", goal=clarification)
        result = score(location_score=0, location_mismatch_evidence=LOCATION_NO)
        main_quote = LOCATION_NO
    else:
        item = fake_candidate(location="Не знаю", business=BUSINESS_NO, goal=clarification)
        result = score(business_score=0, business_mismatch_evidence=BUSINESS_NO)
        main_quote = BUSINESS_NO
    statuses = loc_checks(goal="supports_rejection") if criterion == "location" else biz_checks(goal="supports_rejection")
    payload = json.loads(review(criterion, main_quote, checks=statuses))
    payload["field_evidence"]["goal"]["supporting_quotes"] = [clarification]
    wrong_but_valid_review = parse_semantic_review(json.dumps(payload, ensure_ascii=False))
    scorer = SimpleNamespace(
        score=AsyncMock(return_value=(result, "{}")),
        review_rejection=AsyncMock(return_value=wrong_but_valid_review),
    )
    db, bot, handler = route(item, scorer)
    scorer.review_rejection.assert_awaited_once()
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN
    assert handler._send_admin_card.await_args.kwargs["status_note"]


@pytest.mark.parametrize('criterion,continuation', [
    ('location', 'Уже в Тбилиси'),
    ('location', 'Только что прилетел в Батуми'),
    ('location', 'Уже обосновался в Тбилиси'),
    ('business', 'Я фотограф, снимаю свадьбы по заказу'),
    ('business', 'У меня небольшое кафе, обслуживаю клиентов'),
    ('business', 'Занимаюсь репетиторством и даю частные уроки'),
])
def test_primary_answer_extra_sentence_never_calls_telegram_decline(criterion, continuation):
    # The first AI incorrectly chooses an authentic negative excerpt. A
    # conflicting sentence in the *same field* must prevent even a stage-2 call.
    initial = LOCATION_NO if criterion == 'location' else BUSINESS_NO
    mixed = initial + '. ' + continuation
    item = (fake_candidate(location=mixed) if criterion == 'location' else
            fake_candidate(location='Не знаю', business=mixed))
    result = score(location_score=0, location_mismatch_evidence=initial) if criterion == 'location' else \
             score(business_score=0, business_mismatch_evidence=initial)
    scorer = Scorer(result, True)
    db, bot, _ = route(item, scorer)
    assert scorer.review_calls == []
    bot.decline_chat_join_request.assert_not_awaited()
    assert db.states[-1] is CandidateStatus.PENDING_ADMIN
