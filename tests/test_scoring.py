from app.spam_review import SpamVerification
from app.core_review import CoreVerification

from app.scoring import (
    SYSTEM_PROMPT,
    ScoreResult,
    auto_reject_reason,
    calculate_confidence_score,
    parse_score_json,
)


def make_score(**overrides):
    base = dict(
        location_score=80,
        business_score=85,
        intent_score=75,
        spam_score=5,
        ai_confidence=90,
        confidence_score=78,
        summary="Предприниматель в Батуми с понятной целью вступления.",
        risk_flags=[],
    )
    base.update(overrides)
    return ScoreResult(**base)


def test_calculate_confidence_score_penalizes_spam():
    clean = calculate_confidence_score(90, 90, 90, 0)
    spammy = calculate_confidence_score(90, 90, 90, 90)
    assert clean > spammy
    assert 0 <= spammy <= 100


def test_auto_reject_obvious_spam():
    score = make_score(
        spam_score=95, ai_confidence=95,
        spam_verification=SpamVerification("confirmed", "Доказанная массовая рассылка"),
    )
    assert auto_reject_reason(
        score,
        spam_threshold=85,
        core_score_max=15,
        min_ai_confidence=80,
    ) is not None


def test_low_ai_confidence_never_auto_rejects():
    score = make_score(spam_score=99, ai_confidence=50)
    assert auto_reject_reason(
        score,
        spam_threshold=85,
        core_score_max=15,
        min_ai_confidence=80,
    ) is None


def test_auto_reject_clear_location_mismatch():
    score = make_score(
        location_score=0, ai_confidence=95,
        core_verifications=(CoreVerification(
            "location", "confirmed", "Кандидат явно не планирует приезжать в Грузию"
        ),),
    )
    reason = auto_reject_reason(
        score,
        spam_threshold=85,
        core_score_max=15,
        min_ai_confidence=80,
    )
    assert reason and "локации" in reason


def test_ambiguous_case_goes_to_admin():
    score = make_score(location_score=35, business_score=50, ai_confidence=90)
    assert auto_reject_reason(
        score,
        spam_threshold=85,
        core_score_max=15,
        min_ai_confidence=80,
    ) is None


def test_parse_score_json():
    result = parse_score_json(
        '{"location_score":90,"business_score":80,"intent_score":70,'
        '"spam_score":5,"ai_confidence":88,"summary":"OK","risk_flags":[]}'
    )
    assert result.location_score == 90
    assert result.confidence_score > 0


def test_georgia_location_prompt_covers_all_regions_and_planned_arrivals():
    assert "на территории Грузии" in SYSTEM_PROMPT
    assert "Батуми, Тбилиси, Кутаиси" in SYSTEM_PROMPT
    assert "конкретный план и даты приезда в Грузию" in SYSTEM_PROMPT
    assert "в любом другом регионе Грузии" in SYSTEM_PROMPT


def test_location_rejection_reason_names_georgia():
    reason = auto_reject_reason(
        make_score(
            location_score=0, ai_confidence=95,
            core_verifications=(CoreVerification(
                "location", "confirmed", "Кандидат не планирует поездок в Грузию"
            ),),
        ),
        spam_threshold=85,
        core_score_max=15,
        min_ai_confidence=80,
    )
    assert reason is not None
    assert "Грузии" in reason
    assert "несоответствие" in reason
