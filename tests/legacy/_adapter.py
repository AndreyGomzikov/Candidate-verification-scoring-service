"""Translate the previous ScoreResult's fixtures into the merged evidence contract.

Only for regression tests, never used by bot or production Decision Gate.
"""
from app.scoring import ScoreResult, CriterionConfidence
from app.core_review import CoreEvidence
from app.spam_review import SpamEvidence


def score_as_current(*args, **kwargs):
    assert not args, "Legacy tests use keyword fixtures"
    legacy_location = kwargs.pop("location_mismatch_evidence", "")
    legacy_business = kwargs.pop("business_mismatch_evidence", "")
    legacy_spam = kwargs.pop("spam_evidence", "")
    legacy_spam_type = kwargs.pop("spam_type", "none")
    core = []
    grades = []
    for field, quote in (("location", legacy_location), ("business", legacy_business)):
        if quote:
            core.append(CoreEvidence(field, quote, "Строгая проверка старой регрессии"))
            grades.append(CriterionConfidence(
                field, kwargs.get("ai_confidence", 0), 100, 100, 100, 100,
                True, quote,
            ))
    spam = ()
    if legacy_spam:
        old_to_new = {
            "advertising": "mass_advertising",
            "fraud": "fraudulent_scheme",
        }
        category = old_to_new.get(legacy_spam_type)
        if category:
            spam = (SpamEvidence(category, "goal", legacy_spam, "Прямое рекламное обращение"),)
            grades.append(CriterionConfidence(
                "spam", kwargs.get("ai_confidence", 0), 100, 100, 100, 100,
                True, legacy_spam,
            ))
    kwargs["core_evidence"] = tuple(core)
    kwargs["spam_evidence"] = spam
    kwargs["decision_confidences"] = tuple(grades)
    return ScoreResult(**kwargs)
