"""Stage 3: deterministic, fail-closed authorization of automatic rejection.

This module is the *only* production path allowed to authorize a Telegram
join-request decline on the basis of AI scoring. Neither AI pass can decline
alone. The gate makes no network requests and never approves a candidate.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .scoring import ScoreResult

Criterion = Literal["location", "business", "spam"]
Status = Literal["supports_rejection", "no_relevant_information", "context_only", "contradicts", "uncertain"]
FIELDS = ("location", "business", "goal")
ALLOWED_STATUSES = frozenset((
    "supports_rejection", "no_relevant_information", "context_only",
    "contradicts", "uncertain"
))
SPAM_TYPES = frozenset(("advertising", "fraud", "mass_template", "gibberish", "manipulation"))


@dataclass(frozen=True)
class RejectionProposal:
    criterion: Criterion
    evidence_quote: str
    spam_type: str | None
    reason: str


@dataclass(frozen=True)
class SemanticReview:
    verdict: Literal["reject", "review"]
    criterion: Criterion
    spam_type: str
    field_checks: tuple[tuple[str, Status], ...]
    evidence_quote: str
    no_relevance_basis: tuple[tuple[str, str], ...]
    field_evidence: tuple[tuple[str, "FieldEvidence"], ...]
    overall_consistency: Literal["consistent", "contradicted", "uncertain"]
    eligibility_signal: Literal["absent", "present", "uncertain"]
    explanation_code: Literal[
        "explicit_location_mismatch", "explicit_business_mismatch", "explicit_spam",
        "needs_review",
    ]

    def evidence(self) -> dict[str, "FieldEvidence"]:
        return dict(self.field_evidence)

    def checks(self) -> dict[str, Status]:
        return dict(self.field_checks)

    def absence_bases(self) -> dict[str, str]:
        return dict(self.no_relevance_basis)


@dataclass(frozen=True)
class FieldEvidence:
    supporting_quotes: tuple[str, ...]
    contradicting_quotes: tuple[str, ...]
    uncertainty_reason: str


@dataclass(frozen=True)
class GateDecision:
    allow_auto_reject: bool
    code: str
    reason: str | None = None


def parse_semantic_review(raw: str) -> SemanticReview:
    """Strict version-3 JSON; old/partial reviews always require manual review."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty semantic review")
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean).strip()
    data = json.loads(clean)
    required = {
        "schema_version", "verdict", "criterion", "spam_type", "field_checks",
        "field_evidence", "no_relevance_basis", "overall_consistency",
        "eligibility_signal", "explanation_code", "evidence_quote",
    }
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("Unexpected or incomplete semantic review fields")
    if type(data["schema_version"]) is not int or data["schema_version"] != 3:
        raise ValueError("Unknown review schema version")
    if data["verdict"] not in ("reject", "review") or data["criterion"] not in (
        "location", "business", "spam"
    ):
        raise ValueError("Invalid review verdict or criterion")
    if data["spam_type"] not in SPAM_TYPES | {"none"}:
        raise ValueError("Invalid spam type")
    checks = data["field_checks"]
    if not isinstance(checks, dict) or set(checks) != set(FIELDS):
        raise ValueError("Each answer requires a status")
    if any(not isinstance(checks[key], str) or checks[key] not in ALLOWED_STATUSES
           for key in FIELDS):
        raise ValueError("Invalid field check")
    bases = data["no_relevance_basis"]
    if (not isinstance(bases, dict) or set(bases) != set(FIELDS)
            or any(bases[f] not in ("explicit_neutral", "not_applicable")
                   for f in FIELDS)):
        raise ValueError("Invalid no_relevance_basis")
    evidence = data["field_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(FIELDS):
        raise ValueError("Each answer requires separate evidence")
    parsed_evidence = []
    for field in FIELDS:
        block = evidence[field]
        if not isinstance(block, dict) or set(block) != {
            "supporting_quotes", "contradicting_quotes", "uncertainty_reason"
        }:
            raise ValueError("Invalid field evidence block")
        for key in ("supporting_quotes", "contradicting_quotes"):
            quotes = block[key]
            if (not isinstance(quotes, list) or len(quotes) > 5
                    or any(not isinstance(q, str) or not q.strip() or len(q) > 1000
                           for q in quotes)):
                raise ValueError("Invalid evidence quotes")
        note = block["uncertainty_reason"]
        if not isinstance(note, str) or len(note) > 500:
            raise ValueError("Invalid uncertainty reason")
        parsed_evidence.append((field, FieldEvidence(
            tuple(block["supporting_quotes"]),
            tuple(block["contradicting_quotes"]), note,
        )))
    if data["overall_consistency"] not in ("consistent", "contradicted", "uncertain"):
        raise ValueError("Invalid consistency status")
    if data["eligibility_signal"] not in ("absent", "present", "uncertain"):
        raise ValueError("Invalid eligibility signal")
    if data["explanation_code"] not in (
        "explicit_location_mismatch", "explicit_business_mismatch", "explicit_spam",
        "needs_review",
    ):
        raise ValueError("Invalid explanation code")
    quote = data["evidence_quote"]
    if not isinstance(quote, str) or len(quote) > 2500:
        raise ValueError("Invalid evidence quote")
    return SemanticReview(
        verdict=data["verdict"], criterion=data["criterion"],
        spam_type=data["spam_type"], field_checks=tuple((f, checks[f]) for f in FIELDS),
        field_evidence=tuple(parsed_evidence),
        no_relevance_basis=tuple((f, bases[f]) for f in FIELDS),
        overall_consistency=data["overall_consistency"],
        eligibility_signal=data["eligibility_signal"],
        explanation_code=data["explanation_code"],
        evidence_quote=quote,
    )


# This list is an intentionally TINY, exhaustive allowance for answers that
# make no factual claims. It is NOT a list of ways to describe someone's job or
# travel plans; unknown phrases must go to a human instead of being classified
# as irrelevant by a mistaken LLM. Maintain a conservative allowlist.
_EXPLICIT_NEUTRAL_ANSWERS = frozenset({
    "нет", "не знаю", "не указано", "без комментариев", "нет данных",
    "нет дополнительных сведений", "сведения отсутствуют", "ничего добавить",
    "не применимо", "no information", "no additional information", "n a",
    "none", "not applicable", "no comment", "хочу вступить в сообщество",
})


def _explicitly_neutral_answer(answer: str) -> bool:
    from .scoring import _words
    normalized = _words(answer)
    return bool(normalized) and normalized in _EXPLICIT_NEUTRAL_ANSWERS


def verified_rejection_explanation(proposal: RejectionProposal, review: SemanticReview) -> str:
    """Deterministic audit text; never incorporate AI-authored explanations."""
    labels = {
        "location": "Явное несоответствие критерию локации в Грузии",
        "business": "Явное несоответствие критерию деловой деятельности",
        "spam": "Подтверждённое нарушение правил: спам",
    }
    # Evidence is already required to be present verbatim in the original
    # questionnaire. Limit its size for audit/chat readability.
    quote = " ".join(review.evidence_quote.split())
    if len(quote) > 160:
        quote = quote[:159] + "…"
    return f"{labels[proposal.criterion]} (два AI-этапа и Decision Gate): «{quote}»"


def review_contract_failure(review: SemanticReview, answers: Sequence[str]) -> str | None:
    """Check internal JSON consistency and locate EVERY quoted claim in its field.

    This only validates *reported* facts, not the absence of facts AI missed.
    """
    from .scoring import _normalize_for_match  # import cycle safe
    from .safety_rules import _verified_non_spam_context
    if len(answers) != 3 or any(not isinstance(a, str) for a in answers):
        return "invalid_answers"
    statuses = review.checks()
    evidence = review.evidence()
    bases = review.absence_bases()
    if (len(review.field_checks) != 3 or len(review.field_evidence) != 3
            or len(review.no_relevance_basis) != 3
            or set(statuses) != set(FIELDS) or set(evidence) != set(FIELDS)
            or set(bases) != set(FIELDS)):
        return "invalid_evidence_schema"
    supporting = []
    for field, answer in zip(FIELDS, answers):
        block = evidence[field]
        if not isinstance(block, FieldEvidence):
            return "invalid_evidence_schema"
        if any("…" in quote or "..." in quote for quote in
               (*block.supporting_quotes, *block.contradicting_quotes)):
            return "fragmented_field_quote"
        for quote in (*block.supporting_quotes, *block.contradicting_quotes):
            normalized = _normalize_for_match(quote)
            source = _normalize_for_match(answer)
            if len(normalized) < 12 or not re.search(
                r"(?<!\w)" + re.escape(normalized) + r"(?!\w)", source
            ):
                return "unverifiable_field_quote"
        status = statuses[field]
        basis = bases[field]
        if basis not in ("explicit_neutral", "not_applicable"):
            return "invalid_no_relevance_basis"
        if status == "no_relevant_information":
            if basis != "explicit_neutral" or not _explicitly_neutral_answer(answer):
                return "unverified_absence_of_relevance"
        elif basis != "not_applicable":
            return "unexpected_no_relevance_basis"
        if status not in ALLOWED_STATUSES:
            return "invalid_field_status"
        if status == "supports_rejection":
            if (not block.supporting_quotes or block.contradicting_quotes
                    or block.uncertainty_reason.strip()):
                return "inconsistent_support_evidence"
            supporting.extend(block.supporting_quotes)
        elif status == "no_relevant_information":
            if (block.supporting_quotes or block.contradicting_quotes
                    or block.uncertainty_reason.strip()):
                return "unreported_relevant_evidence"
        elif status == "context_only":
            # Only the spam pathway can treat substantive, independently
            # checked location/profession facts as non-exculpatory context.
            # It may NEVER be used to hide a contradiction in a core rejection.
            if review.criterion != "spam":
                return "context_only_not_allowed_for_core"
            if (block.supporting_quotes or block.contradicting_quotes
                    or block.uncertainty_reason.strip()):
                return "inconsistent_context_evidence"
            if not _verified_non_spam_context(field, answer):
                return "unverified_context_only"
        elif status == "contradicts":
            if not block.contradicting_quotes or block.supporting_quotes:
                return "inconsistent_contradiction_evidence"
        elif status == "uncertain" and not block.uncertainty_reason.strip():
            return "missing_uncertainty_reason"
    # A supporting quote in a non-criterion field can be literally accurate
    # while proving the OPPOSITE of the proposed core rejection. Neither LLM
    # label nor literal matching establishes the quote's meaning. For location
    # and business rejections, allow auto-reject ONLY when the other two entire
    # answers are independently known to be explicitly neutral. Otherwise a
    # human must review the full questionnaire. This also covers the case where
    # BOTH models make the same incorrect supports_rejection classification.
    # Spam has its own independent evidence and context checks below.
    if review.criterion in ("location", "business"):
        for field, answer in zip(FIELDS, answers):
            if field == review.criterion:
                continue
            if (statuses[field] != "no_relevant_information"
                    or bases[field] != "explicit_neutral"
                    or not _explicitly_neutral_answer(answer)):
                return "noncriterion_answer_requires_review"
    if review.verdict != "reject":
        return "review_verdict"
    if (review.overall_consistency != "consistent"
            or review.eligibility_signal != "absent"):
        return "global_conflict_or_uncertainty"
    if any(statuses[f] in ("uncertain", "contradicts") for f in FIELDS):
        return "field_conflict_or_uncertainty"
    expected_code = {
        "location": "explicit_location_mismatch",
        "business": "explicit_business_mismatch",
        "spam": "explicit_spam",
    }[review.criterion]
    if review.explanation_code != expected_code:
        return "explanation_code_mismatch"
    if (not review.evidence_quote.strip() or
            _normalize_for_match(review.evidence_quote) not in {
                _normalize_for_match(q) for q in supporting
            }):
        return "quote_not_in_supporting_evidence"
    # Stage-3 independent guard on the COMPLETE primary answer. Even when
    # both models label the same negative excerpt as supports_rejection,
    # nothing authorizes rejecting a candidate who adds substantive text in
    # that SAME field. The deterministic full-answer grammar must be satisfied.
    # This check is deliberately duplicated after stage 1: a malformed or
    # fabricated stage-2 review cannot bypass it.
    if review.criterion in ("location", "business"):
        from .scoring import _fully_verified_core_answer
        index = FIELDS.index(review.criterion)
        if not _fully_verified_core_answer(
            review.criterion, review.evidence_quote, answers[index]
        ):
            return "ambiguous_criterion_answer"
    return None


def decide_auto_rejection(
    *, proposal: RejectionProposal | None, review: SemanticReview | None,
    score: ScoreResult, candidate_answers: Sequence[str],
    spam_threshold: int, core_score_max: int, min_ai_confidence: int,
) -> GateDecision:
    """Authorize ONLY if the two AI passes and independent checks agree.

    Re-evaluating the first-stage proposal prevents downstream code from
    supplying an arbitrary reason or an unchecked quote. All failures abstain.
    """
    from . import scoring  # local import avoids an import cycle

    deny = lambda code: GateDecision(False, code)
    if not isinstance(proposal, RejectionProposal) or not isinstance(review, SemanticReview):
        return deny("missing_proposal_or_review")
    if len(candidate_answers) != 3 or any(not isinstance(a, str) or not a.strip()
                                           for a in candidate_answers):
        return deny("incomplete_questionnaire")
    expected = scoring.propose_auto_rejection(
        score, spam_threshold=spam_threshold, core_score_max=core_score_max,
        min_ai_confidence=min_ai_confidence, candidate_answers=candidate_answers,
    )
    if expected is None or proposal != expected:
        return deny("proposal_mismatch")
    if proposal.criterion not in ("location", "business", "spam"):
        return deny("invalid_criterion")
    if review.verdict != "reject" or review.criterion != proposal.criterion:
        return deny("verdict_or_criterion_mismatch")
    contract_error = review_contract_failure(review, candidate_answers)
    if contract_error is not None:
        return deny(contract_error)
    if proposal.criterion == "spam":
        if proposal.spam_type not in SPAM_TYPES or review.spam_type != proposal.spam_type:
            return deny("spam_type_mismatch")
    elif proposal.spam_type is not None or review.spam_type != "none":
        return deny("unexpected_spam_type")
    checks = review.checks()
    if (len(review.field_checks) != len(FIELDS) or set(checks) != set(FIELDS)
            or any(checks[key] not in ALLOWED_STATUSES for key in FIELDS)):
        return deny("invalid_field_checks")
    if any(checks[key] in ("uncertain", "contradicts") for key in FIELDS):
        return deny("contradiction_or_uncertainty")
    if proposal.criterion in ("location", "business"):
        if checks[proposal.criterion] != "supports_rejection":
            return deny("missing_criterion_support")
    elif "supports_rejection" not in checks.values():
        return deny("missing_spam_support")
    quote = review.evidence_quote
    if not quote.strip() or "…" in quote or "..." in quote:
        return deny("missing_or_fragmented_quote")
    if proposal.criterion == "location":
        if not scoring._explicit_location_mismatch(
            quote, candidate_answers[0], candidate_answers
        ):
            return deny("unconfirmed_location_evidence")
    elif proposal.criterion == "business":
        if not scoring._explicit_business_mismatch(
            quote, candidate_answers[1], candidate_answers
        ):
            return deny("unconfirmed_business_evidence")
    else:
        if (not scoring.evidence_found_in_answers(quote, candidate_answers)
                or not scoring._unambiguous_spam_evidence(proposal.spam_type, quote)
                or any(scoring._has_spam_warning_or_reported_context(answer)
                       for answer in candidate_answers)):
            return deny("unconfirmed_spam_evidence")
    return GateDecision(True, "confirmed", verified_rejection_explanation(proposal, review))
