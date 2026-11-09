from __future__ import annotations

import asyncio
import html
import logging
import json
import re
import math
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from io import BytesIO
from weakref import WeakValueDictionary
from datetime import datetime, timedelta, timezone

from telegram import Bot, InputFile, Update
from telegram.constants import ChatMemberStatus, ChatType, ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from . import texts
from .config import Settings
from .db import Database
from .keyboards import moderation_keyboard
from .models import Candidate, CandidateStatus
from .status_report import build_status_report
from .scoring import (
    AIScorer, ScoreResult, propose_auto_rejection, with_spam_verification,
    finalize_candidate_score,
)
from .decision import GateDecision, decide_auto_rejection
from .core_review import CoreVerification
from .spam_review import SpamVerification
from .support import SupportHandlers

logger = logging.getLogger(__name__)

_TELEGRAM_TEXT_LIMIT = 4096
_ADMIN_CARD_SAFE_LIMIT = 3900


def _join_request_missing(exc: TelegramError) -> bool:
    """Telegram no longer has a pending join request for this user/chat.

    This is deliberately narrow: lack of permission, rate limits, and network
    errors must NOT be mistaken for a withdrawn or processed request.
    """
    return "hide_requester_missing" in str(exc).casefold()


def _newer_join_request(req: object, active: Candidate) -> bool:
    """Only replace an active flow for a strictly newer Telegram request.

    ChatJoinRequest.date is second-precision and stable across update retries;
    candidate.created_at is NOT the request date and must not be compared.
    Same-second withdraw/reapply cannot be distinguished reliably by Bot API:
    fail closed and keep the current interview rather than silently replacing it.
    Legacy rows without request_date cannot be distinguished either.
    """
    if active.status not in (
        CandidateStatus.INTERVIEW.value, CandidateStatus.SCORING.value,
        CandidateStatus.PENDING_ADMIN.value,
    ):
        return False
    request_date = getattr(req, "date", None)
    previous_date = getattr(active, "request_date", None)
    if not isinstance(request_date, datetime) or not isinstance(previous_date, datetime):
        return False
    if request_date.tzinfo is None:
        request_date = request_date.replace(tzinfo=timezone.utc)
    if previous_date.tzinfo is None:
        previous_date = previous_date.replace(tzinfo=timezone.utc)
    return request_date > previous_date


def _escape_clip(value: str, limit: int) -> str:
    """HTML-escape text while limiting the *escaped* output length."""
    if limit <= 0:
        return ""
    out: list[str] = []
    used = 0
    for char in value:
        escaped = html.escape(char)
        if used + len(escaped) > limit - 1:
            out.append("…")
            break
        out.append(escaped)
        used += len(escaped)
    return "".join(out)


class GatekeeperHandlers:
    """Telegram event flow adapted from Ringo's join-request/questionnaire approach."""

    def __init__(self, *, settings: Settings, db: Database, scorer: AIScorer):
        self.settings = settings
        self.db = db
        self.scorer = scorer
        self._active_scoring_ids: set[int] = set()
        # Prevent one admin from starting overlapping full-history deliveries.
        self._status_running_users: set[int] = set()
        # Keep maintenance from treating a slow, still-running Telegram update
        # as a process crash. The database ledger survives a real restart.
        self._inflight_join_update_ids: set[int] = set()
        # Serializes actions for one (community, Telegram user) in this bot.
        # Weak references prevent retaining a lock for every past candidate.
        self._join_locks: WeakValueDictionary[tuple[int, int], asyncio.Lock] = WeakValueDictionary()
        self.support = SupportHandlers(
            settings=settings, db=db, is_allowed_admin=self._is_allowed_admin
        )

    async def on_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.effective_chat and update.effective_chat.type == ChatType.PRIVATE:
            # The channel's "Ответить" button opens this deep link. Telegram
            # sends /start sreply_N.
            args = getattr(context, "args", None) or []
            if len(args) == 1 and re.fullmatch(r"sreply_[1-9][0-9]*", args[0]):
                await self.support.start_private_admin_reply(
                    update, context, int(args[0].split("_", 1)[1])
                )
                return
            await update.effective_message.reply_text(
                "Этот бот обрабатывает заявки на вступление в Solidus. "
                "Подайте заявку через ссылку сообщества — бот автоматически начнёт анкету."
            )

    @asynccontextmanager
    async def _request_lock(self, target_chat_id: int, user_id: int):
        """Serialize Telegram side effects + DB updates per applicant.

        PostgreSQL's advisory transaction lock also covers another bot worker.
        Telegram itself has no per-attempt join-request ID, so user-initiated
        withdrawal/reapplication outside the bot remains an API limitation.
        """
        key = (target_chat_id, user_id)
        lock = self._join_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._join_locks[key] = lock
        async with lock:
            if isinstance(self.db, Database):
                async with self.db.join_request_lock(target_chat_id, user_id):
                    yield
            else:
                yield

    async def on_join_request(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        req = update.chat_join_request
        if req is None or req.chat.id != self.settings.target_chat_id:
            return
        # Real Telegram updates always have update_id. Persist BEFORE the first
        # external Telegram action. Test fakes without update_id remain valid.
        update_id = getattr(update, "update_id", None)
        audit_id = update_id if isinstance(update_id, int) and isinstance(self.db, Database) else None
        logger.info(
            "join_request_received update_id=%s chat_id=%s user_id=%s date=%s",
            update_id, req.chat.id, req.from_user.id, getattr(req, "date", None),
        )
        if audit_id is not None:
            self._inflight_join_update_ids.add(audit_id)
        try:
            if audit_id is not None:
                try:
                    await self.db.record_join_request_event(
                        update_id=audit_id, target_chat_id=req.chat.id,
                        user_id=req.from_user.id, request_date=getattr(req, "date", None),
                    )
                except Exception:
                    logger.exception("Could not persist incoming join event update=%s", audit_id)
                    # If the audit DB is down, the normal candidate DB is very
                    # likely down too. Raise and warn the moderation channel.
                    try:
                        await context.bot.send_message(
                            self.settings.admin_chat_id,
                            "⚠️ Сбой регистрации входящей заявки Telegram. "
                            f"TG ID: {req.from_user.id}, update_id: {audit_id}. "
                            "Проверьте работу PostgreSQL и бота.",
                        )
                    except Exception:
                        logger.exception("Could not warn administrators about DB failure")
                    raise
            try:
                async with self._request_lock(req.chat.id, req.from_user.id):
                    await self._handle_join_request(req, context, audit_update_id=audit_id)
            except Exception as exc:
                logger.exception(
                    "Join request processing interrupted update_id=%s user_id=%s",
                    update_id, req.from_user.id,
                )
                if audit_id is not None:
                    try:
                        await self.db.update_join_request_event(
                            audit_id, "processing_error", error_type=type(exc).__name__,
                            alert_required=True,
                        )
                    except Exception:
                        logger.exception("Failed to persist join processing error update=%s", audit_id)
                raise
        finally:
            if audit_id is not None:
                self._inflight_join_update_ids.discard(audit_id)

    async def _handle_join_request(self, req, context, *, audit_update_id: int | None = None) -> None:
        async def audit(status: str, **fields) -> None:
            if audit_update_id is not None:
                await self.db.update_join_request_event(audit_update_id, status, **fields)

        # Telegram may redeliver updates or a user may rapidly re-apply. Reuse an
        # already active flow instead of creating duplicate interviews/LLM calls.
        active = await self.db.get_active_candidate_for_chat(
            req.from_user.id, req.chat.id
        )
        if active is not None and _newer_join_request(req, active):
            logger.info(
                "Fresh join request supersedes candidate=%s user=%s",
                active.id, req.from_user.id,
            )
            # create_candidate() atomically marks the old active row superseded.
            # An old moderator card cannot approve/reject the new request.
            active = None
        if active is not None:
            # Ignore an older Telegram update arriving after a newer request.
            # In particular, never replace the current temporary DM address
            # with a stale user_chat_id from a superseded attempt.
            current_date = getattr(req, "date", None)
            stored_date = getattr(active, "request_date", None)
            if isinstance(current_date, datetime) and isinstance(stored_date, datetime):
                if current_date.tzinfo is None:
                    current_date = current_date.replace(tzinfo=timezone.utc)
                if stored_date.tzinfo is None:
                    stored_date = stored_date.replace(tzinfo=timezone.utc)
                if current_date < stored_date:
                    logger.info("Ignoring older join update for candidate=%s", active.id)
                    await audit("ignored_older", candidate_id=active.id)
                    return
            active = await self.db.set_user_chat_id(active.id, req.user_chat_id)
            if active.status == CandidateStatus.INTERVIEW.value:
                delivered = await self._send_candidate_message(
                    context.bot, active, self._question_for_stage(active.stage)
                )
            elif active.status == CandidateStatus.SCORING.value:
                delivered = await self._send_candidate_message(context.bot, active, texts.PROCESSING)
            else:
                delivered = await self._send_candidate_message(context.bot, active, texts.PENDING_ADMIN)
            if not delivered and active.status == CandidateStatus.INTERVIEW.value:
                # Do not count failed delivery of an interview question as a
                # candidate who declined to answer.
                await self.db.set_status(
                    active.id, CandidateStatus.PENDING_ADMIN,
                    reason="Не удалось повторно отправить вопрос анкеты; требуется ручная проверка",
                    expected_status=CandidateStatus.INTERVIEW,
                )
            await audit(
                "active_flow_notified" if delivered else "active_delivery_failed",
                candidate_id=active.id, alert_required=not delivered,
                **({"error_type": "TelegramDeliveryFailed"} if not delivered else {}),
            )
            return

        # A finished candidate is not "active", but Telegram may redeliver
        # the *same* chat_join_request much later. Compare against the latest
        # attempt for THIS chat before opening another interview or declining
        # a purported reapplication during cooldown.
        #
        # Legacy rows may not have request_date: we cannot infer their Telegram
        # request timestamp from created_at, so allow the new event in that case.
        latest = await self.db.get_latest_candidate_for_chat(
            req.from_user.id, req.chat.id
        )
        incoming_date = getattr(req, "date", None)
        last_date = getattr(latest, "request_date", None)
        if isinstance(last_date, datetime):
            if not isinstance(incoming_date, datetime):
                logger.info("Ignoring undated join update after candidate=%s", latest.id)
                await audit("ignored_undated", candidate_id=latest.id)
                return
            if incoming_date.tzinfo is None:
                incoming_date = incoming_date.replace(tzinfo=timezone.utc)
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            if incoming_date <= last_date:
                logger.info("Ignoring replayed/older join update after candidate=%s", latest.id)
                await audit("ignored_replayed", candidate_id=latest.id)
                return

        # Cooldown refusals are not Candidate rows. Without a persistent record
        # of those events, an old declined update can resurrect an interview as
        # soon as the cooldown expires. Keep a separate per-chat replay ledger.
        last_cooldown_date = await self.db.last_cooldown_declined_request_date(
            req.from_user.id, req.chat.id
        )
        if isinstance(last_cooldown_date, datetime):
            if not isinstance(incoming_date, datetime):
                logger.info("Ignoring undated event after cooldown decline user=%s", req.from_user.id)
                await audit("ignored_cooldown")
                return
            if incoming_date.tzinfo is None:
                incoming_date = incoming_date.replace(tzinfo=timezone.utc)
            if last_cooldown_date.tzinfo is None:
                last_cooldown_date = last_cooldown_date.replace(tzinfo=timezone.utc)
            if incoming_date <= last_cooldown_date:
                logger.info("Ignoring replayed cooldown-declined join user=%s", req.from_user.id)
                await audit("ignored_cooldown")
                return

        if self.settings.reapplication_cooldown_minutes > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(
                minutes=self.settings.reapplication_cooldown_minutes
            )
            recent = await self.db.get_recent_rejected_candidate(
                req.from_user.id, req.chat.id, cutoff
            )
            if recent is not None:
                # This timestamp is mandatory for persistent replay protection.
                # Telegram's actual ChatJoinRequest always supplies it; if an
                # anomalous event lacks it, fail closed without a Telegram side
                # effect, rather than creating an untraceable decline.
                if not isinstance(incoming_date, datetime):
                    logger.error(
                        "Cannot safely decline undated cooldown join user=%s chat=%s",
                        req.from_user.id, req.chat.id,
                    )
                    await audit("cooldown_unknown_date", error_type="UndatedJoinRequest", alert_required=True)
                    return
                if incoming_date.tzinfo is None:
                    incoming_date = incoming_date.replace(tzinfo=timezone.utc)
                # FIRST commit the intent. If the process crashes at any later
                # point, the marker blocks this event from reopening an
                # interview, even after the cooldown expires.
                started = await self.db.begin_cooldown_decline(
                    req.from_user.id, req.chat.id, incoming_date,
                )
                if not started:
                    logger.info("Ignoring duplicate cooldown intent user=%s", req.from_user.id)
                    await audit("ignored_cooldown")
                    return
                try:
                    await context.bot.decline_chat_join_request(
                        chat_id=req.chat.id, user_id=req.from_user.id
                    )
                except TelegramError:
                    # A timeout might mean Telegram did process the request.
                    # Retain the event marker; never blindly retry this call.
                    await self.db.complete_cooldown_decline(
                        req.from_user.id, req.chat.id, incoming_date,
                        confirmed=False,
                    )
                    logger.exception(
                        "Uncertain cooldown decline; manual verification required user=%s",
                        req.from_user.id,
                    )
                    await audit("cooldown_uncertain")  # Existing cooldown alert ledger handles notification.
                    return
                await self.db.complete_cooldown_decline(
                    req.from_user.id, req.chat.id, incoming_date,
                    confirmed=True,
                )
                # Use the fresh temporary chat id supplied by this join request.
                try:
                    await context.bot.send_message(req.user_chat_id, texts.COOLDOWN)
                except TelegramError:
                    logger.info(
                        "Unable to notify cooldown applicant user=%s", req.from_user.id
                    )
                await audit("cooldown_declined")
                return

        candidate = await self.db.create_candidate(
            user_id=req.from_user.id,
            user_chat_id=req.user_chat_id,
            target_chat_id=req.chat.id,
            username=req.from_user.username,
            first_name=req.from_user.first_name,
            request_date=getattr(req, "date", None),
        )
        await audit("candidate_created", candidate_id=candidate.id)
        logger.info("join_candidate_created update_id=%s candidate=%s", audit_update_id, candidate.id)

        try:
            sent = await context.bot.send_message(
                chat_id=req.user_chat_id,
                text=f"{texts.INTRO}\n\n{texts.QUESTION_LOCATION}",
            )
        except TelegramError as exc:
            # A Telegram timeout is not evidence of non-delivery. Never
            # automatically resend a potentially delivered first question.
            logger.exception("Failed to start interview for candidate=%s", candidate.id)
            await audit(
                "first_question_failed", candidate_id=candidate.id,
                error_type=type(exc).__name__, alert_required=True,
            )
            candidate = await self.db.set_status(
                candidate.id,
                CandidateStatus.PENDING_ADMIN,
                reason=f"Не удалось начать ЛС: {type(exc).__name__}",
            )
            try:
                await self._send_admin_card(
                    context.bot, candidate, score=None,
                    scoring_error=(
                        "Не удалось начать личный диалог с кандидатом. "
                        "Автоматический отказ НЕ выполнен; требуется решение администратора."
                    ),
                )
            except Exception:
                logger.exception("Admin card delivery queued for retry candidate=%s", candidate.id)
            else:
                if audit_update_id is not None:
                    await self.db.mark_join_request_alerted(audit_update_id)
            return
        # Only a successful send_message return confirms Telegram accepted the
        # question. This is NOT proof that the user read it.
        message_id = getattr(sent, "message_id", None)
        await audit("first_question_sent", candidate_id=candidate.id,
                    **({"first_message_id": message_id} if isinstance(message_id, int) else {}))
        logger.info(
            "join_first_question_sent update_id=%s candidate=%s message_id=%s",
            audit_update_id, candidate.id, message_id,
        )

    async def on_private_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (
            update.effective_chat is None
            or update.effective_chat.type != ChatType.PRIVATE
            or update.effective_user is None
            or update.effective_message is None
            or not update.effective_message.text
        ):
            return

        # A moderator replying to the bot's private support prompt must not
        # accidentally create a candidate support ticket or answer an interview.
        if await self.support.maybe_handle_private_admin_reply(update, context):
            return

        prelim = await self.db.get_active_interview(update.effective_user.id)
        if prelim is None:
            await self.support.on_candidate_message(update, context)
            return

        async with self._request_lock(prelim.target_chat_id, prelim.user_id):
            # The expiry task or a new join request may have changed the row
            # between our initial lookup and acquiring the lock.
            current = await self.db.get_active_interview(update.effective_user.id)
            if current is None or current.id != prelim.id:
                return
            to_score = await self._handle_private_interview_answer(update, current)
        if to_score is not None:
            # Never hold the applicant lock for two remote LLM calls.
            await self._score_and_route(context.bot, to_score)

    async def _handle_private_interview_answer(self, update, candidate: Candidate) -> Candidate | None:
        # After the candidate has replied, this is the authoritative private-chat
        # id for future notifications. Do not assume from_user.id == user_chat_id.
        if candidate.user_chat_id != update.effective_chat.id:
            candidate = await self.db.set_user_chat_id(
                candidate.id, update.effective_chat.id
            )

        answer = update.effective_message.text.strip()
        if len(answer) > self.settings.max_answer_length:
            await update.effective_message.reply_text(texts.TOO_LONG)
            return
        if not answer:
            return

        if candidate.stage == 1:
            await self.db.save_answer(
                candidate.id,
                field="location_answer",
                answer=answer,
                next_stage=2,
            )
            await update.effective_message.reply_text(texts.QUESTION_BUSINESS)
            return

        if candidate.stage == 2:
            await self.db.save_answer(
                candidate.id,
                field="business_answer",
                answer=answer,
                next_stage=3,
            )
            await update.effective_message.reply_text(texts.QUESTION_GOAL)
            return

        if candidate.stage != 3:
            return

        # One transaction prevents a crash from leaving stage=4 in interview.
        candidate = await self.db.complete_interview(candidate.id, answer)
        if candidate is None:
            return  # duplicate delivery or already processed by another handler
        # Candidate messaging is best effort; an outage must not suppress AI.
        try:
            await update.effective_message.reply_text(texts.PROCESSING)
        except TelegramError:
            logger.warning(
                "Candidate processing notification failed candidate=%s; scoring continues",
                candidate.id,
            )
        return candidate

    async def _score_and_route(self, bot: Bot, candidate: Candidate) -> None:
        """Track live work so maintenance never recovers an in-flight LLM call."""
        self._active_scoring_ids.add(candidate.id)
        try:
            await self._score_and_route_untracked(bot, candidate)
        finally:
            self._active_scoring_ids.discard(candidate.id)

    async def _score_and_route_untracked(self, bot: Bot, candidate: Candidate) -> None:
        """Three stages: model proposal → independent v3 review → pure Python gate.

        The only path to decline_chat_join_request passes a positive
        GateDecision. Malformed reviews, failed requests, weak/unknown evidence
        and exceptions all default to administrator moderation.
        """
        answers = (
            candidate.location_answer or "",
            candidate.business_answer or "",
            candidate.goal_answer or "",
        )
        proposal = None
        review = None
        review_error = None
        decision = GateDecision(False, "no_proposal")
        try:
            score, raw = await self.scorer.score(
                location=answers[0], business=answers[1], goal=answers[2]
            )
            proposal = propose_auto_rejection(
                score,
                spam_threshold=self.settings.auto_reject_spam_score,
                core_score_max=self.settings.auto_reject_core_score_max,
                min_ai_confidence=self.settings.auto_reject_min_ai_confidence,
                candidate_answers=answers,
            )
            logger.info(
                "Stage1 candidate=%s criterion=%s ai_confidence=%s "
                "criterion_confidence=%s spam_score=%s result=%s",
                candidate.id, proposal.criterion if proposal else "none",
                score.ai_confidence,
                score.confidence_for(proposal.criterion) if proposal else 0,
                score.spam_score, "proposed" if proposal else "manual",
            )
            if proposal is not None:
                try:
                    review = await self.scorer.review_rejection(
                        criterion=proposal.criterion, spam_type=proposal.spam_type,
                        location=answers[0], business=answers[1], goal=answers[2],
                    )
                    logger.info(
                        "Stage2 candidate=%s criterion=%s verdict=%s "
                        "field_checks=%s consistency=%s eligibility=%s",
                        candidate.id, proposal.criterion, review.verdict,
                        dict(review.field_checks), review.overall_consistency,
                        review.eligibility_signal,
                    )
                except Exception as exc:
                    review_error = type(exc).__name__
                    logger.warning(
                        "Stage2 candidate=%s criterion=%s unavailable error=%s",
                        candidate.id, proposal.criterion, review_error,
                    )
                try:
                    decision = decide_auto_rejection(
                        proposal=proposal, review=review, score=score,
                        candidate_answers=answers,
                        spam_threshold=self.settings.auto_reject_spam_score,
                        core_score_max=self.settings.auto_reject_core_score_max,
                        min_ai_confidence=self.settings.auto_reject_min_ai_confidence,
                    )
                except Exception as exc:
                    logger.exception(
                        "Stage3 candidate=%s unexpected gate error=%s",
                        candidate.id, type(exc).__name__,
                    )
                    decision = GateDecision(False, "decision_gate_error")
            logger.info(
                "Stage3 candidate=%s criterion=%s allow=%s code=%s",
                candidate.id, proposal.criterion if proposal else "none",
                decision.allow_auto_reject, decision.code,
            )
            # Show stage-2/3 outcome in the admin card, without implying that
            # agreement by both models alone authorizes a decline.
            if proposal is not None and proposal.criterion in ("location", "business"):
                score = replace(score, core_verifications=(CoreVerification(
                    proposal.criterion,
                    "confirmed" if decision.allow_auto_reject else "not_confirmed",
                    f"Decision Gate: {decision.code}",
                    proposal.evidence_quote if decision.allow_auto_reject else "",
                ),))
            if proposal is not None and proposal.criterion == "spam":
                if decision.allow_auto_reject:
                    original = next((e for e in score.spam_evidence
                                     if e.quote == proposal.evidence_quote), None)
                    spam_verdict = SpamVerification(
                        "confirmed", "Stage 2 и Decision Gate подтвердили нарушение",
                        original.category if original else "",
                        original.source_field if original else "",
                        original.quote if original else "",
                    )
                else:
                    spam_verdict = SpamVerification(
                        "not_confirmed", f"Decision Gate: {decision.code}"
                    )
                score, raw = with_spam_verification(score, raw, spam_verdict)
            score, raw = finalize_candidate_score(
                score, raw, spam_threshold=self.settings.auto_reject_spam_score
            )
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                payload = {"model_response": payload}
            payload["_gatekeeper_decision"] = {
                "version": 3,
                "stage1_proposal": asdict(proposal) if proposal else None,
                "stage1_criterion_confidence": (
                    score.confidence_for(proposal.criterion) if proposal else None
                ),
                "stage2_review": asdict(review) if review else None,
                "stage2_error_type": review_error,
                "stage3": {"allow_auto_reject": decision.allow_auto_reject,
                           "code": decision.code,
                           "reason": decision.reason},
            }
            candidate = await self.db.save_score(
                candidate.id, score, json.dumps(payload, ensure_ascii=False)
            )
            if candidate is None:
                logger.info("Scoring candidate superseded before decision; no Telegram action")
                return
        except Exception as exc:
            logger.exception("AI scoring/routing failed for candidate=%s", candidate.id)
            async with self._request_lock(candidate.target_chat_id, candidate.user_id):
                current = await self.db.get_candidate(candidate.id)
                if current is None or current.status != CandidateStatus.SCORING.value:
                    return  # Candidate was replaced during remote AI work.
                candidate = await self.db.set_status(
                    candidate.id, CandidateStatus.PENDING_ADMIN,
                    reason=f"AI scoring error: {type(exc).__name__}",
                    expected_status=CandidateStatus.SCORING,
                )
                if candidate is None:
                    return
                await self._send_admin_card(
                    bot, candidate, score=None,
                    scoring_error="AI-скоринг временно недоступен; требуется решение администратора.",
                )
                await self._send_candidate_message(bot, candidate, texts.PENDING_ADMIN)
                return

        # Another join event may have superseded this application while the
        # model was working. Protect the final Telegram operation and DB write.
        async with self._request_lock(candidate.target_chat_id, candidate.user_id):
            current = await self.db.get_candidate(candidate.id)
            if current is None or current.status != CandidateStatus.SCORING.value:
                logger.info("Skipping obsolete scored candidate=%s", candidate.id)
                return
            # The bot must never decline solely based on the first score or
            # independent AI verdict; only a deterministic positive gate is enough.
            reason = decision.reason if decision.allow_auto_reject else None
            if reason:
                try:
                    await bot.decline_chat_join_request(
                        chat_id=candidate.target_chat_id, user_id=candidate.user_id
                    )
                except TelegramError as exc:
                    logger.exception("Failed to decline join request candidate=%s", candidate.id)
                    candidate = await self.db.set_status(
                        candidate.id, CandidateStatus.PENDING_ADMIN,
                        reason=(f"Decision Gate разрешил автоотказ: {reason}. "
                                f"Telegram decline failed: {type(exc).__name__}"),
                        expected_status=CandidateStatus.SCORING,
                    )
                    if candidate is None:
                        return
                    await self._send_admin_card(
                        bot, candidate, score=score,
                        status_note=("⚠️ Decision Gate разрешил автоотказ, но Telegram "
                                     "не подтвердил отклонение. Решает администратор."),
                        with_buttons=True,
                    )
                    await self._send_candidate_message(bot, candidate, texts.TECHNICAL_REVIEW)
                    return
                candidate = await self.db.set_status(
                    candidate.id, CandidateStatus.AUTO_REJECTED, reason=reason,
                    expected_status=CandidateStatus.SCORING,
                )
                if candidate is None:
                    logger.error("Telegram declined but DB status changed candidate=%s", current.id)
                    return
                logger.info("Telegram auto-decline completed candidate=%s gate=%s",
                            candidate.id, decision.code)
                await self._send_candidate_message(bot, candidate, texts.REJECTED)
                await self._send_admin_card(
                    bot, candidate, score=score,
                    status_note="🤖 Заявка автоматически отклонена: " + reason,
                    with_buttons=False,
                )
                return

            candidate = await self.db.set_status(
                candidate.id, CandidateStatus.PENDING_ADMIN,
                expected_status=CandidateStatus.SCORING,
            )
            if candidate is None:
                return
            status_note = (
                "🔎 Предложенный автоотказ не прошёл независимый контроль "
                f"({decision.code}). Требуется решение администратора."
                if proposal is not None else None
            )
            await self._send_admin_card(bot, candidate, score=score, status_note=status_note)
            await self._send_candidate_message(bot, candidate, texts.PENDING_ADMIN)

    async def _is_allowed_admin(
        self, context: ContextTypes.DEFAULT_TYPE, user_id: int
    ) -> bool:
        if self.settings.admin_user_ids:
            return user_id in self.settings.admin_user_ids
        try:
            member = await context.bot.get_chat_member(
                self.settings.admin_chat_id, user_id
            )
        except TelegramError:
            return False
        return member.status in {
            ChatMemberStatus.OWNER,
            ChatMemberStatus.ADMINISTRATOR,
        }

    async def on_admin_decision(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None or query.data is None or update.effective_user is None:
            return
        if query.message is None or query.message.chat.id != self.settings.admin_chat_id:
            await query.answer("Недопустимый чат", show_alert=True)
            return
        if not await self._is_allowed_admin(context, update.effective_user.id):
            await query.answer("Недостаточно прав", show_alert=True)
            return

        try:
            _, action, candidate_id_raw = query.data.split(":", 2)
            candidate_id = int(candidate_id_raw)
        except (ValueError, AttributeError):
            await query.answer("Некорректные данные кнопки", show_alert=True)
            return

        candidate = await self.db.get_candidate(candidate_id)
        if candidate is None:
            await query.answer("Кандидат не найден", show_alert=True)
            return
        async with self._request_lock(candidate.target_chat_id, candidate.user_id):
            candidate = await self.db.get_candidate(candidate_id)
            if candidate is None:
                await query.answer("Кандидат не найден", show_alert=True)
                return
            if candidate.status != CandidateStatus.PENDING_ADMIN.value:
                await query.answer("По этой заявке решение уже принято", show_alert=True)
                return

            try:
                if action == "approve":
                    await context.bot.approve_chat_join_request(
                        chat_id=candidate.target_chat_id, user_id=candidate.user_id
                    )
                    candidate = await self.db.transition_status_if(
                        candidate.id,
                        CandidateStatus.APPROVED,
                        expected_status=CandidateStatus.PENDING_ADMIN,
                        reason="Одобрено администратором",
                        decided_by_user_id=update.effective_user.id,
                    )
                    if candidate is None:
                        logger.error("Approved Telegram but candidate status changed id=%s", candidate_id)
                        await query.answer("Состояние заявки изменилось, проверьте журнал", show_alert=True)
                        return
                    await self._send_candidate_message(
                        context.bot, candidate, texts.APPROVED
                    )
                    verdict = (
                        "✅ Одобрено администратором "
                        + html.escape(update.effective_user.full_name)
                    )
                elif action == "reject":
                    await context.bot.decline_chat_join_request(
                        chat_id=candidate.target_chat_id, user_id=candidate.user_id
                    )
                    candidate = await self.db.transition_status_if(
                        candidate.id,
                        CandidateStatus.REJECTED,
                        expected_status=CandidateStatus.PENDING_ADMIN,
                        reason="Отклонено администратором",
                        decided_by_user_id=update.effective_user.id,
                    )
                    if candidate is None:
                        logger.error("Declined Telegram but candidate status changed id=%s", candidate_id)
                        await query.answer("Состояние заявки изменилось, проверьте журнал", show_alert=True)
                        return
                    await self._send_candidate_message(
                        context.bot, candidate, texts.REJECTED
                    )
                    verdict = (
                        "❌ Отклонено администратором "
                        + html.escape(update.effective_user.full_name)
                    )
                else:
                    await query.answer("Неизвестное действие", show_alert=True)
                    return
            except TelegramError as exc:
                if _join_request_missing(exc):
                    # No live join request exists. Do not record an approval or
                    # rejection, and do not leave dead moderator buttons enabled.
                    await self.db.transition_status_if(
                        candidate.id,
                        CandidateStatus.REQUEST_MISSING,
                        expected_status=CandidateStatus.PENDING_ADMIN,
                        reason=(
                            "Telegram join request absent or already processed "
                            "(Hide_requester_missing); admin action not performed"
                        ),
                    )
                    try:
                        await query.answer(
                            "Заявка отсутствует в Telegram или уже обработана. "
                            "Карточка закрыта без одобрения или отказа.",
                            show_alert=True,
                        )
                    except TelegramError:
                        # An old callback may have expired; still try to remove
                        # obsolete buttons from the moderator card.
                        logger.warning(
                            "Could not answer stale moderator callback candidate=%s",
                            candidate.id,
                        )
                    try:
                        await query.edit_message_reply_markup(reply_markup=None)
                    except TelegramError:
                        logger.warning(
                            "Could not remove stale moderator buttons candidate=%s",
                            candidate.id,
                        )
                    return
                # Unknown Telegram failure: never claim that an action succeeded.
                logger.exception(
                    "Admin decision Telegram API failed candidate=%s action=%s",
                    candidate.id,
                    action,
                )
                await query.answer(
                    "Telegram не выполнил действие. Заявка могла быть отозвана; "
                    "состояние в БД не изменено.",
                    show_alert=True,
                )
                await self._send_admin_audit(
                    context.bot,
                    candidate,
                    f"⚠️ Не удалось выполнить {action}: {type(exc).__name__}. "
                    "Карточка оставлена без изменений.",
                )
                return

            await query.answer("Готово")
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(
                self.settings.admin_chat_id,
                verdict,
                parse_mode=ParseMode.HTML,
            )

    async def _can_view_status(self, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
        """Allow /status only to a *current* target-group owner/admin.

        ADMIN_USER_IDS is an optional additional restriction, not a substitute
        for current Telegram permissions. Fail closed if Telegram cannot verify
        group membership (e.g. network error or removed admin privileges).
        """
        if self.settings.admin_user_ids and user_id not in self.settings.admin_user_ids:
            return False
        try:
            member = await context.bot.get_chat_member(self.settings.target_chat_id, user_id)
        except TelegramError:
            logger.warning("Cannot verify /status permissions for user_id=%s", user_id,
                           exc_info=True)
            return False
        return member.status in {ChatMemberStatus.OWNER, ChatMemberStatus.ADMINISTRATOR}

    async def on_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Short full-period application history on /status in admin DM only.

        Sending candidate answers into a channel is intentionally prohibited:
        Telegram channel posts cannot reliably authenticate their author.
        """
        if update.effective_message is None or update.effective_chat is None:
            return
        if update.effective_chat.type != ChatType.PRIVATE:
            await update.effective_message.reply_text(
                "История заявок доступна только в личном чате с ботом. "
                "Напишите мне /status в ЛС."
            )
            return
        if update.effective_user is None or not await self._can_view_status(
            context, update.effective_user.id
        ):
            await update.effective_message.reply_text(
                "Доступ к истории заявок разрешён только администраторам."
            )
            return

        # /status is the only command: one Telegram text message when possible,
        # otherwise one TXT document containing the complete history.
        args = getattr(context, "args", None) or []
        if args:
            await update.effective_message.reply_text(
                "Использование: /status — краткая история всех заявок за весь период."
            )
            return
        user_id = update.effective_user.id
        if user_id in self._status_running_users:
            await update.effective_message.reply_text(
                "Отчёт /status уже отправляется. Дождитесь завершения предыдущего отчёта."
            )
            return
        self._status_running_users.add(user_id)
        try:
            candidates, cooldowns = await self.db.get_status_history(
                self.settings.target_chat_id
            )
            orphaned = (await self.db.list_unlinked_join_request_events(self.settings.target_chat_id)
                        if isinstance(self.db, Database) else [])
            summary, pages = build_status_report(candidates, cooldowns, orphaned)
            history = "\n\n".join(pages) if pages else "Заявок в базе пока нет."
            report = summary + "\n\n" + history
            # Telegram limits text messages to 4096 characters. Counting UTF-16
            # code units is conservative for non-BMP symbols such as emoji.
            # Never drop candidates or split the response into many messages.
            if len(report.encode("utf-16-le")) // 2 <= _TELEGRAM_TEXT_LIMIT:
                await update.effective_message.reply_text(report)
            else:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=InputFile(
                        BytesIO(report.encode("utf-8")),
                        filename="solidus-status-all.txt",
                    ),
                    caption="SOLIDUS — полный отчёт /status за весь период. "
                            "История не помещается в одно сообщение Telegram.",
                )
        finally:
            self._status_running_users.discard(user_id)

    async def recover_stale_scoring(self, bot: Bot, *, startup: bool = False) -> int:
        """Fail closed after a crash/timeout: manual moderation, NEVER auto-reject.

        At startup no scoring coroutine from this process is alive, so recover
        all unfinished rows immediately. During normal operation only recover
        old rows not currently being handled by this process.
        """
        now = datetime.now(timezone.utc)
        # Enough time for two sequential LLM calls plus DB/Telegram overhead.
        grace_minutes = max(10, math.ceil(3 * self.settings.llm_timeout_seconds / 60) + 2)
        cutoff = now + timedelta(seconds=1) if startup else now - timedelta(minutes=grace_minutes)
        stale = await self.db.list_stale_scoring(cutoff)
        count = 0
        for item in stale:
            if item.id in self._active_scoring_ids:
                continue
            recovered = await self.db.recover_scoring_to_admin(item.id, cutoff)
            if recovered is not None:
                count += 1
                logger.warning(
                    "Scoring recovery candidate=%s mode=%s previous_status=%s "
                    "new_status=pending_admin reason=interrupted",
                    item.id, "startup" if startup else "stale", item.status,
                )
        # Retry unsent recovery cards persistently, including if Telegram was
        # unavailable on the first attempt or the process crashed after CAS.
        for item in await self.db.list_pending_recovery_notices():
            try:
                await self._send_admin_card(
                    bot, item, score=None,
                    scoring_error=(
                        "AI-проверка прервалась или не завершилась. "
                        "Восстановление не выполняет автоотказ. "
                        "Проверьте, осталась ли заявка в Telegram: "
                        "действие до сбоя могло уже произойти."
                    ),
                    with_buttons=True,
                )
            except TelegramError:
                logger.warning(
                    "Scoring recovery admin notice failed candidate=%s; retry later",
                    item.id,
                )
                continue
            except Exception:
                logger.exception("Unexpected recovery notice error candidate=%s", item.id)
                continue
            if await self.db.mark_recovery_notice_sent(item.id):
                logger.info("Scoring recovery admin notified candidate=%s", item.id)
                await self._send_candidate_message(bot, item, texts.PENDING_ADMIN)
        return count

    async def recover_cooldown_decline_intents(self, bot: Bot) -> int:
        """Notify admins of ambiguous/crashed cooldown declines.

        We intentionally do not retry the Telegram side effect: without a
        per-attempt Telegram identifier it might decline a newer application.
        Replay markers survive until an operator reviews the situation.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        changed = await self.db.mark_stale_cooldown_intents(cutoff)
        if changed:
            logger.warning("Recovered %s stale cooldown intents as uncertain", changed)
        notices = await self.db.list_unalerted_cooldown_intents()
        for item in notices:
            text = (
                "⚠️ Неопределённый результат отклонения заявки в cooldown.\n"
                f"Чат: {item.target_chat_id}, пользователь: {item.user_id}, "
                f"дата заявки (UTC): {item.request_date}.\n"
                "Бот НЕ повторяет отклонение автоматически: Telegram не "
                "предоставляет ID отдельной заявки. Проверьте вручную "
                "текущую заявку пользователя и действия бота."
            )
            try:
                await bot.send_message(chat_id=self.settings.admin_chat_id, text=text)
            except TelegramError:
                logger.exception("Failed to alert admin about cooldown intent=%s", item.id)
                # Leave marker unalerted; maintenance can try again.
                continue
            await self.db.mark_cooldown_intent_alerted(item.id)
        return changed

    async def recover_and_alert_join_requests(self, bot: Bot) -> int:
        """Reconcile interrupted join events and retry durable admin alerts.

        At-least-once alerts: a crash between Telegram success and DB ack may
        duplicate a moderation card, never an approve/decline or first question.
        """
        if not isinstance(self.db, Database):
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=90)
        recovered = await self.db.recover_unconfirmed_join_requests(
            cutoff, excluding_update_ids=self._inflight_join_update_ids,
        )
        pending = await self.db.list_pending_join_request_alerts()
        for event in pending:
            try:
                candidate = (await self.db.get_candidate(event.candidate_id)
                             if event.candidate_id is not None else None)
                if candidate is not None:
                    await self._send_admin_card(
                        bot, candidate, score=None,
                        scoring_error=(
                            "⚠️ Технический сбой обработки заявки: "
                            f"{event.error_type or event.status}. "
                            "Доставка первого вопроса не подтверждена; "
                            "не отклонять автоматически."
                        ),
                        with_buttons=(candidate.status == CandidateStatus.PENDING_ADMIN.value),
                    )
                else:
                    await bot.send_message(
                        chat_id=self.settings.admin_chat_id,
                        text=(
                            "⚠️ Telegram-заявка получена, но анкета не зарегистрирована.\n"
                            f"TG ID: {event.user_id}\n"
                            f"Обновление: {event.update_id}\n"
                            f"Этап: {event.status}; ошибка: {event.error_type or 'неизвестна'}.\n"
                            "Нужна ручная проверка администратором."
                        ),
                    )
            except Exception:
                logger.exception("Failed to alert admin about join update=%s", event.update_id)
                continue
            await self.db.mark_join_request_alerted(event.update_id)
            logger.info("join_admin_alert_sent update_id=%s", event.update_id)
        return recovered

    async def expire_stale_interviews(self, bot: Bot) -> int:
        """Close stale questionnaires without acting on a superseded DB row.

        Both the re-check and Telegram decline are protected by the same
        per-user lock used by new join events and moderator decisions.
        The database CAS prevents replacing a status changed by another worker.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self.settings.interview_timeout_minutes
        )
        candidates = await self.db.list_expired_interviews(cutoff)
        processed = 0
        for item in candidates:
            async with self._request_lock(item.target_chat_id, item.user_id):
                candidate = await self.db.get_candidate(item.id)
                if candidate is None or candidate.status != CandidateStatus.INTERVIEW.value:
                    continue
                updated_at = candidate.updated_at
                if updated_at is None:
                    continue
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                if candidate.stage > 3 or updated_at >= cutoff:
                    continue
                active = await self.db.get_active_candidate_for_chat(
                    candidate.user_id, candidate.target_chat_id
                )
                if active is None or active.id != candidate.id:
                    logger.info("Skipping superseded expiry candidate=%s", item.id)
                    continue
                try:
                    await bot.decline_chat_join_request(
                        chat_id=candidate.target_chat_id, user_id=candidate.user_id
                    )
                except TelegramError as exc:
                    if _join_request_missing(exc):
                        logger.info("Expired candidate=%s: Hide_requester_missing", candidate.id)
                        updated = await self.db.transition_status_if(
                            candidate.id, CandidateStatus.EXPIRED,
                            expected_status=CandidateStatus.INTERVIEW,
                            expected_stage=candidate.stage, older_than=cutoff,
                            reason=(
                                "Анкета просрочена; Telegram-заявка отсутствует или "
                                "уже обработана (Hide_requester_missing). "
                                "Отклонение не выполнялось."
                            ),
                        )
                        if updated is not None:
                            processed += 1
                        continue
                    logger.exception("Failed to expire stale interview candidate=%s", candidate.id)
                    updated = await self.db.transition_status_if(
                        candidate.id, CandidateStatus.PENDING_ADMIN,
                        expected_status=CandidateStatus.INTERVIEW,
                        expected_stage=candidate.stage, older_than=cutoff,
                        reason=(
                            "Interview timeout, Telegram decline failed: "
                            f"{type(exc).__name__}"
                        ),
                    )
                    if updated is not None:
                        await self._send_admin_card(
                            bot, updated, score=None,
                            scoring_error=(
                                "Анкета просрочена, но Telegram не подтвердил "
                                "автоматическое отклонение. Требуется решение администратора."
                            ),
                        )
                    continue

                updated = await self.db.transition_status_if(
                    candidate.id, CandidateStatus.EXPIRED,
                    expected_status=CandidateStatus.INTERVIEW,
                    expected_stage=candidate.stage, older_than=cutoff,
                    reason="Анкета не завершена до истечения таймаута",
                )
                if updated is not None:
                    await self._send_candidate_message(bot, updated, texts.TIMEOUT)
                    processed += 1
        return processed

    async def _send_candidate_message(
        self, bot: Bot, candidate: Candidate, text: str
    ) -> bool:
        try:
            await bot.send_message(candidate.user_chat_id, text)
            return True
        except TelegramError:
            logger.exception(
                "Failed to notify candidate=%s chat_id=%s",
                candidate.id,
                candidate.user_chat_id,
            )
            return False

    async def _send_admin_audit(
        self,
        bot: Bot,
        candidate: Candidate,
        note: str,
    ) -> None:
        await bot.send_message(
            self.settings.admin_chat_id,
            f"{note}\nКандидат: {self._candidate_name(candidate)}\nID: {candidate.user_id}",
        )

    def _build_admin_card(
        self,
        candidate: Candidate,
        *,
        score: ScoreResult | None,
        scoring_error: str | None = None,
        status_note: str | None = None,
    ) -> str:
        def render(answer_limit: int, summary_limit: int, flag_limit: int) -> str:
            username = f"@{candidate.username}" if candidate.username else "—"
            lines = [
                "<b>Новая заявка Solidus</b>",
                f"Кандидат: {_escape_clip(self._candidate_name(candidate), 140)}",
                f"Username: {_escape_clip(username, 140)}",
                f"Telegram ID: <code>{candidate.user_id}</code>",
                "",
                "<b>1. Локация</b>",
                _escape_clip(candidate.location_answer or "—", answer_limit),
                "",
                "<b>2. Бизнес-профиль</b>",
                _escape_clip(candidate.business_answer or "—", answer_limit),
                "",
                "<b>3. Цель вступления</b>",
                _escape_clip(candidate.goal_answer or "—", answer_limit),
                "",
            ]

            if score is not None:
                lines.extend(
                    [
                        (
                            f"<b>Confidence Score (без неподтверждённого спам-штрафа):</b> "
                            f"{score.confidence_score}/100"
                            if score.spam_penalty_withheld else
                            f"<b>Confidence Score:</b> {score.confidence_score}/100"
                        ),
                        (
                            f"Локация: {score.location_score}/100 · "
                            f"Бизнес: {score.business_score}/100"
                        ),
                        (
                            f"Цель: {score.intent_score}/100 · "
                            f"Спам-риск: {score.spam_score}/100"
                        ),
                        f"Уверенность оценки: {score.ai_confidence}/100",
                        "",
                        "<b>AI-саммари</b>",
                        _escape_clip(score.summary, summary_limit),
                    ]
                )
                if score.spam_penalty_withheld:
                    lines.extend([
                        "",
                        f"<b>Исходный Score со спам-штрафом:</b> "
                        f"{score.preliminary_confidence_score}/100",
                        "⚠️ Высокий спам-риск модели не подтверждён для автоотказа. "
                        "Оценка без штрафа предварительная; требуется решение модератора.",
                    ])
                if score.core_verifications:
                    lines.append("")
                    for review in score.core_verifications:
                        name = "Локация" if review.field == "location" else "Бизнес"
                        label = ("подтверждено" if review.permits_auto_reject
                                 else "НЕ подтверждено для автоотказа")
                        lines.append(
                            f"<b>Проверка {name}:</b> {label} — "
                            + _escape_clip(review.reason, 145)
                        )
                elif (
                    score.location_score <= getattr(self.settings, "auto_reject_core_score_max", 15)
                    or score.business_score <= getattr(self.settings, "auto_reject_core_score_max", 15)
                ):
                    lines.extend(["", "⚠️ Критически низкий балл: нет подтверждения для автоотказа"])

                # Render confidence sections in a fixed order, instead of inserting
                # a line relative to the list tail (which changes with optional blocks).
                if score.confidence_details is not None:
                    details = score.confidence_details
                    lines.extend([
                        "",
                        "<b>Основания уверенности</b>",
                        f"Полнота: {details.completeness} · Цитаты: {details.specificity}",
                        f"Согласованность: {details.consistency} · "
                        f"Определённость: {details.certainty}",
                    ])
                    # A weak global rubric does not prohibit assessment of a
                    # specific, independently verified ground.
                    if details.notes and not any(
                        grade.evidence_verified for grade in score.decision_confidences
                    ):
                        lines.append("⚠️ " + _escape_clip(details.notes[-1], 140))

                if score.decision_confidences:
                    names = {"location": "локации", "business": "бизнесу",
                             "spam": "спаму"}
                    for grade in score.decision_confidences[:3]:
                        lines.extend([
                            "",
                            f"<b>Уверенность по {names.get(grade.criterion, grade.criterion)}:</b> "
                            f"{grade.score}/100",
                            f"Полнота: {grade.completeness} · Цитата: {grade.specificity}",
                            f"Согласованность: {grade.consistency} · "
                            f"Определённость: {grade.certainty}",
                        ])
                if score.spam_score >= getattr(self.settings, "auto_reject_spam_score", 85):
                    verdict = score.spam_verification
                    if verdict is None:
                        spam_note = "Подтверждение спама не проводилось (низкая уверенность)"
                    elif verdict.permits_auto_reject:
                        spam_note = "Повторная проверка подтвердила явные признаки спама"
                    else:
                        spam_note = (
                            "Спам не подтверждён для автоотказа: " + verdict.reason
                        )
                    lines.extend(["", "<b>Проверка спам-риска:</b>",
                                  _escape_clip(spam_note, 280)])
                if score.risk_flags:
                    lines.extend(
                        [
                            "",
                            "<b>Флаги риска</b>",
                            "\n".join(
                                f"• {_escape_clip(flag, flag_limit)}"
                                for flag in score.risk_flags[:4]
                            ),
                        ]
                    )
            elif scoring_error:
                lines.extend(
                    ["<b>AI:</b> " + _escape_clip(scoring_error, 320)]
                )

            if status_note:
                lines.extend(["", _escape_clip(status_note, 320)])
            return "\n".join(lines)

        card = render(answer_limit=560, summary_limit=520, flag_limit=90)
        if len(card) <= _ADMIN_CARD_SAFE_LIMIT:
            return card

        # Defensive compact rendering for pathological names/answers/HTML entities.
        card = render(answer_limit=300, summary_limit=320, flag_limit=60)
        if len(card) <= _ADMIN_CARD_SAFE_LIMIT:
            return card

        # Minimal rendering avoids ever cutting through an HTML tag/entity.
        minimal = [
            "<b>Новая заявка Solidus</b>",
            f"Кандидат: {_escape_clip(self._candidate_name(candidate), 80)}",
            f"Telegram ID: <code>{candidate.user_id}</code>",
        ]
        if score is not None:
            minimal.extend(
                [
                    f"<b>Confidence Score{' (без неподтверждённого спам-штрафа)' if score.spam_penalty_withheld else ''}:</b> {score.confidence_score}/100",
                    f"Спам-риск: {score.spam_score}/100 · Уверенность оценки: {score.ai_confidence}/100",
                    *([f"Исходный Score: {score.preliminary_confidence_score}/100"]
                      if score.spam_penalty_withheld else []),
                    "",
                    "<b>AI-саммари</b>",
                    _escape_clip(score.summary, 180),
                ]
            )
        elif scoring_error:
            minimal.append("<b>AI:</b> " + _escape_clip(scoring_error, 180))
        if status_note:
            minimal.append(_escape_clip(status_note, 180))
        return "\n".join(minimal)

    async def _send_admin_card(
        self,
        bot: Bot,
        candidate: Candidate,
        *,
        score: ScoreResult | None,
        scoring_error: str | None = None,
        status_note: str | None = None,
        with_buttons: bool = True,
    ) -> None:
        card = self._build_admin_card(
            candidate,
            score=score,
            scoring_error=scoring_error,
            status_note=status_note,
        )
        if len(card) > _TELEGRAM_TEXT_LIMIT:
            raise RuntimeError("Admin card renderer exceeded Telegram text limit")

        try:
            await bot.send_message(
                chat_id=self.settings.admin_chat_id,
                text=card,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=moderation_keyboard(candidate.id) if with_buttons else None,
            )
        except TelegramError:
            logger.exception("Failed to send full admin card candidate=%s", candidate.id)
            # Last-resort plain-text card keeps moderation operable even if Telegram
            # rejects HTML for an unforeseen edge case.
            fallback = (
                "Новая заявка Solidus\n"
                f"Кандидат: {self._candidate_name(candidate)}\n"
                f"Telegram ID: {candidate.user_id}\n"
                f"Confidence Score: {score.confidence_score if score else 'нет'}"
                f"{' (без неподтверждённого спам-штрафа)' if score and score.spam_penalty_withheld else ''}\n"
                f"{'Исходный Score со штрафом: ' + str(score.preliminary_confidence_score) + chr(10) if score and score.spam_penalty_withheld else ''}"
                "Подробная карточка не отправилась из-за ошибки форматирования."
            )
            await bot.send_message(
                chat_id=self.settings.admin_chat_id,
                text=fallback[:_ADMIN_CARD_SAFE_LIMIT],
                reply_markup=moderation_keyboard(candidate.id) if with_buttons else None,
            )

    @staticmethod
    def _question_for_stage(stage: int) -> str:
        if stage <= 1:
            return f"{texts.INTRO}\n\n{texts.QUESTION_LOCATION}"
        if stage == 2:
            return texts.QUESTION_BUSINESS
        if stage == 3:
            return texts.QUESTION_GOAL
        return texts.PENDING_ADMIN

    @staticmethod
    def _candidate_name(candidate: Candidate) -> str:
        if candidate.first_name:
            return candidate.first_name
        if candidate.username:
            return candidate.username
        return str(candidate.user_id)
