from __future__ import annotations

import html
import logging
import json
import re
import math
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

from telegram import Bot, Update
from telegram.constants import ChatMemberStatus, ChatType, ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from . import texts
from .config import Settings
from .db import Database
from .keyboards import moderation_keyboard
from .models import Candidate, CandidateStatus
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

    async def on_join_request(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        req = update.chat_join_request
        if req is None or req.chat.id != self.settings.target_chat_id:
            return

        # Telegram may redeliver updates or a user may rapidly re-apply. Reuse an
        # already active flow instead of creating duplicate interviews/LLM calls.
        active = await self.db.get_active_candidate_for_chat(
            req.from_user.id, req.chat.id
        )
        if active is not None:
            active = await self.db.set_user_chat_id(active.id, req.user_chat_id)
            if active.status == CandidateStatus.INTERVIEW.value:
                await self._send_candidate_message(
                    context.bot, active, self._question_for_stage(active.stage)
                )
            elif active.status == CandidateStatus.SCORING.value:
                await self._send_candidate_message(context.bot, active, texts.PROCESSING)
            else:
                await self._send_candidate_message(context.bot, active, texts.PENDING_ADMIN)
            return

        if self.settings.reapplication_cooldown_minutes > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(
                minutes=self.settings.reapplication_cooldown_minutes
            )
            recent = await self.db.get_recent_rejected_candidate(
                req.from_user.id, req.chat.id, cutoff
            )
            if recent is not None:
                try:
                    await context.bot.decline_chat_join_request(
                        chat_id=req.chat.id, user_id=req.from_user.id
                    )
                except TelegramError:
                    logger.exception(
                        "Failed to decline cooldown re-application user=%s",
                        req.from_user.id,
                    )
                    return
                # Use the fresh temporary chat id supplied by this join request.
                try:
                    await context.bot.send_message(req.user_chat_id, texts.COOLDOWN)
                except TelegramError:
                    logger.info(
                        "Unable to notify cooldown applicant user=%s", req.from_user.id
                    )
                return

        candidate = await self.db.create_candidate(
            user_id=req.from_user.id,
            user_chat_id=req.user_chat_id,
            target_chat_id=req.chat.id,
            username=req.from_user.username,
            first_name=req.from_user.first_name,
        )

        try:
            await context.bot.send_message(
                chat_id=req.user_chat_id,
                text=f"{texts.INTRO}\n\n{texts.QUESTION_LOCATION}",
            )
        except TelegramError as exc:
            # A transport/API error is not evidence that the candidate is spam or
            # unsuitable. Do not auto-decline on a technical failure.
            logger.exception("Failed to start interview for candidate=%s", candidate.id)
            candidate = await self.db.set_status(
                candidate.id,
                CandidateStatus.PENDING_ADMIN,
                reason=f"Не удалось начать ЛС: {type(exc).__name__}",
            )
            await self._send_admin_card(
                context.bot,
                candidate,
                score=None,
                scoring_error=(
                    "Не удалось начать личный диалог с кандидатом. "
                    "Автоматический отказ НЕ выполнен; требуется решение администратора."
                ),
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

        candidate = await self.db.get_active_interview(update.effective_user.id)
        if candidate is None:
            await self.support.on_candidate_message(update, context)
            return

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
        await self._score_and_route(context.bot, candidate)

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
        except Exception as exc:
            logger.exception("AI scoring/routing failed for candidate=%s", candidate.id)
            candidate = await self.db.set_status(
                candidate.id, CandidateStatus.PENDING_ADMIN,
                reason=f"AI scoring error: {type(exc).__name__}",
            )
            await self._send_admin_card(
                bot, candidate, score=None,
                scoring_error="AI-скоринг временно недоступен; требуется решение администратора.",
            )
            await self._send_candidate_message(bot, candidate, texts.PENDING_ADMIN)
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
                )
                await self._send_admin_card(
                    bot, candidate, score=score,
                    status_note=("⚠️ Decision Gate разрешил автоотказ, но Telegram "
                                 "не подтвердил отклонение. Решает администратор."),
                    with_buttons=True,
                )
                await self._send_candidate_message(bot, candidate, texts.TECHNICAL_REVIEW)
                return
            candidate = await self.db.set_status(
                candidate.id, CandidateStatus.AUTO_REJECTED, reason=reason
            )
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
            candidate.id, CandidateStatus.PENDING_ADMIN
        )
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
        if candidate.status != CandidateStatus.PENDING_ADMIN.value:
            await query.answer("По этой заявке решение уже принято", show_alert=True)
            return

        try:
            if action == "approve":
                await context.bot.approve_chat_join_request(
                    chat_id=candidate.target_chat_id, user_id=candidate.user_id
                )
                candidate = await self.db.set_status(
                    candidate.id,
                    CandidateStatus.APPROVED,
                    reason="Одобрено администратором",
                    decided_by_user_id=update.effective_user.id,
                )
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
                candidate = await self.db.set_status(
                    candidate.id,
                    CandidateStatus.REJECTED,
                    reason="Отклонено администратором",
                    decided_by_user_id=update.effective_user.id,
                )
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
            # Do not mutate DB status when Telegram did not confirm the action.
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

    async def on_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (
            update.effective_chat is None
            or update.effective_chat.id != self.settings.admin_chat_id
            or update.effective_user is None
        ):
            return
        if not await self._is_allowed_admin(context, update.effective_user.id):
            return
        # Translate status labels only for Telegram; database values stay unchanged.
        status_labels = {
            "interview": "Проходит анкетирование",
            "scoring": "Проходит AI-проверку",
            "pending_admin": "Ожидает решения администратора",
            "approved": "Одобрено",
            "rejected": "Отклонено администратором",
            "auto_rejected": "Отклонено автоматически",
            "expired": "Срок анкеты истёк",
            "superseded": "Заменено новой заявкой",
            "error": "Ошибка обработки",
        }
        counts = await self.db.count_by_status()
        if not counts:
            text = "Заявок пока нет."
        else:
            text = "Статусы заявок:\n" + "\n".join(
                f"• {status_labels.get(key, key)}: {value}"
                for key, value in sorted(counts.items())
            )
        await update.effective_message.reply_text(text)

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

    async def expire_stale_interviews(self, bot: Bot) -> int:
        """Close abandoned questionnaires so pending join requests do not hang forever."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self.settings.interview_timeout_minutes
        )
        candidates = await self.db.list_expired_interviews(cutoff)
        processed = 0
        for candidate in candidates:
            try:
                await bot.decline_chat_join_request(
                    chat_id=candidate.target_chat_id, user_id=candidate.user_id
                )
            except TelegramError as exc:
                logger.exception(
                    "Failed to expire stale interview candidate=%s", candidate.id
                )
                candidate = await self.db.set_status(
                    candidate.id,
                    CandidateStatus.PENDING_ADMIN,
                    reason=f"Interview timeout, Telegram decline failed: {type(exc).__name__}",
                )
                await self._send_admin_card(
                    bot,
                    candidate,
                    score=None,
                    scoring_error=(
                        "Анкета просрочена, но Telegram не подтвердил автоматическое "
                        "отклонение. Требуется решение администратора."
                    ),
                )
                continue

            candidate = await self.db.set_status(
                candidate.id,
                CandidateStatus.EXPIRED,
                reason="Анкета не завершена до истечения таймаута",
            )
            await self._send_candidate_message(bot, candidate, texts.TIMEOUT)
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
                            "Локация: "
                            f"{score.location_score}/100 · Бизнес: {score.business_score}/100 · "
                            f"Цель: {score.intent_score}/100 · Спам-риск: {score.spam_score}/100"
                        ),
                        f"Уверенность оценки: {score.ai_confidence}/100",
                        "",
                        "<b>AI-саммари</b>",
                        _escape_clip(score.summary, summary_limit),
                    ]
                )
                if score.spam_penalty_withheld:
                    lines.extend([
                        f"<b>Исходный Score со спам-штрафом:</b> "
                        f"{score.preliminary_confidence_score}/100",
                        "⚠️ Высокий спам-риск модели не подтверждён для автоотказа. "
                        "Оценка без штрафа предварительная; требуется решение модератора.",
                    ])
                if score.core_verifications:
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
                    lines.append("⚠️ Критически низкий балл: нет подтверждения для автоотказа")
                if score.decision_confidences:
                    names = {"location": "локации", "business": "бизнесу",
                             "spam": "спаму"}
                    for grade in score.decision_confidences[:3]:
                        lines.append(
                            f"<b>Уверенность по {names.get(grade.criterion, grade.criterion)}:</b> "
                            f"{grade.score}/100 · "
                            f"полнота {grade.completeness} · цитата {grade.specificity} · "
                            f"согласованность {grade.consistency} · "
                            f"определённость {grade.certainty}"
                        )
                if score.confidence_details is not None:
                    details = score.confidence_details
                    lines.insert(
                        -3,
                        "<b>Основания уверенности:</b> "
                        f"полнота {details.completeness} · "
                        f"цитаты {details.specificity} · "
                        f"согласованность {details.consistency} · "
                        f"определённость {details.certainty}",
                    )
                    # The global rubric can be incomplete while a specific
                    # reason has grounded evidence. Do not mislabel this as
                    # a ban on stage-3 assessment of the specific ground.
                    if details.notes and not any(
                        grade.evidence_verified for grade in score.decision_confidences
                    ):
                        lines.extend([
                            "",
                            "⚠️ " + _escape_clip(details.notes[-1], 140),
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
