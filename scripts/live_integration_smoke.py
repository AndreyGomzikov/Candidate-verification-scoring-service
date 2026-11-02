"""Opt-in live verification with a REAL LLM and a TEST Telegram group.

No join requests are approved or declined. Production target-chat settings are
never read: provide a separate LIVE_TEST_CHAT_ID to avoid touching live groups.

    LIVE_TEST_CHAT_CONFIRMED=yes LIVE_TEST_CHAT_ID=-100... \\
      BOT_TOKEN=... LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=... \\
      python scripts/live_integration_smoke.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Make `app` importable when invoked as a script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.decision import decide_auto_rejection
from app.scoring import AIScorer, propose_auto_rejection


async def main() -> int:
    needed = ("BOT_TOKEN", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LIVE_TEST_CHAT_ID")
    absent = [key for key in needed if not os.getenv(key)]
    if absent:
        print("SKIPPED: Missing test configuration:", ", ".join(absent))
        return 2
    if os.getenv("LIVE_TEST_CHAT_CONFIRMED") != "yes":
        print("SKIPPED: Set LIVE_TEST_CHAT_CONFIRMED=yes only for a disposable TEST chat")
        return 2
    chat_id = os.environ["LIVE_TEST_CHAT_ID"]
    if not chat_id.startswith("-100") or not chat_id[4:].isdigit():
        print("SKIPPED: LIVE_TEST_CHAT_ID must be a Telegram supergroup numeric ID")
        return 2
    if chat_id == os.getenv("TARGET_CHAT_ID"):
        print("SKIPPED: LIVE_TEST_CHAT_ID must NOT equal TARGET_CHAT_ID")
        return 2

    token = os.environ["BOT_TOKEN"]
    async with httpx.AsyncClient(timeout=20) as client:
        # Both Telegram methods are read-only. Never submit an actual decline
        # here: an explicit disposable join request requires a real test user.
        for method, body in (("getMe", {}), ("getChat", {"chat_id": chat_id})):
            response = await client.post(f"https://api.telegram.org/bot{token}/{method}",
                                         json=body)
            response.raise_for_status()
            if response.json().get("ok") is not True:
                print("FAIL: Telegram read-only call failed:", method)
                return 1
            print("PASS: Telegram test connection:", method)

    scorer = AIScorer(
        base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"], timeout_seconds=35,
    )
    cases = (
        ("explicit_mismatch", (
            "Я сейчас не в Грузии и не планирую приезжать ближайшие годы",
            "Не знаю", "Нет дополнительных сведений",
        )),
        ("travel_in_goal", (
            "Я сейчас не в Грузии и не планирую приезжать ближайшие годы",
            "Не знаю", "В декабре у меня конференция в Тбилиси, уже зарегистрировался",
        )),
        ("professional_in_goal", (
            "Не знаю", "У меня нет бизнеса, нет проектов, нигде не работаю",
            "Занимаюсь репетиторством и даю частные уроки",
        )),
        ("travel_in_location", (
            "Я сейчас не в Грузии и не планирую приезжать ближайшие годы. "
            "Только что прилетел в Батуми", "Не знаю", "Нет дополнительных сведений",
        )),
        ("profession_in_business", (
            "Не знаю", "У меня нет бизнеса, нет проектов, нигде не работаю. "
            "Я фотограф, снимаю свадьбы по заказу", "Нет дополнительных сведений",
        )),
    )
    for name, answers in cases:
        result, _raw = await scorer.score(location=answers[0], business=answers[1],
                                         goal=answers[2])
        proposal = propose_auto_rejection(
            result, spam_threshold=85, core_score_max=15,
            min_ai_confidence=80, candidate_answers=answers,
        )
        review = None
        if proposal:
            review = await scorer.review_rejection(
                criterion=proposal.criterion, spam_type=proposal.spam_type,
                location=answers[0], business=answers[1], goal=answers[2],
            )
        decision = decide_auto_rejection(
            proposal=proposal, review=review, score=result,
            candidate_answers=answers, spam_threshold=85,
            core_score_max=15, min_ai_confidence=80,
        )
        print(f"LLM {name}: decision={decision.code}; auto_reject={decision.allow_auto_reject}")
        if name != "explicit_mismatch" and decision.allow_auto_reject:
            print("FAIL: The semantic contradiction must never be automatically rejected")
            return 1
    print("PASS: Live test finished. No Telegram membership was changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        # Do not log exception objects from HTTP requests; they can include
        # authenticated URLs with bot token or sensitive response fragments.
        print("FAIL: External service error:", type(exc).__name__)
        raise SystemExit(1)
