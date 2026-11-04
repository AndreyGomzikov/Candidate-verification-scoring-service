"""Conservative, deterministic full-answer safety rules (ported from the prior Decision Gate).

These rules may only veto automated rejection; unfamiliar language must go to
human moderation. They never assert that candidate facts were verified.
"""
from __future__ import annotations
import re
import unicodedata
from collections.abc import Sequence

MIN_EVIDENCE_LENGTH = 20
MIN_EVIDENCE_WORDS = 3
MIN_CORE_EVIDENCE_LENGTH = 25
_QUOTE_CHARS = "\"'«»“”„‘’`"

def _normalize_for_match(text: str) -> str:
    text = str(text or "").casefold().replace("ё", "е")
    text = text.translate({ord(ch): " " for ch in _QUOTE_CHARS})
    text = text.replace("…", " ").replace("...", " ")
    return " ".join(text.split())


def evidence_found_in_answers(evidence: str, answers: Sequence[str]) -> bool:
    """Check that the complete quote (or each ellipsis fragment) is verbatim.

    Fragments must be non-trivial and occur within *one* answer, not across
    two questions. Matching is case/spacing/ё insensitive but word-boundary
    aware, so a short substring of an innocent word cannot count as evidence.
    """
    if not evidence or not answers or not isinstance(evidence, str):
        return False
    fragments = [
        _normalize_for_match(part)
        for part in re.split(r"…|\.\.\.", str(evidence))
    ]
    if not fragments or any(not fragment for fragment in fragments):
        return False
    if (sum(len(fragment) for fragment in fragments) < MIN_EVIDENCE_LENGTH
            or sum(len(re.findall(r"\w+", fragment)) for fragment in fragments)
            < MIN_EVIDENCE_WORDS):
        return False
    # Do not treat "бота" as evidence from an entirely harmless test reply.
    for answer in answers:
        normalized = _normalize_for_match(answer)
        if all(re.search(r"(?<!\w)" + re.escape(f) + r"(?!\w)", normalized)
               for f in fragments):
            return True
    return False


def _words(text: str) -> str:
    """Plain tokenized text for conservative rule checks (never AI scores)."""
    return " ".join(re.findall(r"[\w]+", str(text or "").casefold().replace("ё", "е")))


def _has_valid_core_quote(evidence: str, answer: str) -> bool:
    # Core mismatch must be a single, sufficiently long, literal passage from
    # the *specific* answer. Disallow fragmented quotes, which lose context.
    return (
        len(_words(evidence)) >= MIN_CORE_EVIDENCE_LENGTH
        and "…" not in evidence and "..." not in evidence
        and evidence_found_in_answers(evidence, (answer,))
    )



# For irreversible core auto-rejections, mere absence of a recognized
# contradiction is NOT evidence that an entire free-form answer is negative.
# This is an intentionally bounded grammar of COMPLETE, unambiguous statements.
# Everything else (including extra sentences, new professions, new cities,
# qualifications, quotations or unknown wording) goes to a human, not an ever
# growing keyword classifier.  Extend only with reviewed tests of both a
# positive case and its contradictory near-neighbors.
_FULL_LOCATION_NEGATIVE = re.compile(
    r"(?:я (?:сейчас )?)?не в грузии (?:и )?"
    r"(?:не планирую приезжать|не собираюсь приезжать|"
    r"приезжать не собираюсь)"
    r"(?: (?:ближайшие годы|в ближайшие годы|пока и не хочу))?"
)
_FULL_BUSINESS_NEGATIVE = re.compile(
    r"(?:у меня )?нет бизнеса нет проектов (?:я )?"
    r"(?:нигде не работаю|не работаю)"
)


def _fully_verified_core_answer(criterion: str, evidence: str, answer: str) -> bool:
    """Allow only a complete, *declarative* negative answer; otherwise abstain.

    A punctuation-stripped token match is NOT sufficient: "Я не в Грузии?"
    is a question, not a verified negative statement. Likewise, a quoted
    sentence or multiple clauses separated by dots cannot serve as independent
    proof of a core mismatch even when both LLM passes agree.
    """
    if not isinstance(answer, str) or not isinstance(evidence, str):
        return False
    # Preserve punctuation when deciding whether the candidate actually made
    # an unambiguous assertion. NFKC catches compatibility question marks
    # such as FULLWIDTH QUESTION MARK, rather than quietly deleting them.
    # Only commas within one sentence and a single optional terminal period
    # are permitted. Every other sign (question/exclamation marks, ellipses,
    # quotes, semicolons, dashes, colons, embedded periods) fails closed.
    raw = unicodedata.normalize("NFKC", answer.strip())
    if not re.fullmatch(r"\w+(?:[ \t]+\w+|[ \t]*,[ \t]+\w+)*\.?", raw):
        return False
    if not _has_valid_core_quote(evidence, answer):
        return False
    normalized = _words(answer)
    if normalized != _words(evidence):
        return False
    patterns = {"location": _FULL_LOCATION_NEGATIVE, "business": _FULL_BUSINESS_NEGATIVE}
    pattern = patterns.get(criterion)
    return bool(pattern is not None and pattern.fullmatch(normalized))


# Extra conservative signals that a candidate could qualify even if an AI
# selected a literal negative quote elsewhere in the questionnaire. These
# checks can only PREVENT an automatic rejection; they never approve anyone.
# They intentionally inspect each answer independently, without concatenating
# field boundaries into a synthetic sentence.
_GEORGIA_DESTINATION = (
    r"(?:грузию|грузии|тбилиси|батуми|кутаиси|рустави|поти|гори)"
)


def _mentions_georgia_visit_or_residence(answer: str) -> bool:
    text = _words(answer)
    destination = _GEORGIA_DESTINATION
    # Examples: "Завтра лечу в Тбилиси", "Еду в Грузию". Avoid matching
    # explicit negations like "не лечу в Тбилиси" as positive travel plans.
    travelling = re.search(
        rf"(?<!не )\b(?:лечу|летим|еду|едем|едут|вылетаю|"
        rf"вылетаем|отправляюсь|отправляемся|направляюсь|"
        rf"выезжаю|выезжаем|путешествую|съезжу|посещу|"
        rf"полечу|собираюсь лететь) (?:[\w]+ ){{0,3}}?(?:в|до) {destination}\b",
        text,
    )
    # Examples: "Со следующей недели буду жить в Тбилиси", "Буду
    # работать в Батуми". The old rule only recognized "буду в ...".
    future_residence = re.search(
        rf"(?<!не )\b(?:буду|планирую|собираюсь|хочу) "
        rf"(?:[\w]+ ){{0,3}}?(?:жить|проживать|работать|"
        rf"находиться|пожить|остаться) в {destination}\b",
        text,
    )
    # Examples: "Через 2 месяца запланирован отпуск в Грузии", "У меня
    # поездка в Тбилиси запланирована". A trip/holiday/visit is enough to
    # make a supposedly definitive 'I won't visit' answer ambiguous.
    trip = r"(?:отпуск\w*|поездк\w*|визит\w*|командировк\w*|путешестви\w*|перелет\w*|рейс\w*)"
    planned_trip = re.search(
        rf"\b(?:запланирован\w*|намечен\w*|назначен\w*|"
        rf"предстоит|планирую|скоро|будет) "
        rf"(?:[\w]+ ){{0,4}}?{trip} (?:[\w]+ ){{0,4}}?(?:в|до) {destination}\b",
        text,
    ) or re.search(
        rf"\b{trip} (?:[\w]+ ){{0,3}}?(?:в|до) {destination} "
        rf"(?:[\w]+ ){{0,3}}?(?:запланирован\w*|намечен\w*|назначен\w*)\b",
        text,
    )
    return bool(travelling or future_residence or planned_trip)


def _mentions_professional_activity(answer: str) -> bool:
    text = _words(answer)
    # Describing a profession, salaried work or a private practice is
    # potentially qualifying even without owning a registered company.
    profession = re.search(
        r"\b(?:я|по профессии|моя профессия|по специальности|"
        r"по образованию|работаю|являюсь|тружусь|я по профессии) "
        r"(?:[\w]+ )?"
        r"(?:бухгалтер\w*|инженер\w*|архитектор\w*|"
        r"дизайнер\w*|аналитик\w*|программист\w*|"
        r"юрист\w*|экономист\w*|финансист\w*|"
        r"разработчик\w*|строитель\w*|маркетолог\w*|"
        r"врач\w*|преподаватель\w*|мастер\w*)\b",
        text,
    )
    # Examples: "веду отчётность фирм", "делаю проектирование",
    # "у меня собственное дело, мастерская".
    practical_work = re.search(
        r"\b(?:веду (?:отчетност\w*|бухгалтерию|учет|клиентов|практику)|"
        r"(?:делаю|выполняю|занимаюсь|оказываю|провожу) "
        r"(?:проектировани\w*|разработк\w*|заказ\w*|"
        r"услуг\w*|консультаци\w*)|"
        r"проектирую|выполняю заказы|работаю на себя|"
        r"(?:есть |имею |у меня есть )?(?:собственн\w*|свое|свой) дело|"
        r"частная практика|частную практику|"
        r"(?:моя |своя |собственная |владею )мастерск\w*)\b",
        text,
    )
    return bool(profession or practical_work)


def _explicit_location_mismatch(
    evidence: str, answer: str, all_answers: Sequence[str]
) -> bool:
    if not _fully_verified_core_answer("location", evidence, answer):
        return False
    # Validate the quote only against the location answer, but consider
    # clarifications in ALL three fields before making an irreversible decision.
    quote, full = _words(evidence), _words(" ".join(all_answers))
    # An explicit negative about being outside Georgia AND no visit planned is
    # required. Silence about arrival or an unclear location is not proof.
    outside = re.search(
        r"\b(?:не в грузии|не нахожусь в грузии|не живу в грузии|"
        r"вне грузии|за пределами грузии|not in georgia|outside georgia)\b",
        quote,
    )
    no_visit = re.search(
        r"\b(?:не (?:планирую|собираюсь|буду) (?:приезжать|приехать|"
        r"ехать|лететь|посещать)|(?:приезжать|приехать|ехать|лететь) "
        r"не (?:планирую|собираюсь|буду)|в грузию (?:ехать |приезжать )?"
        r"не (?:планирую|собираюсь|поеду|приеду)|"
        r"не (?:поеду|приеду) в грузию|"
        r"(?:do not|don t|dont|won t|will not) (?:plan to )?(?:visit|come to|travel to) georgia)\b",
        quote,
    )
    # Contradictory affirmative residence, especially in Tbilisi/Batumi, needs
    # human interpretation; the model may have scored an obviously valid reply 0.
    positive_residence = re.search(
        r"(?<!не )\b(?:живу|нахожусь|работаю|сейчас|я) в "
        r"(?:грузии|тбилиси|батуми|кутаиси|рустави|поти|гори)\b",
        full,
    ) or re.search(
        r"\b(?:i am|i live|i currently live|i m) in "
        r"(?:georgia|tbilisi|batumi|kutaisi)\b", full,
    )
    # Check the *whole* answer, not merely the AI-selected negative quote.
    # A candidate may first say "не планирую приезжать" and later clarify
    # "через три месяца переезжаю в Тбилиси". Never auto-reject in that case.
    future_arrival = re.search(
        r"\b(?:переезжаю|перееду|приеду|приедем|прилечу|поеду|"
        r"планирую переезд|собираюсь переезжать|хочу переехать|"
        r"(?<!не )планирую (?:приехать|переехать|прилететь)|"
        r"(?<!не )собираюсь (?:приехать|переехать|прилететь)|"
        r"скоро буду в (?:грузии|тбилиси|батуми|кутаиси)|"
        r"moving to|relocating to|i will visit|i will come to|"
        r"planning to visit|planning to move|"
        r"(?<!не )буду в (?:грузии|тбилиси|батуми|кутаиси)|"
        r"билет\w* (?:на (?:рейс )?)?(?:до|в) "
        r"(?:грузию|грузии|тбилиси|батуми|кутаиси))\b",
        full,
    )
    # Contrast, examples and hypothetical statements are not reliable proof
    # even if the model quotes a literal negative fragment.
    ambiguity = re.search(
        r"\b(?:но|однако|хотя|при этом|если|например|цитата|"
        r"цитирую|допустим|but|however|although|though|"
        r"for example|hypothetically)\b",
        full,
    )
    other_arrival = any(_mentions_georgia_visit_or_residence(a) for a in all_answers)
    return bool(outside and no_visit and not positive_residence
                and not future_arrival and not other_arrival and not ambiguity)


def _explicit_business_mismatch(
    evidence: str, answer: str, all_answers: Sequence[str]
) -> bool:
    if not _fully_verified_core_answer("business", evidence, answer):
        return False
    # The evidence belongs to the business field; the role may be disclosed
    # in the location or purpose field instead.
    quote, full = _words(evidence), _words(" ".join(all_answers))
    # 'I have no company' is not enough: employees and specialists qualify.
    no_business = re.search(
        r"\b(?:нет (?:ни )?бизнеса|не имею бизнеса|"
        r"не занимаюсь бизнесом|no business|not in business)\b", quote
    )
    no_project = re.search(
        r"\b(?:нет (?:никаких |ни )?(?:проектов|проекта)|"
        r"не имею проектов|не веду проектов|no projects?)\b", quote
    )
    no_work = re.search(
        r"\b(?:не работаю|нигде не работаю|нет работы|"
        r"не занимаюсь никакой деятельностью|not working|no job|unemployed)\b",
        quote,
    )
    # Any affirmative professional role requires manual review.
    professional_role = re.search(
        r"\b(?:я (?:работаю|предприниматель|руководитель|менеджер|"
        r"специалист|разработчик|директор|основатель|владелец)|"
        r"владею (?:бизнесом|компанией)|руковожу|"
        r"(?:веду проект|есть проекты|есть проект)|i (?:work|own|manage)|"
        r"(?<!не )(?:работаю|разрабатываю|управляю|сотрудничаю))\b",
        full,
    )
    # Owning a company is NOT the only qualifying involvement. Investors,
    # shareholders, advisers, freelancers and startup backers need review even
    # when they also say "нет бизнеса, нет проектов, нигде не работаю".
    other_business_activity = re.search(
        r"\b(?:инвестор\w*|инвестир\w*|инвестиц\w*|финансир\w*|"
        r"акционер\w*|учредител\w*|соучредител\w*|"
        r"консультант\w*|консультир\w*|ментор\w*|наставник\w*|"
        r"фриланс\w*|самозанят\w*|предпринимател\w*|"
        r"стартап\w*|партнер\w*|партнерств\w*|"
        r"развиваю|запускаю|участвую|"
        r"investor|investing|investments?|financ(?:e|ing)|"
        r"shareholder|advisor|adviser|consultant|freelancer|"
        r"startup|cofounder|co founder)\b",
        full,
    )
    ambiguity = re.search(
        r"\b(?:но|однако|хотя|при этом|если|например|цитата|"
        r"цитирую|допустим|but|however|although|though|"
        r"for example|hypothetically)\b",
        full,
    )
    other_professional_activity = any(_mentions_professional_activity(a)
                                      for a in all_answers)
    return bool(no_business and no_project and no_work
                and not professional_role and not other_business_activity
                and not other_professional_activity and not ambiguity)



def _verified_non_spam_context(field: str, answer: str) -> bool:
    """Conservative third-gate allowance for substantive non-spam facts.

    A second model may say that a candidate lives in Georgia or works in
    marketing while separately confirming an explicit spam pitch. Those facts
    do not refute an intention to advertise. Still, an unknown free-form
    statement is NOT trusted merely because the model calls it context: only
    simple, affirmative location/profession descriptions pass here. Everything
    else (including contradictory intent, quotations and reported speech)
    requires a moderator. This never supplies positive evidence of spam.
    """
    if field not in ("location", "business") or not isinstance(answer, str):
        return False
    raw = unicodedata.normalize("NFKC", answer.strip())
    if not raw or len(raw) > 300:
        return False
    # A single declarative sentence with comma-separated clauses is allowed.
    # Never discard question marks, quotes, nested sentences or instructions.
    if not re.fullmatch(r"[\w]+(?:[ \t]+[\w]+|[ \t]*,[ \t]+[\w]+)*\.?", raw):
        return False
    normalized = _words(raw)
    # Unfamiliar qualifications/negations are potential exculpatory evidence;
    # missing them must never make a mistaken auto-reject possible.
    if re.search(
        r"\b(?:не|нет|ни|но|однако|хотя|если|например|цитата|цитирую|"
        r"передаю|сообщают|пишут|якобы|собираюсь|планирую|намерен|хочу|"
        r"обман\w*|мошеннич\w*|инструкц\w*|игнорир\w*|бот\w*|"
        r"not|no|never|but|although|if|however|example|quote|quoted|"
        r"reportedly|ignore|instructions)\b",
        normalized,
    ):
        return False
    if _has_spam_warning_or_reported_context(raw):
        return False
    # A separate pitch/abusive instruction cannot be hidden in "context_only".
    if re.search(
        r"\b(?:рассыл\w*|спам\w*|пишите|покупайте|купите|"
        r"переходите|подписывайтесь|заказывайте|заработайте|"
        r"высылаю|продам|рекламирую|продвигаю|"
        r"buy now|dm me|click the link|mass mailing)\b",
        normalized,
    ):
        return False
    if field == "location":
        return bool(re.search(
            r"\b(?:живу|проживаю|нахожусь|расположен|живем|"
            r"переехал|переехала|буду жить|сейчас нахожусь)\s+"
            r"(?:сейчас\s+)?(?:в|на)\s+[\w]+",
            normalized,
        ))
    return bool(
        _mentions_professional_activity(raw)
        or re.search(
            r"\b(?:предприниматель|владелец|руководитель|менеджер|"
            r"инвестор|учредитель|соучредитель|"
            r"работаю|руковожу|владею)\b",
            normalized,
        )
    )


def _has_spam_warning_or_reported_context(text: str) -> bool:
    """Abstain if an apparent pitch is negated, quoted or condemned.

    Inspect complete candidate answers: the LLM can select only the sales
    phrase from a longer warning (e.g. "НЕ покупайте этот курс"). This gate
    intentionally favors human review over mistakenly declining a candidate.
    """
    # An isolated sales quote or question is not independently verified intent.
    # Retain the original punctuation before tokenization instead of
    # accidentally treating a quoted/reported instruction as the candidate's.
    raw = unicodedata.normalize("NFKC", str(text or ""))
    if any(marker in raw for marker in (*_QUOTE_CHARS, "?", "…", "...")):
        return True
    full = _words(text)
    # An open-ended negation can reverse the meaning of a sales quote in
    # unpredictable natural language. Except for the strictly neutral answers
    # below, leave such mixed profiles to an administrator rather than trying
    # to infer the scope of the negation from a matching keyword.
    neutral = {
        "нет", "нет данных", "нет дополнительных сведений", "не знаю",
        "не указано", "не применимо", "no information",
        "no additional information", "none", "not applicable", "no comment",
    }
    if full not in neutral and re.search(
        r"\b(?:не|нет|ни|not|no|never)\b", full
    ):
        return True
    return bool(re.search(
        r"\b(?:не (?:буду|собираюсь|планирую|хочу|намерен|стану) "
        r"(?:[\w]+ ){0,5}?(?:рассыл\w*|реклам\w*|спам\w*|"
        r"продава\w*|предлага\w*)|"
        r"не (?:рассыл\w*|рекламир\w*|спамлю|продаю|предлагаю)|"
        r"отказываюсь от (?:реклам\w*|рассыл\w*|спам\w*)|"
        r"это (?:был|была|было) (?:пример|цитата|шутка)|"
        r"спам выглядит так|это чужая реклама|мне (?:прислали|написали)|"
        r"не (?:[\w]+ ){0,3}(?:покупайте|купите|переходите|"
        r"подписывайтесь|заказывайте|пишите|верьте|отправляйте|"
        r"игнорируй|игнорируйте|рассылаю)|"
        r"не (?:[\w]+ ){0,5}(?:массов\w*|рассылк\w*|"
        r"гарантир\w*|манипулир\w*|инструкц\w*)|"
        r"никогда не|ни в коем случае не|не ведитесь|не рекомендую|"
        r"это обман|это развод|это мошенничество|это скам|"
        r"мошенники|мошенничество|обманщики|разоблачение|"
        r"осторожно|предупреждаю|не верьте|"
        r"как пример|для примера|например|цитирую|цитата|"
        r"если бы|гипотетически|"
        r"don t (?:[\w]+ ){0,3}(?:buy|click|subscribe|trust|"
        r"visit|follow|send|ignore)|"
        r"do not (?:[\w]+ ){0,3}(?:buy|click|subscribe|trust|"
        r"visit|follow|send|ignore)|"
        r"never (?:buy|click|trust)|avoid (?:buying|clicking)|"
        r"this is a scam|scam warning|beware of scams|"
        r"for example|hypothetically|i am quoting)\b",
        full,
    ))


def _unambiguous_spam_evidence(spam_type: str, evidence: str) -> bool:
    """Additional *independent* corroboration of hard spam classification.

    A verbatim quote like 'I am testing the bot' / 'I run an IT business' is
    NOT evidence of manipulation/fraud merely because the model says so.
    Unrecognized patterns always go to admins (false negatives are safer).
    """
    quote = _words(evidence)
    if re.search(r"\b(?:тестирую|тестируем|проверяю|testing|test the bot|"
                 r"just testing|просто шучу|шутка)\b", quote):
        # Only a separately recognizable abusive instruction/pitch can override
        # the otherwise innocuous mention of testing.
        if not re.search(r"\b(?:игнорируй|забудь инструкции|поставь оценку|"
                         r"ставь 100|ignore instructions|buy now|пишите в лс)\b", quote):
            return False
    patterns = {
        "advertising": (
            r"\b(?:пишите в лс|пишите мне|переходите по ссылке|"
            r"подписывайтесь|покупайте|купите|заказывайте|"
            r"buy now|dm me|click the link)\b"
        ),
        "fraud": (
            r"\b(?:заработ\w* .{0,60} без вложени\w*|"
            r"гарантир\w* .{0,35} доход|"
            r"доход .{0,35} гарантир\w*|"
            r"guaranteed (?:income|profit)|earn .{0,40} no investment)\b"
        ),
        "mass_template": r"\b(?:массов\w* рассылк\w*|рассылаю всем|mass mailing)\b",
        "manipulation": (
            r"\b(?:игнорируй .{0,35} инструкц\w*|"
            r"забудь .{0,35} инструкц\w*|"
            r"поставь .{0,30} (?:балл\w*|оценк\w*)|"
            r"ставь (?:100|максимальн\w*)|"
            r"ignore .{0,35} instructions|give me (?:100|full) points)\b"
        ),
        # Gibberish cannot reliably be proved with a substring/keyword rule.
        # It is always reviewed by a human rather than auto-declined.
    }
    pattern = patterns.get(spam_type)
    # If a candidate is *warning about* scams or showing an example, let an
    # admin assess the context instead of automatically rejecting them.
    quoted_warning = re.search(
        r"\b(?:предупреждаю|осторожно|мошенники|не верьте|разоблачаю|"
        r"пример мошенничества|scam warning|beware of scams)\b", quote
    )
    if not pattern or quoted_warning or not re.search(pattern, quote):
        return False
    if spam_type == "advertising":
        # A networking contact invitation is not automatically advertising.
        sales = re.search(
            r"\b(?:курс\w*|обучени\w*|скидк\w*|товар\w*|"
            r"услуг\w*|заработ\w*|покупа\w*|купите|продаж\w*|"
            r"подписк\w*|промокод\w*|buy|discount|course|sale)\b", quote
        )
        return bool(sales)
    return True
