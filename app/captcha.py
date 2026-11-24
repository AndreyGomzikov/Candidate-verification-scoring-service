"""Emoji-by-meaning captcha for join requests.

The applicant is asked to press the one emoji that fits a category
("то, что можно съесть"). The other buttons come from *different* categories,
so there is never more than one correct answer.

Design rules, enforced by tests/test_captcha.py:
* every emoji belongs to exactly one category and is used only once overall;
* categories must not overlap in meaning (no "plants" next to "food");
* only long-established emoji without variation selectors, so that buttons
  look the same on iOS, Android and desktop.

The correct answer is NEVER put into callback_data and is stored server-side.
This stops only cheap scripts. A bot driven by an LLM will solve it, which is
why the captcha is only the first, cheap layer in front of the questionnaire.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# phrase is the accusative object of "Нажмите на ..."
CATEGORIES: dict[str, tuple[str, ...]] = {
    "то, что можно съесть": ("🍎", "🍕", "🍞", "🍌", "🍇", "🍪"),
    "то, что можно выпить": ("☕", "🥛", "🧃", "🍷", "🍺", "🍵"),
    "животное": ("🐕", "🐈", "🐘", "🦊", "🐢", "🐧"),
    "транспорт": ("🚗", "🚲", "🚂", "🚢", "🚁", "🚌"),
    "то, что можно надеть": ("👕", "👖", "🧢", "🧦", "👗", "🧥"),
    "музыкальный инструмент": ("🎸", "🎹", "🥁", "🎻", "🎺", "🎷"),
    "то, что можно увидеть на небе": ("🌙", "⭐", "🌈", "🌞"),
    "мяч": ("⚽", "🏀", "🏈", "🎾", "⚾", "🏐"),
    "часть тела": ("👀", "👂", "👃", "👄", "🦷", "🦵"),
    "электронное устройство": ("📱", "💻", "📷", "📺"),
    "здание": ("🏠", "🏰", "🏢", "🏭", "🏥", "🏫"),
}

DEFAULT_OPTIONS = 4
_SYSTEM_RNG = random.SystemRandom()


@dataclass(frozen=True)
class Captcha:
    prompt: str                 # e.g. "то, что можно съесть"
    options: tuple[str, ...]    # emoji in display order
    correct_index: int          # position of the right emoji in `options`


def make_captcha(
    rng: random.Random | None = None, options_count: int = DEFAULT_OPTIONS
) -> Captcha:
    """Build one fresh captcha: random category, random distractors, random order."""
    rng = rng or _SYSTEM_RNG
    if not 2 <= options_count <= len(CATEGORIES):
        raise ValueError("options_count must be between 2 and the number of categories")
    prompt = rng.choice(sorted(CATEGORIES))
    correct = rng.choice(CATEGORIES[prompt])
    other_categories = [name for name in sorted(CATEGORIES) if name != prompt]
    distractors = [
        rng.choice(CATEGORIES[name])
        for name in rng.sample(other_categories, options_count - 1)
    ]
    options = distractors + [correct]
    rng.shuffle(options)
    return Captcha(prompt=prompt, options=tuple(options), correct_index=options.index(correct))
