from app import texts


def test_first_question_matches_requested_wording():
    assert texts.INTRO == (
        "Здравствуйте! 👋\n\n"
        "Спасибо за интерес к сообществу Solidus в Грузии!\n"
        "Чтобы рассмотреть заявку, ответьте, пожалуйста, на 3 коротких вопроса."
    )
    assert texts.QUESTION_LOCATION == (
        "1/3. Где вы сейчас находитесь? Если вы не в Грузии, укажите "
        "планируемые даты приезда."
    )
    assert (texts.INTRO + "\n\n" + texts.QUESTION_LOCATION).count(
        "Спасибо за интерес к сообществу Solidus в Грузии!"
    ) == 1


def test_second_question_matches_requested_wording():
    assert texts.QUESTION_BUSINESS == (
        "2/3. Расскажите о вашем бизнес-профиле: ваша роль в проекте, ниша, "
        "размер бизнеса. Если есть, ссылка на сайт или профиль в соцсетях."
    )


def test_approval_notification_includes_club_link():
    assert texts.APPROVED == (
        "✅ Ваша заявка одобрена. Добро пожаловать в сообщество Solidus! "
        "(https://t.me/SOLIDUS_business_club)"
    )


def test_rejection_notification_uses_georgia():
    assert "связанных с Грузией" in texts.REJECTED
    assert "Батуми / Аджарией" not in texts.REJECTED
