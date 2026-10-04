# Third-party notices

## Ringo

Repository: https://github.com/why-not-try-calmer/ringo

Solidus AI Gatekeeper intentionally reuses/adapts the core architectural flow demonstrated by Ringo:
Telegram ChatJoinRequest -> private questionnaire -> admin moderation -> approve/decline.
Ringo is licensed under the MIT License. A copy of that license is included in `LICENSE.ringo`.
The implementation in this package is not a byte-for-byte copy of the whole Ringo project; it is an
adaptation of the relevant workflow to the Solidus requirements, with persistent interview state,
AI scoring and a different data model.

## Bouncer

Repository: https://github.com/k4yt3x/bouncer

Bouncer was used as a design reference for LLM-assisted verification and conservative gating.
No Bouncer source code is copied into this package. Bouncer is AGPL-3.0 licensed.
