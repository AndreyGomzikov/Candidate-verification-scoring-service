# AI-Гейткипер — подготовка Epic 6: изолированный PostgreSQL

Этот пакет собран из переданного архива Candidate-verification-scoring-service и не содержит пользовательский `.env`, SSH-пароль, API-ключи или Git-историю. Код бизнес-логики не изменён. Внесены только изменения конфигурации Docker Compose, шаблона окружения, исключений Docker-контекста и скрипт создания резервной копии.

## Перед началом

1. Смените ранее опубликованные учётные данные SSH, токен Telegram-бота через BotFather и API-ключ DeepSeek. Новые значения храните только на сервере.
2. Для первых тестов используйте ОТДЕЛЬНЫЙ тестовый чат Telegram, а не рабочее сообщество.
3. Папка /var/sftp/AI-Gatekeeper уже выделена вам. Не трогайте существующие контейнеры других проектов.
4. Пакет рассчитан на Docker + Docker Compose и 1 экземпляр приложения (long polling).

## Загрузка пакета (PowerShell на Windows)

После скачивания ZIP, находясь в папке, куда скачан файл:

```powershell
scp -P 22 "AI-Gatekeeper_PostgreSQL_пакет.zip" andrewgf1@161.97.72.91:/var/sftp/AI-Gatekeeper/
```

## На сервере — распаковка без запуска бота

```bash
cd /var/sftp/AI-Gatekeeper
unzip -n AI-Gatekeeper_PostgreSQL_пакет.zip
ls -la
```

Если unzip не установлен, используйте `python3 -m zipfile -e AI-Gatekeeper_PostgreSQL_пакет.zip .`; предварительно убедитесь, что не перезаписываете ваши файлы.

## Настройка конфиденциального .env

```bash
cd /var/sftp/AI-Gatekeeper
umask 077
cp -n .env.example .env
openssl rand -hex 24
nano .env
chmod 600 .env
```

Вставьте сгенерированную шестнадцатеричную строку в `POSTGRES_PASSWORD`. Заполните остальные переменные новыми токенами и ID **тестовых** Telegram-чатов. Запрещается пересылать полный `.env` в чат.

**Важно:** В предыдущем `.env` было `LLM_API_URL`; программный код ожидает `LLM_BASE_URL`. Для DeepSeek используется именно `LLM_BASE_URL=https://api.deepseek.com`. Приложение само дописывает `/chat/completions`. `LLM_MODEL=deepseek-v4-pro`.

## Предварительная проверка

```bash
cd /var/sftp/AI-Gatekeeper
docker compose config --quiet
```

Внимание: не показывайте вывод `docker compose config` без `--quiet`, т.к. он может содержать переменные с ключами.

## Запуск ТОЛЬКО отдельной БД, без бота

```bash
docker compose up -d postgres
docker compose ps
docker compose exec -T postgres pg_isready -U gatekeeper -d gatekeeper
```

Ожидается сервис `ai-gatekeeper-postgres-1` со статусом `healthy`, порт 5432 наружу не публикуется.

## После проверки БД и тестовых Telegram ID

**Не запускать до подтверждения токена и ID тестовых чатов:**

```bash
docker compose up -d --build gatekeeper
docker compose ps
docker compose logs --tail=40 gatekeeper
```

В логах не должны отображаться токены. Нельзя использовать один токен Telegram в двух одновременных long-polling процессах.

## Резервные копии (после проверки PostgreSQL)

```bash
bash scripts/backup_db.sh
ls -lh backups/
```

Скрипт создаёт файл вида `backups/gatekeeper_YYYYMMDDTHHMMSSZ.sql.gz`. Рекомендуется дополнительно копировать backup за пределы сервера и тестировать восстановление. **Автоматическое расписание и проверка восстановления пока не настроены.**

## Что остаётся для закрытия Epic 6

- Проверить реальные полномочия бота и тестовые Telegram ID.
- Проверить подключение к DeepSeek без ошибок и старт микросервиса.
- Настроить CI и проверку тестов.
- Проверить бэкап и восстановление; добавить процедуру отката и журнал операций.
- Уточнить третий сторонний лицензируемый код: README ссылается на `LICENSE.ringo` и `THIRD_PARTY_NOTICES.md`, но их нет в исходном архиве.

## Запрещённые/опасные действия

Не используйте `docker compose down -v` — эта команда удалит том PostgreSQL AI-Гейткипера. Не используйте `docker system prune --volumes`, `docker volume prune` или команды остановки/удаления контейнеров других проектов.
