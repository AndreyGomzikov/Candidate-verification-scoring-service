# AI Gatekeeper — изолированное тестовое развёртывание с эмодзи-CAPTCHA

Этот пакет предназначен для отдельного тестового окружения на Contabo. Он содержит исходный код, тесты, документацию и Docker-конфигурацию. Файл `.env` с секретами в архив не включён.

## Целевая директория на сервере

```text
/home/andrewgf1/solidus-captcha-test
```

В этой директории уже должен находиться подготовленный файл `.env` с отдельными тестовыми секретами и параметрами. Не копируйте рабочий `.env` из проекта `ai-gatekeeper`.

## 1. Загрузка ZIP с Windows PowerShell

```powershell
scp -P 22 "solidus-captcha-test.zip" andrewgf1@161.97.72.91:/home/andrewgf1/solidus-captcha-test/
```

## 2. Распаковка на сервере

```bash
cd /home/andrewgf1/solidus-captcha-test
ls -la
unzip -n solidus-captcha-test.zip
```

Архив распаковывается прямо в текущую директорию, без дополнительной вложенной папки. Ключ `-n` запрещает перезапись уже существующих файлов. В архиве нет `.env`, поэтому подготовленный серверный `.env` сохраняется.

Проверьте наличие конфигурации и права доступа, не выводя её содержимое:

```bash
test -s .env && echo 'OK: .env exists' || echo 'ERROR: .env missing'
stat -c '%a %n' .env
chmod 600 .env
```

Ожидаемые права доступа — `600`.

## 3. Проверка Compose без запуска контейнеров

```bash
docker compose -p solidus-captcha-test --env-file .env config -q
```

Команда должна завершиться без ошибок. Не запускайте `docker compose config` без `-q` и не публикуйте его полный вывод: он может содержать секреты.

## 4. Сначала запустить только отдельную PostgreSQL

```bash
docker compose -p solidus-captcha-test --env-file .env up -d postgres
docker compose -p solidus-captcha-test ps
docker compose -p solidus-captcha-test exec -T postgres pg_isready -U gatekeeper -d gatekeeper
```

Дождитесь состояния `healthy` и ответа `accepting connections`.

## 5. Затем запустить тестового бота

Перед запуском убедитесь, что `BOT_TOKEN` принадлежит отдельному тестовому боту и не используется другим запущенным процессом long polling. `TARGET_CHAT_ID` должен указывать на тестовую группу, `ADMIN_CHAT_ID` — на тестовый административный чат. Если нужно строго закрепить одного модератора, заполните `ADMIN_USER_IDS` его числовым Telegram User ID.

```bash
docker compose -p solidus-captcha-test --env-file .env up -d --build gatekeeper
docker compose -p solidus-captcha-test ps
docker compose -p solidus-captcha-test logs --tail=60 gatekeeper
```

Не отправляйте в чат полный `.env` и не публикуйте логи, если в них случайно попали секреты или персональные данные.

## 6. Изоляция и запреты

- Compose project принудительно задаётся как `solidus-captcha-test`; в `docker-compose.yml` также установлено безопасное имя этого тестового проекта.
- PostgreSQL использует отдельный Compose volume с префиксом `solidus-captcha-test`.
- Порт PostgreSQL наружу не публикуется.
- Не используйте рабочий токен Telegram одновременно в двух процессах long polling.
- Не направляйте тестового бота в рабочую группу.
- Не выполняйте `docker compose down -v`, `docker volume prune` или `docker system prune --volumes`.
- Не останавливайте и не пересоздавайте контейнеры `ai-gatekeeper`, `v-it`, `ai_digest_bot` и других проектов.

Остановка только тестового проекта (без удаления базы данных):

```bash
docker compose -p solidus-captcha-test stop
```

Для последующего запуска:

```bash
docker compose -p solidus-captcha-test --env-file .env up -d
```
