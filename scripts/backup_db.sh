#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
mkdir -p backups
chmod 700 backups
file="backups/gatekeeper_$(date -u +%Y%m%dT%H%M%SZ).sql.gz"
# Database-only logical backup. Never run `docker compose down -v`.
if docker compose exec -T postgres pg_dump --no-owner --no-acl -U gatekeeper -d gatekeeper | gzip -c > "$file"; then
    echo "Backup created: $file"
else
    rm -f "$file"
    echo "Backup failed" >&2
    exit 1
fi
