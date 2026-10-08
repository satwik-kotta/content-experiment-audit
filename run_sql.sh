#!/usr/bin/env bash
# Usage: ./run_sql.sh <database>   (needs PostgreSQL and psql on PATH; run from repo root)
set -e
DB=${1:-content}
for f in sql/0*.sql; do echo "== $f"; psql -d "$DB" -f "$f"; done
