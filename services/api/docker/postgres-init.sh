#!/bin/bash
# Creates the restricted runtime role. Plain LOGIN: no SUPERUSER and the
# default NOBYPASSRLS, which is what makes FORCE RLS actually bind. The
# superuser ($POSTGRES_USER) is reserved for migrations/provisioning.
set -euo pipefail

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<-SQL
    CREATE ROLE corag_app LOGIN PASSWORD '${CORAG_APP_DB_PASSWORD:-corag_app}';
    GRANT CONNECT ON DATABASE $POSTGRES_DB TO corag_app;
SQL
