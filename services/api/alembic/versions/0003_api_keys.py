"""Developer API keys.

Identity-path table (like app_user): verification maps a presented key to
its tenant BEFORE any tenant context exists, so it is deliberately not
RLS'd. Management queries filter by tenant_id explicitly.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-12
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

APP_ROLE = "corag_app"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE api_key (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
            name text NOT NULL DEFAULT '',
            key_hash text NOT NULL UNIQUE,
            key_prefix text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            last_used_at timestamptz,
            revoked_at timestamptz
        )
        """
    )
    op.execute("CREATE INDEX ix_api_key_tenant ON api_key (tenant_id)")
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON api_key TO {APP_ROLE};
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS api_key")
