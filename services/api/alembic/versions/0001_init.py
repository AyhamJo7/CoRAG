"""Initial schema: identity, tenancy (FORCE RLS), documents, chunks, billing.

Revision ID: 0001
Revises:
Create Date: 2026-07-12
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Tables scoped by tenant_id under FORCE RLS. `tenant` itself keys on `id`.
TENANT_TABLES = ["document", "chunk", "question_log"]

APP_ROLE = "corag_app"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    # --- identity (deliberately NOT RLS'd: login precedes tenant context) ---
    op.execute(
        """
        CREATE TABLE app_user (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            email citext NOT NULL,
            name text NOT NULL,
            password_hash text NOT NULL,
            session_version integer NOT NULL DEFAULT 1,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX ux_app_user_email ON app_user (email)")

    op.execute(
        """
        CREATE TABLE tenant (
            id uuid PRIMARY KEY,
            name text NOT NULL,
            plan text NOT NULL DEFAULT 'trial',
            trial_ends_at timestamptz,
            stripe_customer_id text UNIQUE,
            stripe_subscription_id text UNIQUE,
            subscription_status text,
            current_period_end timestamptz,
            questions_used integer NOT NULL DEFAULT 0,
            docs_count integer NOT NULL DEFAULT 0,
            storage_bytes_used bigint NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE user_tenant (
            user_id uuid NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
            tenant_id uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
            role text NOT NULL CHECK (role IN ('owner', 'member')),
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, tenant_id)
        )
        """
    )
    op.execute("CREATE INDEX ix_user_tenant_tenant ON user_tenant (tenant_id)")

    # --- tenant data ---
    op.execute(
        """
        CREATE TABLE document (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
            title text NOT NULL,
            filename text NOT NULL,
            mime text NOT NULL,
            size_bytes bigint NOT NULL,
            status text NOT NULL DEFAULT 'uploaded'
                CHECK (status IN ('uploaded', 'processing', 'indexed', 'failed')),
            error text,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_document_tenant ON document (tenant_id, created_at)")

    op.execute(
        """
        CREATE TABLE chunk (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES document(id) ON DELETE CASCADE,
            chunk_index integer NOT NULL,
            text text NOT NULL,
            tokens integer NOT NULL DEFAULT 0,
            start_char integer NOT NULL DEFAULT 0,
            end_char integer NOT NULL DEFAULT 0,
            embedding vector(1536),
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_chunk_tenant_document ON chunk (tenant_id, document_id)"
    )
    op.execute(
        "CREATE INDEX ix_chunk_embedding_hnsw ON chunk "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    op.execute(
        """
        CREATE TABLE question_log (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
            user_id uuid NOT NULL,
            question text NOT NULL,
            answer text NOT NULL DEFAULT '',
            citations jsonb NOT NULL DEFAULT '[]'::jsonb,
            num_steps integer NOT NULL DEFAULT 0,
            num_chunks integer NOT NULL DEFAULT 0,
            latency_ms integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_question_log_tenant ON question_log (tenant_id, created_at)"
    )

    # --- work queue (no RLS: the worker claims jobs across tenants; rows
    # carry only ids/status, never content) ---
    op.execute(
        """
        CREATE TABLE ingest_job (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id uuid NOT NULL,
            document_id uuid NOT NULL,
            status text NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'done', 'failed')),
            attempts integer NOT NULL DEFAULT 0,
            last_error text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_ingest_job_status ON ingest_job (status, id)")

    # --- webhook idempotency (no RLS: no tenant content) ---
    op.execute(
        """
        CREATE TABLE stripe_event (
            id text PRIMARY KEY,
            type text NOT NULL,
            processed_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    # --- FORCE RLS ---
    op.execute("ALTER TABLE tenant ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenant FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_tenant_isolation ON tenant
            USING (id = current_setting('app.tenant_id')::uuid)
            WITH CHECK (id = current_setting('app.tenant_id')::uuid)
        """
    )
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_tenant_isolation ON {table}
                USING (tenant_id = current_setting('app.tenant_id')::uuid)
                WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid)
            """
        )

    # --- identity-path RLS exception: login must list a user's workspaces
    # (tenant name/plan) before any tenant context exists. SECURITY DEFINER
    # runs as the migration owner (admin), scoped to exactly this query. ---
    op.execute(
        """
        CREATE FUNCTION user_memberships(p_user_id uuid)
        RETURNS TABLE (id uuid, name text, role text, plan text)
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = public
        AS $$
            SELECT t.id, t.name, ut.role, t.plan
            FROM user_tenant ut
            JOIN tenant t ON t.id = ut.tenant_id
            WHERE ut.user_id = p_user_id
            ORDER BY ut.created_at
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION user_memberships(uuid) FROM PUBLIC")

    # --- app-role grants (role created by docker/postgres-init.sh; guarded so
    # the migration also runs against databases prepared differently) ---
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON
                    app_user, user_tenant, tenant, document, chunk,
                    question_log, ingest_job, stripe_event
                    TO {APP_ROLE};
                GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE};
                GRANT EXECUTE ON FUNCTION user_memberships(uuid) TO {APP_ROLE};
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS user_memberships(uuid)")
    for table in [
        "stripe_event",
        "ingest_job",
        "question_log",
        "chunk",
        "document",
        "user_tenant",
        "tenant",
        "app_user",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
