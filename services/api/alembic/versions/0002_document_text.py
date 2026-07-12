"""Store extracted document text for chunking/embedding.

v1 keeps the extracted plain text on the document row instead of adding an
object store: files are capped per plan, and the original bytes are not
needed after extraction.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-12
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE document ADD COLUMN text text NOT NULL DEFAULT ''")


def downgrade() -> None:
    op.execute("ALTER TABLE document DROP COLUMN text")
