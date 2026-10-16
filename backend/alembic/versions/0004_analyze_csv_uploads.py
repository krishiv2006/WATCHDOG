"""Add analyze-csv idempotency ledger."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_analyze_csv_uploads"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "analyze_csv_uploads" in inspector.get_table_names():
        return
    op.create_table(
        "analyze_csv_uploads",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(length=64), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id", "file_hash", name="uq_analyze_csv_model_hash"),
    )
    op.create_index("ix_analyze_csv_model", "analyze_csv_uploads", ["model_id"])


def downgrade() -> None:
    op.drop_index("ix_analyze_csv_model", table_name="analyze_csv_uploads")
    op.drop_table("analyze_csv_uploads")
