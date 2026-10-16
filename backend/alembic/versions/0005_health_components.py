"""Persist health score component breakdowns."""

from alembic import op
import sqlalchemy as sa


revision = "0005_health_components"
down_revision = "0004_analyze_csv_uploads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("models")}
    if "health_components" in columns:
        return
    op.add_column("models", sa.Column("health_components", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "health_components")
