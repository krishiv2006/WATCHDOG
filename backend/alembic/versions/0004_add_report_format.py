"""Add report format to legacy report tables.

Revision ID: 0004
Revises: 0003
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "reports" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("reports")}
    if "format" not in columns:
        op.add_column("reports", sa.Column("format", sa.String(20), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "reports" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("reports")}
    if "format" in columns:
        op.drop_column("reports", "format")
