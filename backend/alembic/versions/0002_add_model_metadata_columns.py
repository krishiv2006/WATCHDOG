"""Add model metadata columns to databases created before phase 2.

Revision ID: 0002
Revises: 0001
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "models" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("models")
    }
    columns = {
        "task_type": sa.Column("task_type", sa.String(50), nullable=True),
        "prediction_column": sa.Column(
            "prediction_column", sa.String(100), nullable=True
        ),
        "label_delay_hint": sa.Column(
            "label_delay_hint", sa.String(50), nullable=True
        ),
        "retired_at": sa.Column(
            "retired_at", sa.DateTime(timezone=True), nullable=True
        ),
        "training_data_ref": sa.Column(
            "training_data_ref", sa.String(255), nullable=True
        ),
    }

    for name, column in columns.items():
        if name not in existing_columns:
            op.add_column("models", column)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "models" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("models")
    }
    for name in (
        "training_data_ref",
        "retired_at",
        "label_delay_hint",
        "prediction_column",
        "task_type",
    ):
        if name in existing_columns:
            op.drop_column("models", name)
