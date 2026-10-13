"""Complete columns for databases created before the phase-two schema.

Revision ID: 0003
Revises: 0002
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLE_COLUMNS = {
    "models": {
        "task_type": sa.String(50),
        "prediction_column": sa.String(100),
        "label_delay_hint": sa.String(50),
        "retired_at": sa.DateTime(timezone=True),
        "training_data_ref": sa.String(255),
    },
    "predictions_log": {
        "prediction_id": sa.String(255),
        "model_version": sa.String(50),
        "score": sa.Float(),
        "actual_value": sa.Float(),
        "labeled_at": sa.DateTime(timezone=True),
        "latency_ms": sa.Float(),
    },
    "drift_scores": {
        "js_distance": sa.Float(),
        "ks_statistic": sa.Float(),
        "ks_pvalue": sa.Float(),
        "baseline_std": sa.Float(),
        "current_std": sa.Float(),
        "null_rate_baseline": sa.Float(),
        "null_rate_live": sa.Float(),
        "out_of_range_rate": sa.Float(),
        "n_live": sa.Integer(),
        "n_baseline": sa.Integer(),
        "confidence": sa.String(10),
        "reason": sa.Text(),
        "baseline_version_id": sa.String(),
    },
    "alerts": {
        "fingerprint": sa.String(255),
        "first_seen_at": sa.DateTime(timezone=True),
        "last_seen_at": sa.DateTime(timezone=True),
        "occurrences": sa.Integer(),
        "auto_resolved": sa.Boolean(),
        "alert_rule_id": sa.String(),
        "metric_value": sa.Float(),
    },
    "alert_rules": {
        "feature_name": sa.String(255),
        "comparison": sa.String(10),
        "min_samples": sa.Integer(),
        "consecutive_windows": sa.Integer(),
        "cooldown_minutes": sa.Integer(),
        "channels": sa.JSON(),
    },
    "performance_snapshots": {
        "n_labeled": sa.Integer(),
        "label_coverage": sa.Float(),
        "perf_status": sa.String(30),
    },
    "reports": {
        "name": sa.String(255),
    },
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    for table_name, columns in TABLE_COLUMNS.items():
        if table_name not in tables:
            continue
        existing = {column["name"] for column in inspector.get_columns(table_name)}
        for name, column_type in columns.items():
            if name not in existing:
                op.add_column(table_name, sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    for table_name, columns in TABLE_COLUMNS.items():
        if table_name not in tables:
            continue
        existing = {column["name"] for column in inspector.get_columns(table_name)}
        for name in reversed(list(columns)):
            if name in existing:
                op.drop_column(table_name, name)
