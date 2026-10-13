"""initial_schema — full table creation for fresh databases.

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00

For existing databases that were created via Base.metadata.create_all,
run:  alembic stamp head
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "models" not in existing:
        op.create_table(
            "models",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("version", sa.String(50), nullable=False),
            sa.Column("algorithm", sa.String(100), nullable=False),
            sa.Column("environment", sa.String(50), nullable=False, server_default="production"),
            sa.Column("status", sa.String(50), nullable=False, server_default="production"),
            sa.Column("task_type", sa.String(50), nullable=True),
            sa.Column("prediction_column", sa.String(100), nullable=True),
            sa.Column("label_delay_hint", sa.String(50), nullable=True),
            sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_evaluated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("health_score", sa.Float(), nullable=True),
            sa.Column("training_data_ref", sa.String(255), nullable=True),
        )

    if "alert_rules" not in existing:
        op.create_table(
            "alert_rules",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=True),
            sa.Column("metric", sa.String(100), nullable=False),
            sa.Column("threshold_warning", sa.Float(), nullable=False),
            sa.Column("threshold_critical", sa.Float(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=True),
            sa.Column("feature_name", sa.String(255), nullable=True),
            sa.Column("comparison", sa.String(10), nullable=True),
            sa.Column("min_samples", sa.Integer(), nullable=True),
            sa.Column("consecutive_windows", sa.Integer(), nullable=True),
            sa.Column("cooldown_minutes", sa.Integer(), nullable=True),
            sa.Column("channels", sa.JSON(), nullable=True),
        )

    if "alerts" not in existing:
        op.create_table(
            "alerts",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=False),
            sa.Column("severity", sa.String(20), nullable=False),
            sa.Column("category", sa.String(50), nullable=False),
            sa.Column("title", sa.String(500), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("detected_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("status", sa.String(30), nullable=False, server_default="unacknowledged"),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("fingerprint", sa.String(255), nullable=True, index=True),
            sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("occurrences", sa.Integer(), nullable=True),
            sa.Column("auto_resolved", sa.Boolean(), nullable=True),
            sa.Column("alert_rule_id", sa.String(), sa.ForeignKey("alert_rules.id"), nullable=True),
            sa.Column("metric_value", sa.Float(), nullable=True),
        )

    if "model_baselines" not in existing:
        op.create_table(
            "model_baselines",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=False),
            sa.Column("feature_name", sa.String(255), nullable=False),
            sa.Column("mean", sa.Float(), nullable=True),
            sa.Column("std", sa.Float(), nullable=True),
            sa.Column("histogram_counts", sa.JSON(), nullable=True),
            sa.Column("histogram_edges", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("accuracy", sa.Float(), nullable=True),
            sa.Column("precision_score", sa.Float(), nullable=True),
            sa.Column("recall", sa.Float(), nullable=True),
            sa.Column("f1", sa.Float(), nullable=True),
        )

    if "baseline_versions" not in existing:
        op.create_table(
            "baseline_versions",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=False),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("feature_profiles", sa.JSON(), nullable=False),
            sa.Column("sample_size", sa.Integer(), nullable=True),
            sa.Column("source_file_name", sa.String(500), nullable=True),
            sa.Column("source_file_hash", sa.String(64), nullable=True),
            sa.Column("performance_metrics", sa.JSON(), nullable=True),
            sa.Column("prediction_distribution", sa.JSON(), nullable=True),
            sa.Column("n_psi_buckets", sa.Integer(), nullable=False, server_default="10"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_baseline_model_active", "baseline_versions", ["model_id", "is_active"])

    if "predictions_log" not in existing:
        op.create_table(
            "predictions_log",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=False),
            sa.Column("prediction_id", sa.String(255), nullable=True, index=True),
            sa.Column("model_version", sa.String(50), nullable=True),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True, index=True),
            sa.Column("feature_values", sa.JSON(), nullable=False),
            sa.Column("predicted_value", sa.Float(), nullable=True),
            sa.Column("score", sa.Float(), nullable=True),
            sa.Column("actual_value", sa.Float(), nullable=True),
            sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("latency_ms", sa.Float(), nullable=True),
            sa.UniqueConstraint("model_id", "prediction_id", name="uq_pred_model_prediction_id"),
        )
        op.create_index("ix_pred_model_timestamp", "predictions_log", ["model_id", "timestamp"])
        op.create_index("ix_pred_model_predid", "predictions_log", ["model_id", "prediction_id"])

    if "drift_scores" not in existing:
        op.create_table(
            "drift_scores",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=False),
            sa.Column("feature_name", sa.String(255), nullable=False),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True, index=True),
            sa.Column("psi_score", sa.Float(), nullable=True),
            sa.Column("js_distance", sa.Float(), nullable=True),
            sa.Column("ks_statistic", sa.Float(), nullable=True),
            sa.Column("ks_pvalue", sa.Float(), nullable=True),
            sa.Column("baseline_mean", sa.Float(), nullable=True),
            sa.Column("current_mean", sa.Float(), nullable=True),
            sa.Column("baseline_std", sa.Float(), nullable=True),
            sa.Column("current_std", sa.Float(), nullable=True),
            sa.Column("null_rate_baseline", sa.Float(), nullable=True),
            sa.Column("null_rate_live", sa.Float(), nullable=True),
            sa.Column("out_of_range_rate", sa.Float(), nullable=True),
            sa.Column("n_live", sa.Integer(), nullable=True),
            sa.Column("n_baseline", sa.Integer(), nullable=True),
            sa.Column("status", sa.String(20), nullable=False, server_default="stable"),
            sa.Column("confidence", sa.String(10), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("baseline_version_id", sa.String(), sa.ForeignKey("baseline_versions.id"), nullable=True),
        )
        op.create_index("ix_drift_model_feature_ts", "drift_scores", ["model_id", "feature_name", "timestamp"])

    if "performance_snapshots" not in existing:
        op.create_table(
            "performance_snapshots",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=False),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True, index=True),
            sa.Column("accuracy", sa.Float(), nullable=True),
            sa.Column("precision_score", sa.Float(), nullable=True),
            sa.Column("recall", sa.Float(), nullable=True),
            sa.Column("f1", sa.Float(), nullable=True),
            sa.Column("auc", sa.Float(), nullable=True),
            sa.Column("mae", sa.Float(), nullable=True),
            sa.Column("rmse", sa.Float(), nullable=True),
            sa.Column("r_squared", sa.Float(), nullable=True),
            sa.Column("sample_size", sa.Integer(), nullable=True),
            sa.Column("n_labeled", sa.Integer(), nullable=True),
            sa.Column("label_coverage", sa.Float(), nullable=True),
            sa.Column("perf_status", sa.String(30), nullable=True),
        )

    if "reports" not in existing:
        op.create_table(
            "reports",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=False),
            sa.Column("date_range_start", sa.DateTime(timezone=True), nullable=False),
            sa.Column("date_range_end", sa.DateTime(timezone=True), nullable=False),
            sa.Column("report_type", sa.String(100), nullable=False),
            sa.Column("format", sa.String(20), nullable=True),
            sa.Column("status", sa.String(20), nullable=False, server_default="generating"),
            sa.Column("file_path", sa.String(500), nullable=True),
            sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("name", sa.String(255), nullable=True),
        )


def downgrade() -> None:
    for tbl in [
        "reports", "performance_snapshots", "drift_scores",
        "predictions_log", "baseline_versions", "model_baselines",
        "alerts", "alert_rules", "models",
    ]:
        op.drop_table(tbl)
