"""Report management + generation endpoints."""

import csv
import io
import os
import threading
import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models import Report, Model as ModelDB, DriftScore, PerformanceSnapshot
from app.schemas import ReportCreate

router = APIRouter(prefix="/api", tags=["reports"])


def _format_date_range(start: datetime, end: datetime) -> str:
    return f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"


def _format_time_ago(dt: datetime | None) -> str:
    if not dt:
        return "In progress..."
    delta = datetime.utcnow() - dt
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() / 60)} mins ago"
    if delta.total_seconds() < 86400:
        h = int(delta.total_seconds() / 3600)
        return f"{h} hour{'s' if h > 1 else ''} ago"
    d = int(delta.days)
    return f"{d} day{'s' if d > 1 else ''} ago"


@router.get("/reports")
def list_reports(db: Session = Depends(get_db)):
    """Report history list."""
    reports = db.query(Report).order_by(Report.generated_at.desc().nullsfirst()).all()
    result = []
    for r in reports:
        model = db.query(ModelDB).filter(ModelDB.id == r.model_id).first()
        model_name = f"{model.name} {model.version}" if model else "Unknown"
        result.append({
            "id": r.id,
            "name": r.name or r.report_type.replace("_", " ").title(),
            "modelId": r.model_id,
            "modelName": model_name,
            "dateRange": _format_date_range(r.date_range_start, r.date_range_end),
            "status": r.status,
            "generatedAt": _format_time_ago(r.generated_at) if r.status != "failed" else f"Failed {_format_time_ago(r.generated_at)}",
        })
    return result


@router.post("/reports/generate", status_code=201)
def generate_report(payload: ReportCreate, db: Session = Depends(get_db)):
    """Kick off async report generation."""
    model = db.query(ModelDB).filter(ModelDB.id == payload.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        start = datetime.fromisoformat(payload.date_range_start)
        end = datetime.fromisoformat(payload.date_range_end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    report = Report(
        model_id=payload.model_id,
        name=f"{payload.report_type.replace('_', ' ').title()} Report",
        date_range_start=start,
        date_range_end=end,
        report_type=payload.report_type,
        status="generating",
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Run generation in background thread
    report_id = report.id
    threading.Thread(target=_generate_report_async, args=(report_id,), daemon=True).start()

    return {"id": report.id, "status": "generating"}


@router.get("/reports/{report_id}/download")
def download_report(report_id: str, db: Session = Depends(get_db)):
    """Download report as a CSV file."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    os.makedirs("reports_output", exist_ok=True)
    filepath = report.file_path or f"reports_output/{report_id}.csv"

    if not os.path.exists(filepath):
        drifts = db.query(DriftScore).filter(DriftScore.model_id == report.model_id).all()
        snapshots = db.query(PerformanceSnapshot).filter(
            PerformanceSnapshot.model_id == report.model_id,
        ).order_by(PerformanceSnapshot.timestamp).all()

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "Metric / Feature", "Value", "Status"])
            for d in drifts:
                writer.writerow([
                    "Feature Drift", d.feature_name,
                    f"PSI={d.psi_score:.4f}, JS={d.js_distance:.4f}",
                    d.status,
                ])
            for s in snapshots:
                writer.writerow([
                    "Performance", s.timestamp.strftime("%Y-%m-%d"),
                    f"Acc={s.accuracy:.4f}, Prec={s.precision_score:.4f}, Rec={s.recall:.4f}, F1={s.f1:.4f}",
                    "snapshot",
                ])

        report.file_path = filepath
        report.status = "ready"
        db.commit()

    return FileResponse(filepath, media_type="text/csv", filename=f"report_{report_id[:8]}.csv")


def _generate_report_async(report_id: str):
    """Background report generation (produces CSV)."""
    time.sleep(1.5)  # Processing time

    db = SessionLocal()
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            return

        try:
            # Generate CSV data
            os.makedirs("reports_output", exist_ok=True)
            filepath = f"reports_output/{report_id}.csv"

            # Fetch relevant data
            drifts = db.query(DriftScore).filter(
                DriftScore.model_id == report.model_id,
            ).all()

            snapshots = db.query(PerformanceSnapshot).filter(
                PerformanceSnapshot.model_id == report.model_id,
                PerformanceSnapshot.timestamp >= report.date_range_start,
                PerformanceSnapshot.timestamp <= report.date_range_end,
            ).order_by(PerformanceSnapshot.timestamp).all()

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Section", "Metric / Feature", "Value", "Status"])

                # Drift section
                for d in drifts:
                    writer.writerow([
                        "Feature Drift", d.feature_name,
                        f"PSI={d.psi_score:.4f}, JS={d.js_distance:.4f}",
                        d.status,
                    ])

                # Performance section
                for s in snapshots:
                    writer.writerow([
                        "Performance", s.timestamp.strftime("%Y-%m-%d"),
                        f"Acc={s.accuracy:.4f}, Prec={s.precision_score:.4f}, Rec={s.recall:.4f}, F1={s.f1:.4f}",
                        "snapshot",
                    ])

            report.status = "ready"
            report.file_path = filepath
            report.generated_at = datetime.utcnow()
        except Exception as e:
            report.status = "failed"
            report.generated_at = datetime.utcnow()

        db.commit()
    finally:
        db.close()

