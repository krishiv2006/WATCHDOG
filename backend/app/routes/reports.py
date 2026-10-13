"""Report management + generation endpoints."""

import csv
import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models import Report, Model as ModelDB, DriftScore, PerformanceSnapshot
from app.schemas import ReportCreate
from app.config import get_settings

router = APIRouter(prefix="/api", tags=["reports"])
settings = get_settings()


def _format_date_range(start: datetime, end: datetime) -> str:
    return f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"


def _format_time_ago(dt: datetime | None) -> str:
    if not dt:
        return "In progress..."
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
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

    # Schedule generation via APScheduler instead of raw threading
    report_id = report.id
    try:
        from app.services.scheduler import _scheduler
        if _scheduler and _scheduler.running:
            _scheduler.add_job(
                generate_report_file,
                args=[report_id],
                id=f"report_{report_id}",
                replace_existing=True,
            )
        else:
            import threading
            threading.Thread(target=generate_report_file, args=(report_id,), daemon=True).start()
    except Exception:
        import threading
        threading.Thread(target=generate_report_file, args=(report_id,), daemon=True).start()

    return {"id": report.id, "status": "generating"}


@router.get("/reports/{report_id}/download")
def download_report(report_id: str, db: Session = Depends(get_db)):
    """Download report as a CSV file."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.status != "ready" or not report.file_path:
        raise HTTPException(status_code=404, detail="Report not ready yet")
    if not os.path.exists(report.file_path):
        raise HTTPException(status_code=404, detail="Report file missing")
    return FileResponse(report.file_path, media_type="text/csv", filename=f"report_{report_id[:8]}.csv")


def generate_report_file(report_id: str) -> None:
    """Generate report CSV. Called by scheduler or fallback thread."""
    db = SessionLocal()
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            return
        try:
            reports_dir = settings.REPORTS_DIR
            os.makedirs(reports_dir, exist_ok=True)
            filepath = os.path.join(reports_dir, f"{report_id}.csv")

            drifts = db.query(DriftScore).filter(
                DriftScore.model_id == report.model_id,
                DriftScore.timestamp >= report.date_range_start,
                DriftScore.timestamp <= report.date_range_end,
            ).order_by(DriftScore.timestamp).all()

            snapshots = db.query(PerformanceSnapshot).filter(
                PerformanceSnapshot.model_id == report.model_id,
                PerformanceSnapshot.timestamp >= report.date_range_start,
                PerformanceSnapshot.timestamp <= report.date_range_end,
            ).order_by(PerformanceSnapshot.timestamp).all()

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Section", "Metric / Feature", "Value", "Status", "Timestamp"])
                for d in drifts:
                    psi = f"{d.psi_score:.4f}" if d.psi_score is not None else "n/a"
                    js = f"{d.js_distance:.4f}" if d.js_distance is not None else "n/a"
                    writer.writerow([
                        "Feature Drift", d.feature_name,
                        f"PSI={psi}, JS={js}, n_live={d.n_live}",
                        d.status,
                        d.timestamp.isoformat() if d.timestamp else "",
                    ])
                for s in snapshots:
                    acc = f"{s.accuracy:.4f}" if s.accuracy is not None else "n/a"
                    f1 = f"{s.f1:.4f}" if s.f1 is not None else "n/a"
                    writer.writerow([
                        "Performance", s.timestamp.strftime("%Y-%m-%d") if s.timestamp else "",
                        f"Acc={acc}, F1={f1}, n_labeled={s.n_labeled}, coverage={s.label_coverage}",
                        s.perf_status or "snapshot",
                        s.timestamp.isoformat() if s.timestamp else "",
                    ])

            report.status = "ready"
            report.file_path = filepath
            report.generated_at = datetime.now(timezone.utc)
        except Exception:
            report.status = "failed"
            report.generated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()

