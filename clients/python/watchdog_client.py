"""
Watchdog AI — Python ingestion client.

Copy-paste or pip-install this file to send predictions and labels
from your ML service to Watchdog.

Usage
-----
    from watchdog_client import WatchdogClient

    client = WatchdogClient(
        base_url="http://localhost:8000",
        model_id="<your-model-id>",
        api_key="<your-ingest-api-key>",   # omit if INGEST_API_KEY is empty
    )

    # Log a single prediction
    client.log_prediction(
        features={"age": 34, "income": 52000, "credit_score": 720},
        prediction=1,
        score=0.87,
        prediction_id="pred-001",   # optional; enables idempotency + label join
        latency_ms=12.4,
    )

    # Log a batch
    client.log_batch([
        {"features": {"age": 28, "income": 41000}, "prediction": 0, "prediction_id": "pred-002"},
        {"features": {"age": 45, "income": 88000}, "prediction": 1, "prediction_id": "pred-003"},
    ])

    # Send ground-truth labels later
    client.log_ground_truth([
        {"predictionId": "pred-001", "actualValue": 1},
        {"predictionId": "pred-002", "actualValue": 0},
    ])
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Optional


class WatchdogClient:
    """Minimal zero-dependency client for the Watchdog ingestion API."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "",
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout

    # ── Internal ────────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"Watchdog API error {e.code}: {body}") from e

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Public API ───────────────────────────────────────────────────────────

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: float,
        score: Optional[float] = None,
        prediction_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        model_version: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """Log a single prediction. Returns the ingestion result."""
        item: dict[str, Any] = {
            "features": features,
            "prediction": prediction,
            "timestamp": timestamp or self._now_iso(),
        }
        if score is not None:
            item["score"] = score
        if prediction_id is not None:
            item["predictionId"] = prediction_id
        if latency_ms is not None:
            item["latencyMs"] = latency_ms
        if model_version is not None:
            item["modelVersion"] = model_version
        return self._post(
            f"/api/models/{self.model_id}/predictions",
            {"predictions": [item]},
        )

    def log_batch(self, predictions: list[dict[str, Any]]) -> dict:
        """
        Log a batch of predictions.

        Each dict must have ``features`` and ``prediction``; all other fields
        (``predictionId``, ``score``, ``latencyMs``, ``modelVersion``,
        ``timestamp``) are optional.
        """
        now = self._now_iso()
        items = []
        for p in predictions:
            item: dict[str, Any] = {
                "features": p["features"],
                "prediction": p["prediction"],
                "timestamp": p.get("timestamp", now),
            }
            for opt in ("predictionId", "score", "latencyMs", "modelVersion"):
                if opt in p:
                    item[opt] = p[opt]
            items.append(item)
        return self._post(
            f"/api/models/{self.model_id}/predictions",
            {"predictions": items},
        )

    def log_ground_truth(self, labels: list[dict[str, Any]]) -> dict:
        """
        Submit ground-truth labels for previously logged predictions.

        Each dict must have ``predictionId`` and ``actualValue``.
        Optional: ``labeledAt`` (ISO timestamp string).

        Returns ``{"matched": N, "orphaned": M}``.
        """
        return self._post(
            f"/api/models/{self.model_id}/ground-truth",
            {"labels": labels},
        )
