"""Shared FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.config import get_settings


def require_ingest_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Enforce ``INGEST_API_KEY`` when configured.

    Empty key (local/dev default) leaves ingest open. Production should set
    ``INGEST_API_KEY`` so prediction/baseline uploads require ``X-API-Key``.
    """
    expected = get_settings().INGEST_API_KEY
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
