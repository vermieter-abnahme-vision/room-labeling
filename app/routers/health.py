from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from app.db import Database
from app.deps import get_database

router = APIRouter(tags=["health"])


@router.get("/healthz", include_in_schema=False)
def healthz(database: Database = Depends(get_database)) -> dict[str, str]:
    try:
        with database.read_session() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="unavailable") from None
    return {"status": "ok"}
