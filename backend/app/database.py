from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import get_settings

settings = get_settings()

import json
import sqlite3
import numpy as np

sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.float32, float)

def _json_dumps(obj):
    def default(o):
        if isinstance(o, (np.int64, np.int32, np.int16, np.int8)):
            return int(o)
        if isinstance(o, (np.float64, np.float32, np.float16)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return json.dumps(obj, default=default)

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine_kwargs = {
    "connect_args": connect_args,
    "json_serializer": _json_dumps,
    "echo": False,
    "pool_pre_ping": True,
}
if not settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["pool_size"] = 5
    engine_kwargs["max_overflow"] = 10

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# Import all ORM models so the metadata is populated before tests or startup
# call Base.metadata.create_all().
import app.models  # noqa: F401


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
