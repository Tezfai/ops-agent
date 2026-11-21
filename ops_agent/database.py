import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def create_database_engine(url: str | None = None) -> Engine:
    value = url or os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is not configured.")
    try:
        parsed = make_url(value)
        if parsed.drivername not in ("postgres", "postgresql", "postgresql+psycopg"):
            raise ValueError("Unsupported database driver.")
        engine = create_engine(parsed.set(drivername="postgresql+psycopg"), hide_parameters=True)
    except Exception:
        raise RuntimeError("Invalid database configuration; check DATABASE_URL.") from None

    @event.listens_for(engine, "connect")
    def set_search_path(dbapi_connection, connection_record):
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET search_path TO public")
        dbapi_connection.commit()

    return engine
