"""
Database connection utilities.
"""

import os
from typing import Generator
from urllib.parse import urlsplit, urlunsplit, quote

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

_DATABASE_URL = os.getenv("DATABASE_URL")

if not _DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Add it in Railway → your service → Variables."
    )

# Percent-encode special characters in the password (e.g. '!' → '%21')
_parts = urlsplit(_DATABASE_URL)
_encoded_password = quote(_parts.password or "", safe="")
_netloc = f"{_parts.username}:{_encoded_password}@{_parts.hostname}"
if _parts.port:
    _netloc += f":{_parts.port}"
_DATABASE_URL = urlunsplit(_parts._replace(netloc=_netloc))

engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"sslmode": "require"},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a Session, closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """Returns True if a SELECT 1 succeeds; raises on failure."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
