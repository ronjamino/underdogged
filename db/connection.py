"""
Database connection utilities.

Usage:
    from db.connection import engine, SessionLocal, get_db

    # FastAPI dependency
    def my_route(db: Session = Depends(get_db)):
        ...

    # Script / one-off
    with SessionLocal() as session:
        result = session.execute(text("SELECT 1")).scalar()
"""

import os
from typing import Generator

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

# Percent-encode special characters in the password portion of the URL
# (e.g. '!' → '%21') so psycopg2 parses the URL correctly.
from urllib.parse import urlsplit, urlunsplit, quote
_parts = urlsplit(_DATABASE_URL)
if _parts.password:
    _encoded_password = quote(_parts.password, safe="")
    _netloc = f"{_parts.username}:{_encoded_password}@{_parts.hostname}"
    if _parts.port:
        _netloc += f":{_parts.port}"
    _DATABASE_URL = urlunsplit(_parts._replace(netloc=_netloc))

# pool_pre_ping keeps connections alive across Railway container restarts.
# sslmode=require is passed via connect_args for Supabase compatibility.
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
