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
import socket
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

# --- URL normalisation -------------------------------------------------------
_parts = urlsplit(_DATABASE_URL)

# 1. Percent-encode special characters in the password (e.g. '!' → '%21')
_encoded_password = quote(_parts.password or "", safe="")

# 2. Force IPv4 resolution — some Railway regions (us-west1) cannot route IPv6,
#    so connecting to the raw hostname may pick an AAAA record and fail with
#    "Network is unreachable". sslmode=require does not verify the cert hostname,
#    so substituting the IPv4 address is safe.
try:
    _ipv4 = socket.getaddrinfo(_parts.hostname, None, socket.AF_INET)[0][4][0]
except (socket.gaierror, IndexError):
    _ipv4 = _parts.hostname  # fallback: keep original hostname

_netloc = f"{_parts.username}:{_encoded_password}@{_ipv4}"
if _parts.port:
    _netloc += f":{_parts.port}"

_DATABASE_URL = urlunsplit(_parts._replace(netloc=_netloc))
# -----------------------------------------------------------------------------

# pool_pre_ping keeps connections alive across Railway container restarts.
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
