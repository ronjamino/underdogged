"""
Apply db/schema.sql against the configured DATABASE_URL.

Usage:
    python -m db.migrate
"""

import os
from pathlib import Path

from sqlalchemy import text

from db.connection import engine, test_connection

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def migrate() -> None:
    print("Testing connection …")
    test_connection()
    print("Connection OK")

    sql = _SCHEMA_PATH.read_text()

    with engine.begin() as conn:
        # Execute each statement individually (psycopg2 doesn't handle
        # multi-statement strings in a single execute call).
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))

    print("Schema applied successfully.")


if __name__ == "__main__":
    migrate()
