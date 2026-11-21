"""Verify the database connection without changing the schema."""

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is missing. Add the direct Neon URL to .env.", file=sys.stderr)
        return 1

    try:
        with psycopg.connect(database_url, connect_timeout=10, autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
        if result != (1,):
            print("Database check failed: SELECT 1 returned an unexpected result.", file=sys.stderr)
            return 1
    except psycopg.Error as exc:
        print(
            f"Database check failed ({type(exc).__name__}). Check the direct Neon URL, "
            "SSL parameters, credentials, branch status, and network access.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"Database check failed ({type(exc).__name__}). Check the database URL "
            "format and local configuration.",
            file=sys.stderr,
        )
        return 1

    print("Database check passed: SELECT 1 returned 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
