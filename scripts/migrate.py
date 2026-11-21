"""Apply migrations without displaying database connection details on failure."""

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config


def main() -> int:
    try:
        config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
        command.upgrade(config, "head")
    except Exception as exc:
        print(f"Migration failed ({type(exc).__name__}); check the database configuration and migration state.", file=sys.stderr)
        return 1
    print("Database migrations applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
