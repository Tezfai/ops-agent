"""Insert deterministic demo data without resetting prior proposals or refunds."""

import sys

from sqlalchemy.orm import Session

from ops_agent.database import create_database_engine
from ops_agent.seed import seed


def main() -> int:
    try:
        engine = create_database_engine()
        try:
            with Session(engine) as session, session.begin():
                seed(session)
        finally:
            engine.dispose()
    except Exception as exc:
        print(f"Seed failed ({type(exc).__name__}); check the database and demo data.", file=sys.stderr)
        return 1
    print("Demo data is present; existing proposals and refunds were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
