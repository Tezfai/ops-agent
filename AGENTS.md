# Project rules

- This is a simulation. Do not connect refund actions to real payment systems.
- The simulated duplicate refund backend uses FastAPI, PostgreSQL, SQLAlchemy,
  Psycopg 3, and Alembic, without an LLM.
- The backend, not an agent or client, owns refund eligibility and approval.
- Apply each approved refund exactly once, including under retries and
  concurrent requests. Enforce this in the database and test it.
- Equal payment amounts on different invoices are not duplicates. Define and
  test duplicate detection using invoice identity and appropriate payment data.
- Keep credentials in the ignored local `.env`, never in source, logs, or
  commits. Do not expose connection URLs or API keys in error messages.
- Add focused tests for behavior and migrations when implementing the backend.
- Do not add an agent loop, UI, or fine-tuning until those milestones are
  requested.
