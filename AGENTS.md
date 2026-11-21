# Project rules

- This is a simulation. Do not connect refund actions to real payment systems.
- The simulated duplicate refund backend uses FastAPI, PostgreSQL, SQLAlchemy,
  Psycopg 3, and Alembic. The hosted agent is separate from the backend.
- The backend, not an agent or client, owns refund eligibility and approval.
- Apply each approved refund exactly once, including under retries and
  concurrent requests. Enforce this in the database and test it.
- Equal payment amounts on different invoices are not duplicates. Define and
  test duplicate detection using invoice identity and appropriate payment data.
- Keep credentials in the ignored local `.env`, never in source, logs, or
  commits. Do not expose connection URLs or API keys in error messages.
- Add focused tests for behavior and migrations when implementing the backend.
- The agent may inspect payments and create proposals, but never approve or
  execute refunds. Approval requires the separate backend endpoint.
- Send only synthetic demo data to hosted providers. Do not send secrets.
- Benchmark held-out families are for final comparison only; never use them
  for training, prompt tuning, or model selection during development.
- Do not add a UI or fine-tuning until those milestones are requested.
