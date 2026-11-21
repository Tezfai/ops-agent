# ops-agent

A portfolio project for a simulated billing support agent that will investigate
duplicate charges, propose refunds, and execute approved refunds. Later work
will fine-tune and benchmark a small open model against a hosted baseline.
The first milestone implements a simulated refund backend; it does not call
payment providers or OpenAI.

## Setup (Windows PowerShell)

From the project directory, using Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
```

The local `.env` is ignored by Git. If it does not exist, first verify that
`git check-ignore .env` reports `.env`, then create it:

```powershell
git check-ignore .env
Copy-Item .env.example .env
```

Do not overwrite an existing `.env` or paste credentials into chat.

1. In the [Neon console](https://console.neon.tech/), create an `ops-agent`
   project and a disposable `dev` branch. Under **Connect**, select `dev` and
   copy its **direct** PostgreSQL URL, including SSL parameters, into
   `DATABASE_URL` in `.env`.
2. In [OpenAI Platform](https://platform.openai.com/settings/organization/projects),
   create or select an API project with access to the model. Create a project
   API key and put it in `OPENAI_API_KEY` in `.env`.
3. `OPENAI_MODEL` defaults to `gpt-4.1-mini`. If unavailable to your project,
   set it in `.env` to a supported tool-capable model accessible to that
   project. If reusing an Azure deployment instead, the configuration and
   model check must be adapted for Azure rather than requiring an OpenAI key.

Existing process environment variables take precedence over `.env`. The model
check uses the OpenAI API and makes one small request, which may incur charges.

## Smoke checks

Before running the checks, verify `.env` is ignored and not tracked, and that
`.env.example` is not ignored:

```powershell
git check-ignore .env
git ls-files --error-unmatch .env
git check-ignore .env.example
```

The first command should print `.env`. The last two should exit nonzero: a
successful `git ls-files` means the secret file is tracked and must be dealt
with before proceeding; a successful `git check-ignore .env.example` means
the template is ignored.

Run each check independently:

```powershell
.\.venv\Scripts\python.exe scripts\check_db.py
.\.venv\Scripts\python.exe scripts\check_model.py
```

The database check executes only `SELECT 1`; it does not create or modify
tables. The model check verifies a response without printing its contents.
Both checks avoid printing credentials or raw service error messages.

## Simulated refund workflow

Only use a disposable Neon `dev` branch or a separate test database for this
project. The commands below **write to the `public` schema of the configured
database**. Confirm that
`DATABASE_URL` in `.env` points to that disposable target before running them.
Environment variables take precedence, so verify that any `DATABASE_URL` in
your PowerShell session also targets the disposable database. Do not use
production data.

```powershell
.\.venv\Scripts\python.exe scripts\migrate.py
.\.venv\Scripts\python.exe scripts\seed.py
.\.venv\Scripts\python.exe -m uvicorn ops_agent.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal:

```powershell
$payments = Invoke-RestMethod http://127.0.0.1:8000/payments
$payments | Format-Table id, invoice_id, amount_cents, currency, captured_at
$duplicate = @($payments | Group-Object invoice_id | Where-Object { $_.Count -eq 2 } | ForEach-Object { $_.Group | Sort-Object captured_at | Select-Object -Last 1 })[0]
$body = @{ payment_id = $duplicate.id } | ConvertTo-Json
$proposal = Invoke-RestMethod http://127.0.0.1:8000/refund-proposals -Method Post -ContentType 'application/json' -Body $body
Invoke-RestMethod http://127.0.0.1:8000/refunds
$refund = Invoke-RestMethod "http://127.0.0.1:8000/refund-proposals/$($proposal.id)/approve" -Method Post
$refund
Invoke-RestMethod "http://127.0.0.1:8000/refund-proposals/$($proposal.id)/approve" -Method Post
Invoke-RestMethod http://127.0.0.1:8000/refunds
```

The empty list before approval and single CAD 2900-cent refund after approval
demonstrate the workflow. The replay returns the same refund. `GET /payments`,
`GET /refund-proposals`, and `GET /refunds` expose only the fixed demo customer.
This is a local simulation: there is no authentication or real payment
integration. Do not expose the server to untrusted networks.

The seed inserts one CAD 29 invoice with two captures and two separate CAD 29
invoices with one legitimate capture each. It is repeatable and preserves
existing proposals and refunds; it does **not** reset an approved scenario.
For a previously approved demo, inspect `GET /refund-proposals` to recover the
proposal ID and replay its approval, or use a fresh disposable branch.

The backend considers a capture a duplicate only if it is captured later than
another captured payment on the **same invoice** with the same full invoice
amount, currency, and simulated payment-method identifier. Equal timestamps
are not ordered as duplicates. Approval rechecks eligibility, locks the payment
and commits the proposal status and simulated full refund together. A database
unique constraint on the refunded payment prevents multiple full refunds,
including competing proposals.

## Integration tests

Tests create and later drop a randomly named schema in the configured
database, run migration upgrade/downgrade, and exercise the API and concurrent
approvals. Set the opt-in only after confirming the database is disposable:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
.\.venv\Scripts\python.exe -m pytest -q --tb=no tests\test_integration.py
Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE
```

The opt-in is a safety reminder, not verification of the database target.
The schema is dropped after the tests, even when a test fails; an interrupted
process may require manual cleanup of its `ops_agent_test_` schema. The tests
do not use or alter the demo data in the default schema.

## Scenario runner

After confirming the configured database is a disposable Neon `dev` branch or
separate test database, run:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'; .\.venv\Scripts\python.exe scripts\run_scenarios.py
```

The runner uses FastAPI's in-process test client, not a running web server. It
creates and migrates a fresh randomly named PostgreSQL schema for each scenario,
seeds isolated data, drives the existing API, and checks final database state.
The three scenarios cover an approved duplicate, equal payments on separate
invoices, and an unapproved proposal. Each schema is dropped after its scenario,
including on failure; an interrupted process may leave an `ops_agent_scenario_`
schema requiring manual cleanup. Normal `public` schema data is not reset or
changed. The opt-in is a reminder, not verification of the database target.
The runner requires the test dependencies installed in the setup above.

OpenAI integration, an agent loop, UI, and fine-tuning are later milestones.
