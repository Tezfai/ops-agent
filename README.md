# ops-agent

A portfolio project for a simulated billing support agent that will investigate
duplicate charges, propose refunds, and execute approved refunds. Later work
will fine-tune and benchmark a small open model against a hosted baseline.
The backend simulates refunds without calling payment providers. A separate
hosted agent can inspect simulated payments and create pending proposals.

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

## Hosted agent

The CLI sends its message and the results of its tools to OpenAI. **Use only
synthetic demo data and messages; never send credentials, real customer data,
or other sensitive information.** Requests can incur charges. It uses
`OPENAI_API_KEY` and `OPENAI_MODEL` from the smoke check, makes at most four
model calls with retries disabled, and reports backend-confirmed proposals and
tool failures rather than trusting model claims. `list_payments` and
`propose_refund` are the only tools. A proposal is not an approval: even a
message saying "yes" cannot approve a refund. Approval uses the existing
`POST /refund-proposals/{id}/approve` endpoint separately.

Start the backend as above against a disposable, seeded database, then in a
second PowerShell terminal run:

```powershell
.\.venv\Scripts\python.exe scripts\agent_demo.py "Inspect the simulated payments and propose a refund for an eligible duplicate."
```

The CLI prints a tool trace and confirmed proposal IDs, amounts in cents, and
currencies. If the demo duplicate was already refunded, use fresh disposable
data instead. The backend has no authentication and must remain local.

Run mocked agent tests and all three agent scenarios:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_agent.py
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
.\.venv\Scripts\python.exe scripts\run_scenarios.py --agent mock
```

For a small live OpenAI test using only generated synthetic data in isolated
schemas, run:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
.\.venv\Scripts\python.exe scripts\run_scenarios.py --agent live --scenario pending
.\.venv\Scripts\python.exe scripts\run_scenarios.py --agent live --scenario separate
Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE
```

`--scenario approved` additionally lets the test harness approve the saved
proposal through the backend endpoint after checking there are no refunds.
Each scenario uses and drops its own schema; the normal `public` data is not
modified. The model provider is isolated behind the `Model` interface in
`ops_agent/agent.py`. Frontend and fine-tuning remain later milestones.

## Versioned agent benchmark

`refund-agent-v1.0` adds 24 synthetic cases to the runner while preserving the
three smoke scenarios. Inputs and expected outcomes are separately defined in
`benchmarks/v1.py`; only the request and backend tool results are sent to the
model. Every case creates and drops a fresh migrated schema, including an
already-refunded fixture where appropriate. The grader independently checks
the decision, exact saved proposal ID/payment/amount/status, and the complete
refund state before and after the agent. A clarification must include a
question. Tool order is not graded. The agent has no approval tool.

The split is by scenario family, not a random row split:

- **development** (12 cases): basic duplicates, separate invoices, failed
  retries, and prior refunds. Use these to inspect failures and improve prompts.
- **heldout** (12 cases): mixed invoices, multiple captures, payment mismatches,
  equal timestamps, ambiguous requests, and explicitly pending proposals.
  Reserve these families for final comparison; do not use them in training,
  prompt tuning, or development model selection.

These are evaluation fixtures, not training examples. The deterministic mock
model is a harness check, not a measurement of reasoning ability. With a
disposable `dev` or test database configured and test dependencies installed:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
.\.venv\Scripts\python.exe scripts\run_scenarios.py --benchmark development --agent mock
.\.venv\Scripts\python.exe scripts\run_scenarios.py --benchmark heldout --agent mock
.\.venv\Scripts\python.exe scripts\run_scenarios.py --benchmark development --agent live --case dup_cad_29 --case different_invoices_cad --case failed_then_success --case already_refunded_cad --report benchmark-dev.json
Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE
```

The report includes the benchmark version, split, configured model identifier,
agent configuration hash, per-case result and failure reasons, agent latency,
model-call count, and input/output tokens when reported by the provider. Latency
excludes fixture setup and teardown; missing token counts are `null`. Reports
contain synthetic case identifiers and evaluation results. Live requests incur
charges. Use `--case` repeatedly for a subset, omit it for the full split, and
use `--report` to persist a JSON result. Failed cases produce a nonzero exit.

For a final comparison after development is frozen, run the held-out split for
each provider, keeping each report separately. The default live adapter uses
`OPENAI_MODEL` and the existing OpenAI configuration. To compare a different
provider or a local small model without changing the backend or grader, supply
`--adapter package.module:factory` instead of `--agent`. The zero-argument
factory must return a fresh object with a nonempty `model` identifier and
`next_turn(history) -> ModelTurn` implementing the `Model` protocol in
`ops_agent/agent.py`; an optional `close()` is called after each case. The
adapter must follow the same tool definitions and final decision JSON contract
and should provide token counts in `ModelTurn` where available. For example:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
.\.venv\Scripts\python.exe scripts\run_scenarios.py --benchmark heldout --adapter my_adapter:create_model --report small-heldout.json
Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE
```

`my_adapter` is an example, not an installed model. Adapters are trusted local
code; install or implement one before running that command. Set
`OPENAI_MODEL` to an accessible model to compare different OpenAI-compatible
deployments using the existing adapter. An interrupted run may leave an
`ops_agent_scenario_` schema; neither benchmark split resets `public` data.

### Untuned local small-model adapter

The adapter `ops_agent.local_model:create_model` connects to an **already
running, loopback-only**, OpenAI-compatible chat-completions server. Select and
install an untuned open-weight instruct model and a server that implements
`/v1/chat/completions`, function calling, tool-result messages and a JSON final
decision. Model compatibility has not been established until the smoke test
passes. Use only a model and server you trust; no model is downloaded or
infrastructure provisioned by this project.

Set `SMALL_MODEL` to the server's model identifier, `SMALL_MODEL_REVISION` to
the exact model release or immutable weight revision you installed, and
`SMALL_MODEL_BASE_URL` to the server's local API base. The revision is
user-supplied metadata, not verified against the model weights. The endpoint
must use HTTP on `127.0.0.1`, `localhost`, or `::1`, without credentials or
query parameters. The adapter sends a placeholder API key only to the local
server, disables proxy environment configuration and retries, and uses a
30-second timeout. The per-call response limit defaults to 500 tokens; set
`SMALL_MODEL_MAX_TOKENS` to a positive integer to override it. Reports record
the configured limit.
Optionally set `SMALL_MODEL_REASONING_EFFORT=none` to send
`reasoning_effort: "none"` to a compatible local server; when unset, the field
is omitted. The setting is recorded in benchmark reports. A successful response
does not prove that the server honored the field.
To test structured final output separately, set `SMALL_MODEL_DECISION_SCHEMA=1`;
after a tool result the adapter requests the benchmark's decision JSON schema,
while keeping tools available. This opt-in changes the decoding format and is
recorded in benchmark reports; it is not directly comparable to prompt-only
runs. When unset, no response format is requested.

The local adapter sends the same system instructions, tool descriptions and
argument schemas as the hosted model. The hosted Responses API requests strict
JSON schema output, while this chat adapter relies on the shared prompt for
JSON (no structured-output enforcement); the report records this adapter
difference. Other server-side defaults, quantization and sampling settings are
determined by your server and should be recorded with your comparison. The
exact immutable version behind a hosted model alias is not guaranteed by the
alias alone. New reports record model identifiers returned by each provider
when available; a server-reported identifier is not proof of the weights.

After confirming the database is disposable, set the three values locally
(replace the placeholders) and run the tool smoke case before the full
development split:

```powershell
$env:SMALL_MODEL_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:SMALL_MODEL = '<server-model-id>'
$env:SMALL_MODEL_REVISION = '<exact-model-release-or-weight-revision>'
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
.\.venv\Scripts\python.exe scripts\run_scenarios.py --benchmark development --adapter ops_agent.local_model:create_model --case different_invoices_cad --require-list-payments --report small-smoke.json
if ($LASTEXITCODE -eq 0) { .\.venv\Scripts\python.exe scripts\run_scenarios.py --benchmark development --adapter ops_agent.local_model:create_model --report small-dev.json }
Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE
```

The smoke check requires a successful `list_payments` call and the correct
decision for those payment records; it does not prove how the model reasoned.
The benchmark grader remains the same. Reports separate decision-format,
tool, business-outcome and model failures. They also record the supplied model
revision and adapter settings, but not the endpoint or credentials. Keep
held-out cases untouched until the final comparison.

## Synthetic supervised demonstrations

`scripts/generate_sft_dataset.py` generates and replays
`synthetic-refund-sft-v1.0`, without calling a model or starting training.
Each JSONL record contains synthetic metadata, the available tool definitions,
and system, user, assistant tool-call, actual tool-result, and final decision
messages. The gateway calls the existing backend. Replay checks the saved
proposal, its payment, amount, currency and pending status against PostgreSQL,
and verifies that the complete refund state does not change. Prior-refund
examples start with one historical refund; no new refund is executed.

The split is by underlying template, not by individual variant: the training
split covers eligible duplicates and failed retries; validation covers equal
payments on separate invoices and previously refunded duplicates. This avoids
near-identical layouts across splits but means validation does not cover every
training family. Ambiguous requests and other held-out benchmark families are
excluded from both splits, so there are no clarification demonstrations. Both
benchmark splits remain evaluation-only. The seed fixes synthetic inputs and
IDs, but backend-generated proposal IDs change on each run.

Confirm `DATABASE_URL` (including any overriding process variable) points to
the disposable Neon `dev` branch or a separate disposable test database.
The generator creates and removes an isolated migrated schema; it resets only
that schema between traces and does not reset normal `public` data. A crash
could leave an isolated schema behind. Output files are never overwritten.
To regenerate, choose a fresh output directory:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
try {
    .\.venv\Scripts\python.exe scripts\generate_sft_dataset.py --seed 20261121 --output data\synthetic-refund-sft-v1.0-rerun
} finally {
    Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE -ErrorAction SilentlyContinue
}
```

The output includes training and validation JSONL files and a summary with
accepted/rejected counts and reasons. A nonzero exit indicates rejection or
generation failure; inspect the summary before using a partial dataset. Future
benchmark reports record the structurally validated decision enum and tool
sequence, not model thinking; the existing database grader remains unchanged.

### Balanced v1.1 dataset

`synthetic-refund-sft-v1.1` provides five cases from each of four allowed
families in each split: eligible duplicate, separate invoices, failed retry,
and already refunded. Each split has five `propose` and fifteen `no_refund`
decisions. Split-specific templates are assigned before variants are generated.
Training identifies the duplicate by capture order, the separate payments by
invoice IDs, and explicitly discloses a failure or previous refund. Validation
identifies payments by ID and requires the tool results to reveal the failure,
previous refund, or distinct invoices. These differences do not by themselves
prove semantic independence. Clarification and reserved benchmark families are
excluded. No benchmark examples are read by this generator.

The generator uses the existing gateway and replays each saved demonstration
against an isolated schema. It compares pre/post proposals and refunds; the
prior-refund family starts with a historical refund. After writing, it audits
the JSONL for family and decision balance, tool/result pairing, expected fields,
configured credentials, and normalized cross-split copies. Grading metadata is
kept in record metadata, not model messages. A failed replay or export audit
causes a nonzero exit; do not use that output as a validated dataset.

Confirm the effective `DATABASE_URL` is the disposable Neon `dev` branch or a
separate disposable test database before running:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
try {
    .\.venv\Scripts\python.exe scripts\generate_sft_v11.py --seed 20261121 --output data\synthetic-refund-sft-v1.1-rerun
} finally {
    Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE -ErrorAction SilentlyContinue
}
```

Choose a directory that does not exist: the generator refuses to overwrite an
existing directory. With the same seed, scenario IDs, payment and invoice
IDs, amounts, currency, dates, and requests are reproducible; proposal IDs are
created by the backend and differ across runs. Schema names are also random and
are not exported. A crash may leave an isolated schema.

### Frozen Qwen3-4B LoRA pilot

The pilot configuration is frozen in `training/pilot_config.json`. It pins
`Qwen/Qwen3-4B` and its tokenizer to upstream revision
`1cfa9a7208912126459214e8b04321603b3df60c`, not the separate Ollama serving
artifact revision. The configuration also pins the two v1.1 export hashes,
seed, LoRA settings, training settings, and greedy inference token cap.
The pilot refuses changed exports or an existing output directory.

An earlier dataset task ran `tests/test_benchmark.py`, which imported and
accessed held-out benchmark definitions. That was a boundary violation, not
a held-out model evaluation. This pilot must not import benchmark modules,
run that test, or collect the full test suite. Its explicitly named training
tests use synthetic mini-fixtures.

For this unquantized BF16 configuration, use an NVIDIA CUDA GPU with BF16
support and at least 20 GiB **free** VRAM; more may be necessary. The current
8 GiB GPU does not satisfy this configuration. No paid resources are
provisioned. Create a separate Python 3.11 environment and install dependencies
on the suitable GPU machine (the CUDA wheel index below is for CUDA 12.4):

```powershell
py -3.11 -m venv .venv-sft
.\.venv-sft\Scripts\python.exe -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
.\.venv-sft\Scripts\python.exe -m pip install -r training\requirements.txt -e '.[test]'
.\.venv-sft\Scripts\python.exe -m training.pilot inspect
.\.venv-sft\Scripts\python.exe -m pytest -q tests\test_training_pilot.py
```

`inspect` downloads tokenizer files only, not model weights. It renders all
training traces with the pinned chat template and reports supervised tokens
for one positive and one negative example. Each assistant turn is a separate
teacher-forced example: prior system, user, and tool-result tokens are masked;
assistant tool calls and final decisions are supervised. Padding is masked.
Overlong examples fail rather than truncate. The example context includes
tools and messages only, never export grading metadata.

After confirming the effective database URL points to a disposable development
or test database, run the prepared pilot:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
try {
    .\.venv-sft\Scripts\python.exe -m training.pilot all --output pilot-runs\qwen3-4b-v1.1
} finally {
    Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE -ErrorAction SilentlyContinue
}
```

This downloads the pinned base weights, trains only on the 20 training
records, saves the adapter separately, and evaluates the base (adapter
disabled) then the tuned model with the same deterministic Transformers
backend, chat template, tools, BF16 precision, and decoding settings.
Evaluation seeds an isolated schema per case state and feeds only its initial
user request to the model. The JSON comparison reports outcomes, errors,
regressions, and newly executed refunds; no approval endpoint is called.
Do not use the historical Ollama benchmark as a matched baseline. This is a
small validation set, not evidence of general improvement.

### RTX 3070 4-bit pilot

The separate `training/qlora_config.json` retains the pinned BF16 pilot's
checkpoint, dataset hashes, seed, LoRA hyperparameters, epochs, learning rate,
and completion cap. It uses NF4 double quantization, FP16 compute, frozen
4-bit weights, LoRA-only training, microbatch 1, gradient accumulation 4,
effective batch size 4 assistant turns, gradient checkpointing, and paged
8-bit AdamW. It trains the existing 45 assistant turns from 20 traces.
The evaluation uses the same quantized base with its adapter disabled/enabled
and reuses the existing agent and state grader. Output settings record installed
versions and GPU. No approval tool or endpoint is used.

On Windows with a CUDA-compatible NVIDIA driver, the exact setup and commands:

```powershell
py -3.11 -m venv .venv-qlora
.\.venv-qlora\Scripts\python.exe -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
.\.venv-qlora\Scripts\python.exe -m pip install -r training\requirements-qlora.txt -e '.[test]'
.\.venv-qlora\Scripts\python.exe -m training.check_4bit
.\.venv-qlora\Scripts\python.exe -m pytest -q tests\test_training_pilot.py
```

Confirm the effective database URL is a disposable dev/test database before
the full pilot. Set its opt-in only for that command:

```powershell
$env:OPS_AGENT_TEST_USE_DEV_DATABASE = '1'
try {
    .\.venv-qlora\Scripts\python.exe -m training.qlora --output pilot-runs\qwen3-4b-qlora-v1.1-rerun
} finally {
    Remove-Item Env:\OPS_AGENT_TEST_USE_DEV_DATABASE -ErrorAction SilentlyContinue
}
```

The output directory must not exist. Before training, the runner resets a
fresh adapter after a longest-training-turn optimizer step and separately
forces 1,024 generated tokens from a synthetic tool-response context. It records
preflight peak memory and stops on failure. Even a passing preflight does not
guarantee sufficient VRAM throughout training or evaluation when GPU memory is
shared. `failure.json` records a failed stage without connection details.
