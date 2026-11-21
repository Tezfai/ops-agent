"""Run scripted refund scenarios against isolated PostgreSQL schemas."""

import argparse
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import event, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ops_agent.agent import Model, ModelFailure, ModelTurn, OpenAIModel, ToolCall, run_agent
from ops_agent.database import create_database_engine
from ops_agent.demo import DEMO_CUSTOMER_ID, demo_id
from ops_agent.main import app
from ops_agent.models import Customer, Invoice, Payment, Refund, RefundProposal

ROOT = Path(__file__).resolve().parent.parent
AMOUNT_CENTS = 2900
CURRENCY = "CAD"


class ScenarioFailure(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioFailure(message)


@contextmanager
def isolated_database(engine: Engine) -> Generator[Engine, None, None]:
    schema = f"ops_agent_scenario_{uuid.uuid4().hex}"
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    isolated = engine.execution_options()

    def set_search_path(dbapi_connection, _connection_record, _connection_proxy):
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema}"')
        dbapi_connection.commit()

    listener_installed = False
    try:
        event.listen(isolated, "checkout", set_search_path)
        listener_installed = True
        config = Config(str(ROOT / "alembic.ini"))
        with isolated.connect() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield isolated
    finally:
        if listener_installed:
            event.remove(isolated, "checkout", set_search_path)
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


def seed_scenario(engine: Engine, duplicate: bool) -> tuple[uuid.UUID, uuid.UUID]:
    first_invoice = demo_id("scenario-invoice-a")
    second_invoice = first_invoice if duplicate else demo_id("scenario-invoice-b")
    first_payment = demo_id("scenario-payment-a")
    second_payment = demo_id("scenario-payment-b")
    with Session(engine) as session, session.begin():
        session.add(Customer(id=DEMO_CUSTOMER_ID, name="Scenario customer"))
        session.flush()
        for invoice_id in {first_invoice, second_invoice}:
            session.add(Invoice(id=invoice_id, customer_id=DEMO_CUSTOMER_ID,
                                amount_cents=AMOUNT_CENTS, currency=CURRENCY))
        session.flush()
        for payment_id, invoice_id, day in (
            (first_payment, first_invoice, 1),
            (second_payment, second_invoice, 2),
        ):
            session.add(Payment(id=payment_id, invoice_id=invoice_id, amount_cents=AMOUNT_CENTS,
                                currency=CURRENCY, payment_method="scenario-card", status="captured",
                                captured_at=datetime(2026, 1, day, 12, tzinfo=timezone.utc)))
    return first_payment, second_payment


def verify_initial_state(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID]) -> None:
    with Session(engine) as session:
        require(session.scalars(select(Refund)).all() == [], "Initial refund state was not empty.")
        require(session.scalars(select(RefundProposal)).all() == [], "Initial proposal state was not empty.")
    response = client.get("/payments")
    require(response.status_code == 200, "Could not inspect scenario payments.")
    payments = response.json()
    require({payment["id"] for payment in payments} == {str(payment_id) for payment_id in payment_ids}
            and len(payments) == 2, "Scenario payments did not match the seed.")


def propose(client: TestClient, payment_id: uuid.UUID) -> dict:
    response = client.post("/refund-proposals", json={"payment_id": str(payment_id)})
    require(response.status_code == 201, "Eligible proposal was not created.")
    proposal = response.json()
    require(proposal["payment_id"] == str(payment_id) and proposal["status"] == "pending",
            "Created proposal did not match the expected pending payment.")
    return proposal


def agent_propose(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID],
                  model: Model) -> dict:
    result = run_agent(model, client, "Inspect the simulated payments and propose a refund for an eligible duplicate.")
    require(result.error is None, f"Agent failed: {result.error}")
    require(len(result.proposals) == 1, "Agent did not create exactly one proposal.")
    proposal = result.proposals[0]
    require(proposal["payment_id"] == str(payment_ids[1]) and proposal["status"] == "pending"
            and proposal["amount_cents"] == AMOUNT_CENTS and proposal["currency"] == CURRENCY,
            "Agent proposal did not match the duplicate.")
    with Session(engine) as session:
        proposals = session.scalars(select(RefundProposal)).all()
        require(len(proposals) == 1 and str(proposals[0].id) == proposal["id"]
                and proposals[0].payment_id == payment_ids[1] and proposals[0].status == "pending",
                "The proposed refund was not saved as pending.")
        require(session.scalars(select(Refund)).all() == [], "Agent created a refund before approval.")
    print("TRACE: " + " -> ".join(event["tool"] for event in result.trace)
          + f" -> pending proposal {proposal['id']}, {proposal['amount_cents']} {proposal['currency']} cents")
    return proposal


def approved_duplicate(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID],
                       model: Model | None = None) -> None:
    proposal = agent_propose(engine, client, payment_ids, model) if model else propose(client, payment_ids[1])
    response = client.post(f"/refund-proposals/{proposal['id']}/approve")
    require(response.status_code == 200, "Duplicate approval failed.")
    result = response.json()
    require(result["payment_id"] == str(payment_ids[1]) and result["proposal_id"] == proposal["id"]
            and result["amount_cents"] == AMOUNT_CENTS and result["currency"] == CURRENCY,
            "Approval response did not match the expected refund.")
    with Session(engine) as session:
        refunds = session.scalars(select(Refund)).all()
        proposals = session.scalars(select(RefundProposal)).all()
        require(len(refunds) == 1, "Expected exactly one refund.")
        refund = refunds[0]
        require(refund.payment_id == payment_ids[1] and refund.amount_cents == AMOUNT_CENTS
                and refund.currency == CURRENCY and str(refund.proposal_id) == proposal["id"]
                and str(refund.id) == result["id"], "Stored refund did not match the approved duplicate.")
        require(len(proposals) == 1 and proposals[0].status == "approved",
                "Expected one approved proposal.")


def separate_invoices(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID],
                      model: Model | None = None) -> None:
    if model:
        result = run_agent(model, client, "Inspect the simulated payments and propose a refund only for an eligible duplicate.")
        require(result.error is None, f"Agent failed: {result.error}")
        require(len(result.proposals) == 0, "Agent proposed a refund for separate invoices.")
        require(any(event["tool"] == "list_payments" for event in result.trace), "Agent did not inspect payments.")
        print("TRACE: " + " -> ".join(event["tool"] for event in result.trace) + " -> no proposal")
    else:
        for payment_id in payment_ids:
            response = client.post("/refund-proposals", json={"payment_id": str(payment_id)})
            require(response.status_code == 409, "Separate-invoice payment was not rejected.")
    with Session(engine) as session:
        payments = session.scalars(select(Payment)).all()
        require(len(payments) == 2 and {payment.id for payment in payments} == set(payment_ids)
                and len({payment.invoice_id for payment in payments}) == 2
                and all(payment.amount_cents == AMOUNT_CENTS and payment.currency == CURRENCY
                        for payment in payments), "Separate-invoice payment data did not match.")
        require(session.scalars(select(Refund)).all() == [], "Expected zero refunds.")
        require(session.scalars(select(RefundProposal)).all() == [], "Expected zero proposals.")


def pending_proposal(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID],
                     model: Model | None = None) -> None:
    proposal = agent_propose(engine, client, payment_ids, model) if model else propose(client, payment_ids[1])
    if isinstance(model, MockScenarioModel):
        class AttemptedApproval:
            def next_turn(self, history: list[dict]) -> ModelTurn:
                return ModelTurn([ToolCall("approval", "approve_refund", json.dumps({"proposal_id": proposal["id"]}))])

        result = run_agent(AttemptedApproval(), client, "yes")
        require(result.error is not None and not result.proposals, "Unexpected approval tool was not rejected.")
    with Session(engine) as session:
        refunds = session.scalars(select(Refund)).all()
        proposals = session.scalars(select(RefundProposal)).all()
        require(len(refunds) == 0, "Unapproved proposal produced a refund.")
        require(len(proposals) == 1 and str(proposals[0].id) == proposal["id"]
                and proposals[0].payment_id == payment_ids[1] and proposals[0].status == "pending",
                "Expected the exact proposal to remain pending.")
        payment = session.get(Payment, payment_ids[1])
        require(payment is not None and payment.amount_cents == AMOUNT_CENTS and payment.currency == CURRENCY,
                "Pending proposal payment did not match.")


SCENARIOS: tuple[tuple[str, bool, Callable[[Engine, TestClient, tuple[uuid.UUID, uuid.UUID], Model | None], None]], ...] = (
    ("eligible duplicate approved", True, approved_duplicate),
    ("equal payments on separate invoices", False, separate_invoices),
    ("valid proposal left pending", True, pending_proposal),
)


class MockScenarioModel:
    def __init__(self, duplicate: bool):
        self.duplicate = duplicate
        self.step = 0

    def next_turn(self, history: list[dict]) -> ModelTurn:
        self.step += 1
        if self.step == 1:
            return ModelTurn([ToolCall("list", "list_payments", "{}")])
        if self.step == 2 and self.duplicate:
            return ModelTurn([ToolCall("proposal", "propose_refund",
                                       json.dumps({"payment_id": str(demo_id("scenario-payment-b"))}))])
        return ModelTurn([], "Finished.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated backend and optional agent scenarios.")
    parser.add_argument("--agent", choices=("mock", "live"), help="Exercise the agent with a mocked or live model.")
    parser.add_argument("--scenario", choices=("approved", "separate", "pending"), help="Run one selected scenario.")
    parser.add_argument("--benchmark", choices=("development", "heldout"), help="Run a versioned benchmark split.")
    parser.add_argument("--case", action="append", default=[], help="Benchmark case ID; may be repeated.")
    parser.add_argument("--adapter", help="Custom model adapter factory as module:factory.")
    parser.add_argument("--report", help="Write benchmark results as JSON to this path.")
    parser.add_argument("--require-list-payments", action="store_true",
                        help="Require a successful single-case smoke run to call list_payments.")
    args = parser.parse_args()
    if args.benchmark:
        if args.scenario or (not args.agent and not args.adapter) or (args.adapter and args.agent):
            parser.error("Benchmark requires --agent mock/live or --adapter, and cannot use --scenario.")
    elif args.case or args.adapter or args.report or args.require_list_payments:
        parser.error("--case, --adapter, --report, and --require-list-payments require --benchmark.")
    if args.require_list_payments and len(args.case) != 1:
        parser.error("--require-list-payments requires exactly one --case.")
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        print("Set OPS_AGENT_TEST_USE_DEV_DATABASE=1 only for a disposable dev/test database.", file=sys.stderr)
        return 1

    try:
        engine = create_database_engine()
    except Exception as exc:
        print(f"Scenario setup failed ({type(exc).__name__}); check the database configuration.", file=sys.stderr)
        return 1

    if args.benchmark:
        from benchmarks.runner import run_benchmark

        try:
            report = run_benchmark(engine, args.benchmark, args.case, args.agent or "custom",
                                   isolated_database, args.adapter)
            print(f"Benchmark {report['benchmark_version']} | split={report['split']} | model={report['model']}")
            print(f"Results: {report['passed']}/{report['total']} passed.")
            if args.require_list_payments:
                used = "list_payments" in report["cases"][0].get("tools_used", [])
                smoke_passed = used and report["passed"] == 1
                report["list_payments_smoke_passed"] = smoke_passed
                print("Tool smoke: " + ("PASS" if smoke_passed else "FAIL: correct decision and list_payments required."))
            if args.report:
                Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return 0 if report["passed"] == report["total"] and (
                not args.require_list_payments or report["list_payments_smoke_passed"]) else 1
        except (ValueError, ImportError, AttributeError) as exc:
            print(f"Benchmark configuration failed ({type(exc).__name__}).", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"Benchmark report could not be written ({type(exc).__name__}).", file=sys.stderr)
            return 1
        finally:
            engine.dispose()

    chosen = SCENARIOS if args.scenario is None else (SCENARIOS[{"approved": 0, "separate": 1, "pending": 2}[args.scenario]],)
    failures = 0
    try:
        for name, duplicate, scenario in chosen:
            try:
                with isolated_database(engine) as isolated:
                    payment_ids = seed_scenario(isolated, duplicate)
                    app.state.session_factory = sessionmaker(isolated, expire_on_commit=False)
                    try:
                        with TestClient(app) as client:
                            verify_initial_state(isolated, client, payment_ids)
                            model = OpenAIModel() if args.agent == "live" else MockScenarioModel(duplicate) if args.agent == "mock" else None
                            try:
                                scenario(isolated, client, payment_ids, model)
                            finally:
                                if isinstance(model, OpenAIModel):
                                    model.close()
                    finally:
                        del app.state.session_factory
                print(f"PASS: {name}")
            except (ScenarioFailure, ModelFailure) as exc:
                failures += 1
                print(f"FAIL: {name}: {exc}", file=sys.stderr)
            except Exception as exc:
                failures += 1
                print(f"FAIL: {name}: {type(exc).__name__}; check database access and test setup.",
                      file=sys.stderr)
    finally:
        engine.dispose()
    print(f"Scenarios: {len(chosen) - failures} passed, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
