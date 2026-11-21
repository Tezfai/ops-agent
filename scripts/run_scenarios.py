"""Run scripted refund scenarios against isolated PostgreSQL schemas."""

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


def approved_duplicate(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID]) -> None:
    proposal = propose(client, payment_ids[1])
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


def separate_invoices(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID]) -> None:
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


def pending_proposal(engine: Engine, client: TestClient, payment_ids: tuple[uuid.UUID, uuid.UUID]) -> None:
    proposal = propose(client, payment_ids[1])
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


SCENARIOS: tuple[tuple[str, bool, Callable[[Engine, TestClient, tuple[uuid.UUID, uuid.UUID]], None]], ...] = (
    ("eligible duplicate approved", True, approved_duplicate),
    ("equal payments on separate invoices", False, separate_invoices),
    ("valid proposal left pending", True, pending_proposal),
)


def main() -> int:
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        print("Set OPS_AGENT_TEST_USE_DEV_DATABASE=1 only for a disposable dev/test database.", file=sys.stderr)
        return 1

    try:
        engine = create_database_engine()
    except Exception as exc:
        print(f"Scenario setup failed ({type(exc).__name__}); check the database configuration.", file=sys.stderr)
        return 1

    failures = 0
    try:
        for name, duplicate, scenario in SCENARIOS:
            try:
                with isolated_database(engine) as isolated:
                    payment_ids = seed_scenario(isolated, duplicate)
                    app.state.session_factory = sessionmaker(isolated, expire_on_commit=False)
                    try:
                        with TestClient(app) as client:
                            verify_initial_state(isolated, client, payment_ids)
                            scenario(isolated, client, payment_ids)
                    finally:
                        del app.state.session_factory
                print(f"PASS: {name}")
            except ScenarioFailure as exc:
                failures += 1
                print(f"FAIL: {name}: {exc}", file=sys.stderr)
            except Exception as exc:
                failures += 1
                print(f"FAIL: {name}: {type(exc).__name__}; check database access and test setup.",
                      file=sys.stderr)
    finally:
        engine.dispose()
    print(f"Scenarios: {len(SCENARIOS) - failures} passed, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
