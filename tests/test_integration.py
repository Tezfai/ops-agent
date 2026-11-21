import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import event, func, inspect, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ops_agent.database import create_database_engine
from ops_agent.demo import demo_id
from ops_agent.main import app
from ops_agent.models import Customer, Invoice, Payment, Refund, RefundProposal
from ops_agent.seed import seed

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def database():
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        pytest.skip("Set OPS_AGENT_TEST_USE_DEV_DATABASE=1 only for a disposable dev/test database.")
    engine = create_database_engine()
    schema = f"ops_agent_test_{uuid.uuid4().hex}"
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    test_engine = engine.execution_options(schema_translate_map=None)

    # Each new session selects the isolated test schema, not the database's public schema.
    @event.listens_for(test_engine, "checkout")
    def set_search_path(dbapi_connection, connection_record, connection_proxy):
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema}"')
        dbapi_connection.commit()

    config = Config(str(ROOT / "alembic.ini"))
    try:
        with test_engine.connect() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield test_engine, config
    finally:
        event.remove(test_engine, "checkout", set_search_path)
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


@pytest.fixture(autouse=True)
def clear_data(database):
    engine, _ = database
    with Session(engine) as session, session.begin():
        for model in (Refund, RefundProposal, Payment, Invoice, Customer):
            session.query(model).delete()
        seed(session)


@pytest.fixture
def client(database):
    engine, _ = database
    app.state.session_factory = sessionmaker(engine, expire_on_commit=False)
    with TestClient(app) as test_client:
        yield test_client
    del app.state.session_factory


def refund_count(engine):
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(Refund))


def create_proposal(client, payment):
    return client.post("/refund-proposals", json={"payment_id": str(payment)})


def test_migration_and_repeatable_seed(database):
    engine, config = database
    with engine.connect() as connection:
        assert {"customers", "invoices", "payments", "refund_proposals", "refunds"}.issubset(
            inspect(connection).get_table_names()
        )
    with Session(engine) as session, session.begin():
        seed(session)
        assert session.scalar(select(func.count()).select_from(Payment)) == 4
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
    with engine.connect() as connection:
        assert "payments" not in inspect(connection).get_table_names()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


def test_end_to_end_and_replay(database, client):
    engine, _ = database
    payments = client.get("/payments")
    assert payments.status_code == 200
    assert len(payments.json()) == 4
    duplicate = demo_id("duplicate-payment")
    proposal = create_proposal(client, duplicate)
    assert proposal.status_code == 201
    assert proposal.json()["status"] == "pending"
    assert refund_count(engine) == 0
    assert client.get("/refunds").json() == []
    url = f"/refund-proposals/{proposal.json()['id']}/approve"
    approval = client.post(url)
    assert approval.status_code == 200
    assert approval.json()["payment_id"] == str(duplicate)
    assert approval.json()["amount_cents"] == 2900
    assert approval.json()["currency"] == "CAD"
    assert refund_count(engine) == 1
    assert client.get("/refunds").json() == [approval.json()]
    assert client.get("/refund-proposals").json()[0]["status"] == "approved"
    with Session(engine) as session, session.begin():
        seed(session)
    assert refund_count(engine) == 1
    replay = client.post(url)
    assert replay.status_code == 200
    assert replay.json() == approval.json()
    assert refund_count(engine) == 1


def test_separate_invoices_and_original_are_ineligible(database, client):
    engine, _ = database
    for name in ("original-payment", "legitimate-payment-a", "legitimate-payment-b"):
        assert create_proposal(client, demo_id(name)).status_code == 409
    assert refund_count(engine) == 0


def test_distinct_proposals_and_concurrent_approval(database, client):
    engine, _ = database
    proposal_ids = [create_proposal(client, demo_id("duplicate-payment")).json()["id"] for _ in range(2)]
    barrier = Barrier(2)

    def approve(proposal_id):
        with TestClient(app) as concurrent_client:
            barrier.wait(timeout=10)
            response = concurrent_client.post(f"/refund-proposals/{proposal_id}/approve")
            return response.status_code, response.json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(approve, proposal_ids))
    assert sorted(code for code, _ in results) == [200, 409]
    assert refund_count(engine) == 1
    winner = next(body for code, body in results if code == 200)
    assert client.post(f"/refund-proposals/{winner['proposal_id']}/approve").json() == winner


def test_concurrent_replay_of_same_proposal(database, client):
    engine, _ = database
    proposal = create_proposal(client, demo_id("duplicate-payment")).json()
    barrier = Barrier(2)

    def approve():
        with TestClient(app) as concurrent_client:
            barrier.wait(timeout=10)
            response = concurrent_client.post(f"/refund-proposals/{proposal['id']}/approve")
            return response.status_code, response.json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(approve) for _ in range(2)]
        results = [future.result() for future in futures]
    assert [code for code, _ in results] == [200, 200]
    assert results[0][1] == results[1][1]
    assert refund_count(engine) == 1


def test_database_unique_constraint(database, client):
    engine, _ = database
    ids = [create_proposal(client, demo_id("duplicate-payment")).json()["id"] for _ in range(2)]
    assert client.post(f"/refund-proposals/{ids[0]}/approve").status_code == 200
    with pytest.raises(IntegrityError):
        with Session(engine) as session, session.begin():
            session.add(Refund(
                payment_id=demo_id("duplicate-payment"), proposal_id=uuid.UUID(ids[1]),
                amount_cents=2900, currency="CAD",
            ))
            session.flush()
    assert refund_count(engine) == 1


def test_other_customer_is_not_visible_or_mutable(database, client):
    engine, _ = database
    customer_id, invoice_id, payment_id, proposal_id = (uuid.uuid4() for _ in range(4))
    with Session(engine) as session, session.begin():
        session.add(Customer(id=customer_id, name="Other customer"))
        session.flush()
        session.add(Invoice(id=invoice_id, customer_id=customer_id, amount_cents=2900, currency="CAD"))
        session.flush()
        session.add(Payment(id=payment_id, invoice_id=invoice_id, amount_cents=2900, currency="CAD",
                            payment_method="demo-card", status="captured",
                            captured_at=datetime.now(timezone.utc)))
        session.flush()
        session.add(RefundProposal(id=proposal_id, payment_id=payment_id, status="pending"))
    assert str(payment_id) not in [payment["id"] for payment in client.get("/payments").json()]
    assert create_proposal(client, payment_id).status_code == 404
    assert client.post(f"/refund-proposals/{proposal_id}/approve").status_code == 404
    assert client.get("/refund-proposals").json() == []
    assert client.get("/refunds").json() == []
    assert refund_count(engine) == 0


def test_approval_rechecks_payment_data(database, client):
    engine, _ = database
    proposal = create_proposal(client, demo_id("duplicate-payment")).json()
    with Session(engine) as session, session.begin():
        payment = session.get(Payment, demo_id("duplicate-payment"))
        payment.payment_method = "different-card"
    assert client.post(f"/refund-proposals/{proposal['id']}/approve").status_code == 409
    assert refund_count(engine) == 0


@pytest.mark.parametrize("field,value", [
    ("amount_cents", 1000),
    ("currency", "USD"),
    ("payment_method", "different-card"),
    ("status", "failed"),
    ("captured_at", datetime(2026, 1, 1, 12, tzinfo=timezone.utc)),
])
def test_different_payment_data_is_ineligible(database, client, field, value):
    engine, _ = database
    with Session(engine) as session, session.begin():
        payment = session.get(Payment, demo_id("duplicate-payment"))
        setattr(payment, field, value)
    assert create_proposal(client, demo_id("duplicate-payment")).status_code == 409
    assert refund_count(engine) == 0


def test_approval_is_atomic_on_database_failure(database, client):
    engine, _ = database
    proposal_id = create_proposal(client, demo_id("duplicate-payment")).json()["id"]

    def reject_refund(mapper, connection, target):
        raise SQLAlchemyError("sensitive test detail")

    event.listen(Refund, "before_insert", reject_refund)
    try:
        response = client.post(f"/refund-proposals/{proposal_id}/approve")
    finally:
        event.remove(Refund, "before_insert", reject_refund)
    assert response.status_code == 503
    assert "sensitive test detail" not in response.text
    assert refund_count(engine) == 0
    assert client.get("/refund-proposals").json()[0]["status"] == "pending"
