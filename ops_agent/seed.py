from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ops_agent.demo import DEMO_CUSTOMER_ID, demo_id
from ops_agent.models import Customer, Invoice, Payment


def seed(session: Session) -> None:
    customers = [{"id": DEMO_CUSTOMER_ID, "name": "Demo customer"}]
    invoices = [
        {"id": demo_id(name), "customer_id": DEMO_CUSTOMER_ID, "amount_cents": 2900, "currency": "CAD"}
        for name in ("duplicate-invoice", "legitimate-invoice-a", "legitimate-invoice-b")
    ]
    payments = [
        {
            "id": demo_id(name),
            "invoice_id": demo_id(invoice),
            "amount_cents": 2900,
            "currency": "CAD",
            "payment_method": "demo-card",
            "status": "captured",
            "captured_at": datetime(2026, 1, day, 12, tzinfo=timezone.utc),
        }
        for name, invoice, day in (
            ("original-payment", "duplicate-invoice", 1),
            ("duplicate-payment", "duplicate-invoice", 2),
            ("legitimate-payment-a", "legitimate-invoice-a", 3),
            ("legitimate-payment-b", "legitimate-invoice-b", 4),
        )
    ]
    for model, rows in ((Customer, customers), (Invoice, invoices), (Payment, payments)):
        for values in rows:
            session.execute(insert(model).values(**values).on_conflict_do_nothing(index_elements=["id"]))
            existing = session.execute(select(model).where(model.id == values["id"])).scalar_one()
            if any(getattr(existing, key) != value for key, value in values.items()):
                raise ValueError(f"Existing demo {model.__tablename__} data differs from the seed.")
