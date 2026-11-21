import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import exists, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ops_agent.database import create_database_engine
from ops_agent.demo import DEMO_CUSTOMER_ID
from ops_agent.models import Invoice, Payment, Refund, RefundProposal

logger = logging.getLogger(__name__)


class PaymentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    invoice_id: uuid.UUID
    amount_cents: int
    currency: str
    payment_method: str
    status: str
    captured_at: datetime
    refunded: bool


class ProposalCreate(BaseModel):
    payment_id: uuid.UUID


class ProposalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    payment_id: uuid.UUID
    status: str
    amount_cents: int
    currency: str


def proposal_output(proposal: RefundProposal, payment: Payment) -> dict:
    return {
        "id": proposal.id,
        "payment_id": proposal.payment_id,
        "status": proposal.status,
        "amount_cents": payment.amount_cents,
        "currency": payment.currency,
    }


class RefundOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    proposal_id: uuid.UUID
    payment_id: uuid.UUID
    amount_cents: int
    currency: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "session_factory"):
        engine = create_database_engine()
        app.state.session_factory = sessionmaker(engine, expire_on_commit=False)
        try:
            yield
        finally:
            engine.dispose()
    else:
        yield


app = FastAPI(title="Simulated refund backend", lifespan=lifespan)


def get_session(request: Request):
    with request.app.state.session_factory() as session:
        try:
            yield session
        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Database operation failed (%s).", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Database operation failed.") from None


def customer_payment_query():
    return select(Payment).join(Invoice).where(Invoice.customer_id == DEMO_CUSTOMER_ID)


def is_duplicate(session: Session, payment: Payment) -> bool:
    if payment.status != "captured":
        return False
    invoice = session.get(Invoice, payment.invoice_id)
    if invoice is None or (payment.amount_cents, payment.currency) != (invoice.amount_cents, invoice.currency):
        return False
    original = session.scalar(
        select(Payment.id).where(
            Payment.invoice_id == payment.invoice_id,
            Payment.amount_cents == payment.amount_cents,
            Payment.currency == payment.currency,
            Payment.payment_method == payment.payment_method,
            Payment.status == "captured",
            Payment.captured_at < payment.captured_at,
        ).limit(1)
    )
    return original is not None


@app.get("/payments", response_model=list[PaymentOut])
def list_payments(session: Session = Depends(get_session)):
    refunded = exists(select(Refund.id).where(Refund.payment_id == Payment.id)).label("refunded")
    rows = session.execute(customer_payment_query().add_columns(refunded)
                           .order_by(Payment.captured_at, Payment.id)).all()
    return [
        {"id": payment.id, "invoice_id": payment.invoice_id, "amount_cents": payment.amount_cents,
         "currency": payment.currency, "payment_method": payment.payment_method, "status": payment.status,
         "captured_at": payment.captured_at, "refunded": is_refunded}
        for payment, is_refunded in rows
    ]


@app.get("/refund-proposals", response_model=list[ProposalOut])
def list_proposals(session: Session = Depends(get_session)):
    rows = session.execute(
        select(RefundProposal, Payment).join(Payment).join(Invoice)
        .where(Invoice.customer_id == DEMO_CUSTOMER_ID)
        .order_by(RefundProposal.created_at, RefundProposal.id)
    ).all()
    return [proposal_output(proposal, payment) for proposal, payment in rows]


@app.get("/refunds", response_model=list[RefundOut])
def list_refunds(session: Session = Depends(get_session)):
    return session.scalars(
        select(Refund).join(Payment).join(Invoice)
        .where(Invoice.customer_id == DEMO_CUSTOMER_ID)
        .order_by(Refund.created_at, Refund.id)
    ).all()


@app.post("/refund-proposals", response_model=ProposalOut, status_code=status.HTTP_201_CREATED)
def create_proposal(body: ProposalCreate, session: Session = Depends(get_session)):
    payment = session.scalar(customer_payment_query().where(Payment.id == body.payment_id).with_for_update(of=Payment))
    if payment is None:
        raise HTTPException(status_code=404, detail="Payment not found.")
    if not is_duplicate(session, payment):
        raise HTTPException(status_code=409, detail="Payment is not an eligible duplicate.")
    if session.scalar(select(Refund.id).where(Refund.payment_id == payment.id)) is not None:
        raise HTTPException(status_code=409, detail="Payment has already been refunded.")
    proposal = RefundProposal(payment_id=payment.id, status="pending")
    session.add(proposal)
    session.commit()
    return proposal_output(proposal, payment)


@app.post("/refund-proposals/{proposal_id}/approve", response_model=RefundOut)
def approve_proposal(proposal_id: uuid.UUID, session: Session = Depends(get_session)):
    payment = session.scalar(
        customer_payment_query()
        .join(RefundProposal, RefundProposal.payment_id == Payment.id)
        .where(RefundProposal.id == proposal_id)
        .with_for_update(of=Payment)
    )
    if payment is None:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    proposal = session.get(RefundProposal, proposal_id)
    existing = session.scalar(select(Refund).where(Refund.payment_id == payment.id))
    if existing is not None:
        if existing.proposal_id == proposal_id:
            return existing
        raise HTTPException(status_code=409, detail="Payment has already been refunded.")
    if proposal is None or proposal.status != "pending" or not is_duplicate(session, payment):
        raise HTTPException(status_code=409, detail="Proposal is not eligible for approval.")
    refund = Refund(
        proposal_id=proposal.id,
        payment_id=payment.id,
        amount_cents=payment.amount_cents,
        currency=payment.currency,
    )
    proposal.status = "approved"
    session.add(refund)
    session.commit()
    return refund
