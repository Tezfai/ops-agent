import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)


class Invoice(Base):
    __tablename__ = "invoices"
    __table_args__ = (
        CheckConstraint("amount_cents > 0", name="ck_invoices_amount_positive"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_invoices_currency"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)


class Payment(Base):
    __tablename__ = "payments"
    __table_args__ = (
        CheckConstraint("amount_cents > 0", name="ck_payments_amount_positive"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_payments_currency"),
        CheckConstraint("status IN ('captured', 'failed')", name="ck_payments_status"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    invoice_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("invoices.id"), nullable=False, index=True)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    payment_method: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RefundProposal(Base):
    __tablename__ = "refund_proposals"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved')", name="ck_proposals_status"),
        UniqueConstraint("id", "payment_id", name="uq_proposal_id_payment"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    payment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("payments.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Refund(Base):
    __tablename__ = "refunds"
    __table_args__ = (
        ForeignKeyConstraint(
            ["proposal_id", "payment_id"],
            ["refund_proposals.id", "refund_proposals.payment_id"],
            name="fk_refund_proposal_payment",
        ),
        UniqueConstraint("payment_id", name="uq_refunds_payment"),
        UniqueConstraint("proposal_id", name="uq_refunds_proposal"),
        CheckConstraint("amount_cents > 0", name="ck_refunds_amount_positive"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_refunds_currency"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    proposal_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    payment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("payments.id"), nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
