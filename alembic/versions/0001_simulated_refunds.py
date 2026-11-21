"""Create simulated billing and refund tables."""

import sqlalchemy as sa
from alembic import op

revision = "0001_simulated_refunds"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "customers",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
    )
    op.create_table(
        "invoices",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("customer_id", sa.Uuid(), sa.ForeignKey("customers.id"), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.CheckConstraint("amount_cents > 0", name="ck_invoices_amount_positive"),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_invoices_currency"),
    )
    op.create_index("ix_invoices_customer_id", "invoices", ["customer_id"])
    op.create_table(
        "payments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("invoice_id", sa.Uuid(), sa.ForeignKey("invoices.id"), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("payment_method", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("amount_cents > 0", name="ck_payments_amount_positive"),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_payments_currency"),
        sa.CheckConstraint("status IN ('captured', 'failed')", name="ck_payments_status"),
    )
    op.create_index("ix_payments_invoice_id", "payments", ["invoice_id"])
    op.create_table(
        "refund_proposals",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("payment_id", sa.Uuid(), sa.ForeignKey("payments.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("status IN ('pending', 'approved')", name="ck_proposals_status"),
        sa.UniqueConstraint("id", "payment_id", name="uq_proposal_id_payment"),
    )
    op.create_index("ix_refund_proposals_payment_id", "refund_proposals", ["payment_id"])
    op.create_table(
        "refunds",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("proposal_id", sa.Uuid(), nullable=False),
        sa.Column("payment_id", sa.Uuid(), sa.ForeignKey("payments.id"), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["proposal_id", "payment_id"], ["refund_proposals.id", "refund_proposals.payment_id"],
            name="fk_refund_proposal_payment",
        ),
        sa.UniqueConstraint("payment_id", name="uq_refunds_payment"),
        sa.UniqueConstraint("proposal_id", name="uq_refunds_proposal"),
        sa.CheckConstraint("amount_cents > 0", name="ck_refunds_amount_positive"),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_refunds_currency"),
    )


def downgrade():
    op.drop_table("refunds")
    op.drop_table("refund_proposals")
    op.drop_table("payments")
    op.drop_table("invoices")
    op.drop_table("customers")
