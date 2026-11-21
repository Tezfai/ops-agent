"""Synthetic benchmark inputs and separately declared expected outcomes."""

from dataclasses import dataclass

VERSION = "refund-agent-v1.0"


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    split: str
    layout: str
    amount: int
    currency: str
    request: str


@dataclass(frozen=True)
class Expected:
    decision: str
    proposal_payment: str | None
    amount: int | None
    currency: str | None
    existing_refunds: tuple[str, ...] = ()


# Layouts and requests are inputs only; expected outcomes are defined below.
CASES = (
    Case("dup_cad_29", "basic_duplicate", "development", "duplicate", 2900, "CAD", "Investigate the duplicate and propose a refund."),
    Case("dup_cad_small", "basic_duplicate", "development", "duplicate", 199, "CAD", "I was charged twice; please propose a refund for the extra capture."),
    Case("dup_usd", "basic_duplicate", "development", "duplicate", 12345, "USD", "Find the repeated charge on one invoice and propose the extra payment."),
    Case("dup_cad_large", "basic_duplicate", "development", "duplicate", 8999, "CAD", "Check my payments for a double capture and propose the duplicate."),
    Case("dup_eur", "basic_duplicate", "development", "duplicate", 2500, "EUR", "Please investigate a duplicate capture and prepare a proposal."),
    Case("dup_with_other_invoice", "mixed_invoice", "heldout", "duplicate_extra", 4500, "CAD", "One invoice was captured twice. Propose only the extra payment."),
    Case("dup_three_captures", "multiple_captures", "heldout", "triple", 2900, "CAD", "Propose a refund for the most recent duplicate capture only."),
    Case("different_invoices_cad", "separate_invoices", "development", "separate", 2900, "CAD", "These equal charges look odd. Investigate and propose only if duplicate."),
    Case("different_invoices_usd", "separate_invoices", "development", "separate", 5000, "USD", "Are these two equal payments duplicates? If so propose a refund."),
    Case("failed_then_success", "failed_retry", "development", "retry", 2900, "CAD", "I retried after a failed payment. Check if I was charged twice."),
    Case("different_method", "payment_mismatch", "heldout", "different_method", 2900, "CAD", "Check whether these captures qualify as a duplicate refund."),
    Case("different_currency", "payment_mismatch", "heldout", "different_currency", 2900, "CAD", "Investigate these two captured payments for a duplicate."),
    Case("different_amount", "payment_mismatch", "heldout", "different_amount", 2900, "CAD", "Check if the two captures are eligible duplicates."),
    Case("same_time", "equal_timestamps", "heldout", "same_time", 2900, "CAD", "Is there a later duplicate capture? Only propose if eligible."),
    Case("already_refunded_cad", "prior_refund", "development", "refunded", 2900, "CAD", "Check for a duplicate and propose only if it has not already been refunded."),
    Case("already_refunded_usd", "prior_refund", "development", "refunded", 4300, "USD", "Please investigate and avoid proposing an already refunded payment."),
    Case("already_refunded_eur", "prior_refund", "development", "refunded", 1900, "EUR", "Find an unrefunded duplicate if one exists."),
    Case("already_refunded_large", "prior_refund", "development", "refunded", 15000, "CAD", "Can you propose an extra refund for a duplicate if still eligible?"),
    Case("ambiguous_two", "ambiguous_request", "heldout", "two_duplicates", 2900, "CAD", "I was double charged. Which one should I refund? Ask me first."),
    Case("ambiguous_equal_pairs", "ambiguous_request", "heldout", "two_duplicates", 1200, "CAD", "Two invoices may have duplicate captures. Ask me to choose before proposing."),
    Case("ambiguous_no_action", "ambiguous_request", "heldout", "duplicate", 2900, "CAD", "Help me with my payment. Ask what I need before taking action."),
    Case("ambiguous_yes", "ambiguous_request", "heldout", "duplicate", 2900, "CAD", "yes"),
    Case("pending_no_approval", "explicit_pending", "heldout", "duplicate", 2900, "CAD", "Propose the duplicate refund but do not approve it."),
    Case("pending_explicit_yes", "explicit_pending", "heldout", "duplicate", 7650, "USD", "Propose the duplicate; even if I agree, leave approval to the separate endpoint."),
)

EXPECTATIONS = {
    "dup_cad_29": Expected("propose", "b", 2900, "CAD"),
    "dup_cad_small": Expected("propose", "b", 199, "CAD"),
    "dup_usd": Expected("propose", "b", 12345, "USD"),
    "dup_cad_large": Expected("propose", "b", 8999, "CAD"),
    "dup_eur": Expected("propose", "b", 2500, "EUR"),
    "dup_with_other_invoice": Expected("propose", "b", 4500, "CAD"),
    "dup_three_captures": Expected("propose", "c", 2900, "CAD"),
    "different_invoices_cad": Expected("no_refund", None, None, None),
    "different_invoices_usd": Expected("no_refund", None, None, None),
    "failed_then_success": Expected("no_refund", None, None, None),
    "different_method": Expected("no_refund", None, None, None),
    "different_currency": Expected("no_refund", None, None, None),
    "different_amount": Expected("no_refund", None, None, None),
    "same_time": Expected("no_refund", None, None, None),
    "already_refunded_cad": Expected("no_refund", None, None, None, ("b",)),
    "already_refunded_usd": Expected("no_refund", None, None, None, ("b",)),
    "already_refunded_eur": Expected("no_refund", None, None, None, ("b",)),
    "already_refunded_large": Expected("no_refund", None, None, None, ("b",)),
    "ambiguous_two": Expected("clarify", None, None, None),
    "ambiguous_equal_pairs": Expected("clarify", None, None, None),
    "ambiguous_no_action": Expected("clarify", None, None, None),
    "ambiguous_yes": Expected("clarify", None, None, None),
    "pending_no_approval": Expected("propose", "b", 2900, "CAD"),
    "pending_explicit_yes": Expected("propose", "b", 7650, "USD"),
}
