"""Generate and independently replay synthetic tool-use demonstrations."""

import argparse
import json
import os
import random
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ops_agent.agent import INSTRUCTIONS, TOOLS, ModelFailure, ToolCall, execute_tool
from ops_agent.database import create_database_engine
from ops_agent.demo import DEMO_CUSTOMER_ID
from ops_agent.main import app
from ops_agent.models import Customer, Invoice, Payment, Refund, RefundProposal
if __package__:
    from scripts.run_scenarios import isolated_database
else:
    from run_scenarios import isolated_database

VERSION = "synthetic-refund-sft-v1.0"
NAMESPACE = uuid.UUID("48b6f316-1490-4dd8-8675-75a00a317943")
TEMPLATES = (
    ("duplicate_pair", "eligible_duplicate", "train"),
    ("equal_invoice_pair", "separate_invoices", "validation"),
    ("single_failed_retry", "failed_retry", "train"),
    ("refunded_pair", "already_refunded", "validation"),
)
WORDS = {
    "duplicate_pair": ("Please investigate the later capture and prepare a refund proposal if it qualifies.",
                       "Review the two captures on this invoice and request the eligible refund."),
    "equal_invoice_pair": ("These invoices happen to have matching totals. Please check if either has an extra capture.",
                           "Review both invoice payments and only request a refund for a genuine duplicate."),
    "single_failed_retry": ("A payment attempt was declined before the successful attempt. Check for an extra capture.",
                            "Please examine the unsuccessful attempt and the completed retry for an overcharge."),
    "refunded_pair": ("Please check whether the returned extra capture still needs another proposal.",
                      "Review the payment history, including the previous refund, for an outstanding extra charge."),
}


@dataclass(frozen=True)
class Scenario:
    id: str
    template: str
    family: str
    split: str
    request: str
    amount: int
    currency: str
    seed: int
    variant: int
    dataset_version: str = VERSION


def ident(case: Scenario, name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, f"{case.dataset_version}:{case.seed}:{case.template}:{case.variant}:{name}")


def scenarios(seed: int) -> list[Scenario]:
    result = []
    for template, family, split in TEMPLATES:
        for variant in range(10):
            rng = random.Random(f"{seed}:{template}:{variant}")
            amount = rng.randrange(201, 19999) * 7 + 3
            currency = rng.choice(("CAD", "USD", "EUR"))
            request = WORDS[template][variant % len(WORDS[template])]
            result.append(Scenario(f"{template}-{variant + 1:02d}", template, family, split,
                                   request, amount, currency, seed, variant))
    return result


def reset_and_seed(engine: Engine, case: Scenario) -> tuple[uuid.UUID | None, list[tuple]]:
    invoice_a, invoice_b = (ident(case, name) for name in ("invoice-a", "invoice-b"))
    a, b = (ident(case, name) for name in ("capture-a", "capture-b"))
    rng = random.Random(f"{case.seed}:{case.id}:layout")
    card = f"synthetic-instrument-{rng.randrange(1000, 10000)}"
    later = rng.randrange(1, 21)
    records: list[tuple[uuid.UUID, uuid.UUID, int, str]] = [(a, invoice_a, 0, "captured")]
    target = None
    if case.template in {"duplicate_pair", "duplicate_time", "duplicate_by_id"}:
        records += [(b, invoice_a, later, "captured")]
        target = b
    elif case.template in {"equal_invoice_pair", "separate_invoice_ids", "separate_payment_ids"}:
        records += [(b, invoice_b, later, "captured")]
    elif case.template in {"single_failed_retry", "retry_disclosed", "retry_investigate"}:
        records = [(a, invoice_a, 0, "failed"), (b, invoice_a, later, "captured")]
    elif case.template in {"refunded_pair", "refund_disclosed", "refund_investigate"}:
        records += [(b, invoice_a, later, "captured")]
    else:
        raise ValueError("Unknown synthetic template.")
    random.Random(f"{case.seed}:{case.id}:insertion").shuffle(records)
    with Session(engine) as session, session.begin():
        schema = session.scalar(text("SELECT current_schema()"))
        if not isinstance(schema, str) or not schema.startswith("ops_agent_scenario_"):
            raise RuntimeError("Refusing to reset data outside an isolated scenario schema.")
        session.execute(text("TRUNCATE refunds, refund_proposals, payments, invoices, customers"))
        session.add(Customer(id=DEMO_CUSTOMER_ID, name="Synthetic training customer"))
        session.flush()
        for invoice in sorted({record[1] for record in records}):
            session.add(Invoice(id=invoice, customer_id=DEMO_CUSTOMER_ID,
                                amount_cents=case.amount, currency=case.currency))
        session.flush()
        for payment, invoice, offset, status in records:
            session.add(Payment(id=payment, invoice_id=invoice, amount_cents=case.amount,
                                currency=case.currency, payment_method=card, status=status,
                                captured_at=datetime(2025, 6, 1, 10, tzinfo=timezone.utc) + timedelta(days=offset)))
        session.flush()
        if case.family == "already_refunded":
            proposal_id, refund_id = ident(case, "prior-proposal"), ident(case, "prior-refund")
            session.add(RefundProposal(id=proposal_id, payment_id=b, status="approved"))
            session.flush()
            session.add(Refund(id=refund_id, proposal_id=proposal_id, payment_id=b,
                               amount_cents=case.amount, currency=case.currency))
    return target, refund_snapshot(engine)


def refund_snapshot(engine: Engine) -> list[tuple]:
    with Session(engine) as session:
        return [(r.id, r.proposal_id, r.payment_id, r.amount_cents, r.currency)
                for r in session.scalars(select(Refund)).all()]


def proposal_snapshot(engine: Engine) -> set[tuple]:
    with Session(engine) as session:
        return {(p.id, p.payment_id, p.status) for p in session.scalars(select(RefundProposal)).all()}


def expected_decision(case: Scenario) -> str:
    if case.family == "eligible_duplicate":
        return "propose"
    return "no_refund"


def replay(engine: Engine, client: TestClient, case: Scenario, expected_messages: list[dict] | None = None) -> list[dict]:
    replay_calls: list[dict] = []
    if expected_messages is not None:
        replay_calls = [m["tool_calls"][0] for m in expected_messages if m.get("tool_calls")]
        expected_names = ["list_payments"] + (["propose_refund"] if case.family == "eligible_duplicate" else [])
        if [call["function"]["name"] for call in replay_calls] != expected_names:
            raise ValueError("Unexpected demonstration tool sequence.")
    if expected_messages is not None and case.family == "eligible_duplicate":
        saved = json.loads(expected_messages[-2]["content"])["proposal"]
        with Session(engine) as session:
            proposal = session.get(RefundProposal, uuid.UUID(saved["id"]))
            if (proposal is None or proposal.payment_id != ident(case, "capture-b")
                    or proposal.status != "pending"):
                raise ValueError("Saved proposal ID does not identify the original persisted proposal.")
    target, initial_refunds = reset_and_seed(engine, case)
    initial_proposals = proposal_snapshot(engine)
    if len(initial_refunds) != (1 if case.family == "already_refunded" else 0):
        raise ValueError("Unexpected initial refund fixture.")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": case.request},
    ]

    def invoke(name: str, arguments: dict, call_id: str) -> dict:
        if expected_messages is not None:
            recorded = replay_calls.pop(0)
            if (recorded["id"] != call_id or recorded["type"] != "function"
                    or json.loads(recorded["function"]["arguments"]) != arguments):
                raise ValueError("Recorded tool call arguments do not match the scenario.")
            name = recorded["function"]["name"]
            arguments = json.loads(recorded["function"]["arguments"])
        call = ToolCall(call_id, name, json.dumps(arguments, separators=(",", ":")))
        messages.append({"role": "assistant", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": call.arguments}}
        ]})
        output = execute_tool(client, call)
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps(output, sort_keys=True, separators=(",", ":"))})
        return output

    payments = invoke("list_payments", {}, "list-1")["payments"]
    if not isinstance(payments, list) or not payments:
        raise ValueError("Payment list missing.")
    saved = None
    if target is not None:
        saved = invoke("propose_refund", {"payment_id": str(target)}, "proposal-1")["proposal"]
        if (saved["payment_id"] != str(target) or saved["amount_cents"] != case.amount
                or saved["currency"] != case.currency or saved["status"] != "pending"):
            raise ValueError("Proposal target, amount, currency, or status is incorrect.")
    decision = expected_decision(case)
    final = {"decision": decision, "message": (
        f"Pending proposal {saved['id']} for {saved['amount_cents']} {saved['currency']} cents."
        if saved is not None
        else "No additional refund proposal is warranted."
    )}
    messages.append({"role": "assistant", "content": json.dumps(final, separators=(",", ":"))})
    with Session(engine) as session:
        proposals = session.scalars(select(RefundProposal)).all()
        expected_count = (1 if case.family == "already_refunded" else 0) + (1 if target else 0)
        if len(proposals) != expected_count:
            raise ValueError("Unexpected proposal count.")
        if target is not None:
            matches = [p for p in proposals if p.payment_id == target and p.status == "pending"]
            if len(matches) != 1 or saved is None or str(matches[0].id) != saved["id"]:
                raise ValueError("Claimed proposal was not persisted for the expected payment.")
        elif any(p.status == "pending" for p in proposals):
            raise ValueError("Ineligible or ambiguous case created a pending proposal.")
    after_proposals = proposal_snapshot(engine)
    if not initial_proposals.issubset(after_proposals) or len(after_proposals - initial_proposals) != (1 if target else 0):
        raise ValueError("Unexpected change to proposal state.")
    if refund_snapshot(engine) != initial_refunds:
        raise ValueError("A refund was created or changed during the demonstration.")
    if expected_messages is not None:
        generated = json.loads(json.dumps(messages))
        original = json.loads(json.dumps(expected_messages))
        if target:
            for trace in (generated, original):
                result = json.loads(trace[-2]["content"])
                proposal_id = result["proposal"]["id"]
                result["proposal"]["id"] = "<persisted-proposal-id>"
                trace[-2]["content"] = json.dumps(result, sort_keys=True, separators=(",", ":"))
                final = json.loads(trace[-1]["content"])
                final["message"] = final["message"].replace(proposal_id, "<persisted-proposal-id>")
                trace[-1]["content"] = json.dumps(final, separators=(",", ":"))
        if generated != original:
            raise ValueError("Replay did not reproduce the saved tool results and decision.")
    return messages


def generate(seed: int, output: Path) -> int:
    files = [output / f"{VERSION}-train.jsonl", output / f"{VERSION}-validation.jsonl",
             output / f"{VERSION}-summary.json"]
    if any(path.exists() for path in files):
        print("Dataset output already exists; choose an empty output directory.", file=sys.stderr)
        return 1
    engine = create_database_engine()
    accepted: dict[str, list[dict]] = {"train": [], "validation": []}
    counts: Counter = Counter()
    reasons: Counter = Counter()
    try:
        with isolated_database(engine) as isolated:
            app.state.session_factory = sessionmaker(isolated, expire_on_commit=False)
            try:
                with TestClient(app) as client:
                    for case in scenarios(seed):
                        try:
                            messages = replay(isolated, client, case)
                            replay(isolated, client, case, messages)
                            accepted[case.split].append({
                                "dataset_version": VERSION, "seed": seed, "scenario_id": case.id,
                                "template": case.template, "family": case.family, "split": case.split,
                                "tools": TOOLS, "messages": messages,
                            })
                            counts[(case.family, "accepted")] += 1
                        except (ValueError, KeyError, ModelFailure) as exc:
                            counts[(case.family, "rejected")] += 1
                            reasons[(case.family, str(exc))] += 1
            finally:
                del app.state.session_factory
    finally:
        engine.dispose()
    if not accepted["train"] or not accepted["validation"]:
        print("Generation failed: one or both splits contain no accepted demonstrations.", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    for split, path in zip(("train", "validation"), files):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in accepted[split]:
                stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
    summary = {"dataset_version": VERSION, "seed": seed,
               "split_counts": {split: len(records) for split, records in accepted.items()},
               "families": {family: {"accepted": counts[(family, "accepted")], "rejected": counts[(family, "rejected")]}
                            for _, family, _ in TEMPLATES},
               "rejections": [{"family": family, "reason": reason, "count": count}
                              for (family, reason), count in sorted(reasons.items())]}
    with files[2].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    return 1 if reasons else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate replay-validated synthetic demonstrations.")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", type=Path, default=Path("data") / VERSION)
    args = parser.parse_args()
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        print("Set OPS_AGENT_TEST_USE_DEV_DATABASE=1 only for a disposable dev/test database.", file=sys.stderr)
        return 1
    try:
        return generate(args.seed, args.output)
    except Exception as exc:
        print(f"Dataset generation failed ({type(exc).__name__}); check the database and output path.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
