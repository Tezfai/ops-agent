"""Run versioned cases through the real tool gateway and independently grade PostgreSQL."""

import importlib
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, ContextManager

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from benchmarks.v1 import CASES, EXPECTATIONS, VERSION, Case
from ops_agent.agent import DECISION_FORMAT, INSTRUCTIONS, TOOLS, Model, ModelFailure, ModelTurn, OpenAIModel, ToolCall, run_agent
from ops_agent.demo import DEMO_CUSTOMER_ID, demo_id
from ops_agent.main import app
from ops_agent.models import Customer, Invoice, Payment, Refund, RefundProposal


def payment_id(case: Case, label: str) -> uuid.UUID:
    return demo_id(f"benchmark-{VERSION}-{case.id}-payment-{label}")


def seed_case(engine: Engine, case: Case) -> None:
    invoice_a = demo_id(f"benchmark-{case.id}-invoice-a")
    invoice_b = demo_id(f"benchmark-{case.id}-invoice-b")
    second_invoice = invoice_b if case.layout == "separate" else invoice_a
    records = [
        ("a", invoice_a, case.amount, case.currency, "card-a",
         "failed" if case.layout == "retry" else "captured", 1),
        ("b", second_invoice, case.amount + (100 if case.layout == "different_amount" else 0),
         "USD" if case.layout == "different_currency" else case.currency,
         "card-b" if case.layout == "different_method" else "card-a", "captured",
         1 if case.layout == "same_time" else 2),
    ]
    if case.layout in ("duplicate_extra", "two_duplicates"):
        records.append(("c", invoice_b, case.amount, case.currency, "card-a", "captured", 3))
    if case.layout == "two_duplicates":
        records.append(("d", invoice_b, case.amount, case.currency, "card-a", "captured", 4))
    if case.layout == "triple":
        records.append(("c", invoice_a, case.amount, case.currency, "card-a", "captured", 3))
    with Session(engine) as session, session.begin():
        session.add(Customer(id=DEMO_CUSTOMER_ID, name="Synthetic benchmark customer"))
        session.flush()
        for invoice in {row[1] for row in records}:
            session.add(Invoice(id=invoice, customer_id=DEMO_CUSTOMER_ID,
                                amount_cents=case.amount, currency=case.currency))
        session.flush()
        for label, invoice, amount, currency, method, status, day in records:
            session.add(Payment(id=payment_id(case, label), invoice_id=invoice, amount_cents=amount,
                                currency=currency, payment_method=method, status=status,
                                captured_at=datetime(2026, 1, day, 12, tzinfo=timezone.utc)))


def snapshot(engine: Engine) -> tuple[list[tuple], list[tuple]]:
    with Session(engine) as session:
        proposals = session.scalars(select(RefundProposal)).all()
        refunds = session.scalars(select(Refund)).all()
        return (
            [(p.id, p.payment_id, p.status) for p in proposals],
            [(r.id, r.proposal_id, r.payment_id, r.amount_cents, r.currency) for r in refunds],
        )


class FixtureModel:
    """Scripted wiring check; it does not measure model reasoning."""

    def __init__(self, case: Case):
        self.case = case
        self.step = 0

    def next_turn(self, history: list[dict[str, Any]]) -> ModelTurn:
        self.step += 1
        if self.step == 1:
            return ModelTurn([ToolCall("list", "list_payments", "{}")])
        is_ambiguous = self.case.family == "ambiguous_request"
        should_propose = self.case.layout in {"duplicate", "duplicate_extra", "triple"} and not is_ambiguous
        if self.step == 2 and should_propose:
            label = "c" if self.case.layout == "triple" else "b"
            return ModelTurn([ToolCall("proposal", "propose_refund",
                                       json.dumps({"payment_id": str(payment_id(self.case, label))}))])
        decision = "clarify" if is_ambiguous else "propose" if should_propose else "no_refund"
        message = "Which payment should I investigate?" if is_ambiguous else "Investigation complete."
        return ModelTurn([], json.dumps({"decision": decision, "message": message}))


def validated_decision(text: str) -> str | None:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if (not isinstance(value, dict) or set(value) != {"decision", "message"}
            or value["decision"] not in ("propose", "no_refund", "clarify")
            or not isinstance(value["message"], str) or not value["message"].strip()):
        return None
    return value["decision"]


def adapter_factory(path: str) -> Callable[[], Model]:
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError("Adapter must be specified as module:factory.")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError("Adapter factory is not callable.")
    return factory


def grade(case: Case, result: Any, before: tuple[list[tuple], list[tuple]],
          after: tuple[list[tuple], list[tuple]]) -> list[str]:
    expected = EXPECTATIONS[case.id]
    failures: list[str] = []
    if result.error:
        failures.append(f"Agent error: {result.error}")
    try:
        answer = json.loads(result.final_text)
        if not isinstance(answer, dict) or set(answer) != {"decision", "message"}:
            raise ValueError
        if answer["decision"] != expected.decision:
            failures.append(f"Expected decision {expected.decision}.")
        if not isinstance(answer["message"], str) or not answer["message"].strip():
            failures.append("Decision message is missing.")
        elif expected.decision == "clarify" and "?" not in answer["message"]:
            failures.append("Expected a clarifying question.")
    except (ValueError, TypeError):
        failures.append("Final decision is not a valid JSON decision object.")

    before_proposals, before_refunds = before
    after_proposals, after_refunds = after
    expected_refunded_ids = {payment_id(case, label) for label in expected.existing_refunds}
    if len(before_refunds) != len(expected.existing_refunds) or {r[2] for r in before_refunds} != expected_refunded_ids:
        failures.append("Initial refund fixture does not match the expected state.")
    if any(r[3] != case.amount or r[4] != case.currency for r in before_refunds):
        failures.append("Initial refund amount or currency does not match the fixture.")
    if set(after_refunds) != set(before_refunds) or len(after_refunds) != len(before_refunds):
        failures.append("Refund state changed without separate approval.")
    if not set(before_proposals).issubset(set(after_proposals)):
        failures.append("Existing proposals changed.")
    new_proposals = [p for p in after_proposals if p not in before_proposals]
    if expected.proposal_payment is None:
        if new_proposals or result.proposals:
            failures.append("Expected no new proposal.")
    else:
        target = payment_id(case, expected.proposal_payment)
        if len(new_proposals) != 1 or new_proposals[0][1:] != (target, "pending"):
            failures.append("Expected exactly one pending proposal for the specified payment.")
        if len(result.proposals) != 1:
            failures.append("Agent did not return exactly one confirmed proposal.")
        else:
            proposal = result.proposals[0]
            if (proposal.get("payment_id") != str(target) or proposal.get("amount_cents") != expected.amount
                    or proposal.get("currency") != expected.currency or proposal.get("status") != "pending"):
                failures.append("Confirmed proposal payment, amount, currency, or status is wrong.")
            if len(new_proposals) == 1 and proposal.get("id") != str(new_proposals[0][0]):
                failures.append("Confirmed proposal ID differs from database.")
    return failures


def run_case(engine: Engine, case: Case, model: Model,
             isolate: Callable[[Engine], ContextManager[Engine]]) -> dict[str, Any]:
    with isolate(engine) as database:
        seed_case(database, case)
        app.state.session_factory = sessionmaker(database, expire_on_commit=False)
        try:
            with TestClient(app) as client:
                if case.layout == "refunded":
                    response = client.post("/refund-proposals", json={"payment_id": str(payment_id(case, "b"))})
                    if response.status_code != 201:
                        raise RuntimeError("Could not create pre-existing refund fixture.")
                    response = client.post(f"/refund-proposals/{response.json()['id']}/approve")
                    if response.status_code != 200:
                        raise RuntimeError("Could not approve pre-existing refund fixture.")
                before = snapshot(database)
                started = time.perf_counter()
                result = run_agent(model, client, case.request)
                latency_ms = round((time.perf_counter() - started) * 1000, 1)
                after = snapshot(database)
                failures = grade(case, result, before, after)
                # Amount is derived from the persisted payment, not just the tool response.
                expectation = EXPECTATIONS[case.id]
                if expectation.proposal_payment is not None:
                    with Session(database) as session:
                        payment = session.get(Payment, payment_id(case, expectation.proposal_payment))
                        if (payment is None or payment.amount_cents != expectation.amount
                                or payment.currency != expectation.currency):
                            failures.append("Database payment amount or currency does not match expectation.")
                categories = {"decision_format": [], "tool": [], "business_outcome": [], "model": []}
                for failure in failures:
                    if failure.startswith("Agent error:"):
                        categories["tool" if result.error_kind == "tool" else "model"].append(failure)
                    elif (failure == "Final decision is not a valid JSON decision object."
                          and not result.final_text and result.error):
                        categories["tool" if result.error_kind == "tool" else "model"].append(failure)
                    elif failure in ("Final decision is not a valid JSON decision object.", "Decision message is missing."):
                        categories["decision_format"].append(failure)
                    else:
                        categories["business_outcome"].append(failure)
                return {"case": case.id, "family": case.family, "status": "pass" if not failures else "fail",
                        "failures": failures, "latency_ms": latency_ms, "model_calls": result.model_calls,
                        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                        "failure_categories": categories,
                        "tools_used": [event["tool"] for event in result.trace],
                        "validated_decision": validated_decision(result.final_text),
                        "reported_models": result.reported_models}
        finally:
            del app.state.session_factory


def run_benchmark(engine: Engine, split: str, selected: list[str], provider: str,
                  isolate: Callable[[Engine], ContextManager[Engine]], factory_path: str | None = None) -> dict[str, Any]:
    cases = [case for case in CASES if case.split == split and (not selected or case.id in selected)]
    invalid = set(selected) - {case.id for case in cases}
    if invalid:
        raise ValueError("Unknown case or case not in selected split: " + ", ".join(sorted(invalid)))
    if not cases:
        raise ValueError("No benchmark cases selected.")
    factory = adapter_factory(factory_path) if factory_path else None
    records = []
    adapter_settings = None
    model_name = "scripted-fixture-model" if provider == "mock" and not factory_path else "unavailable"
    for case in cases:
        model = None
        started = time.perf_counter()
        try:
            model = factory() if factory else OpenAIModel() if provider == "live" else FixtureModel(case)
            name = getattr(model, "model", None) if not isinstance(model, FixtureModel) else "scripted-fixture-model"
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Model adapter must expose a nonempty model identifier.")
            if model_name != "unavailable" and name != model_name:
                raise ValueError("Model identifier changed between scenarios.")
            model_name = name
            settings = getattr(model, "settings", None)
            if settings is not None and not isinstance(settings, dict):
                raise ValueError("Adapter settings must be a dictionary.")
            if adapter_settings is not None and settings != adapter_settings:
                raise ValueError("Adapter settings changed between scenarios.")
            if settings is not None:
                adapter_settings = settings
            record = run_case(engine, case, model, isolate)
        except ModelFailure as exc:
            record = {"case": case.id, "family": case.family, "status": "fail",
                      "failures": [str(exc)], "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                      "model_calls": None, "input_tokens": None, "output_tokens": None}
        except Exception as exc:
            record = {"case": case.id, "family": case.family, "status": "fail",
                      "failures": [f"Scenario execution failed ({type(exc).__name__}); check model and database setup."],
                      "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                      "model_calls": None, "input_tokens": None, "output_tokens": None}
        finally:
            if model is not None and callable(getattr(model, "close", None)):
                model.close()
        records.append(record)
        usage = f"input={record['input_tokens']} output={record['output_tokens']}"
        print(f"{record['status'].upper()} {case.id} [{case.family}] {record['latency_ms']} ms {usage}")
        for reason in record["failures"]:
            print(f"  - {reason}")
    prompt_spec = json.dumps({"instructions": INSTRUCTIONS, "tools": TOOLS, "format": DECISION_FORMAT},
                             sort_keys=True, separators=(",", ":"))
    return {"benchmark_version": VERSION, "split": split, "model": model_name,
            "adapter_settings": adapter_settings,
            "agent_configuration_sha256": hashlib.sha256(prompt_spec.encode("utf-8")).hexdigest(),
            "mock": provider == "mock" and not factory_path, "cases": records,
            "passed": sum(record["status"] == "pass" for record in records), "total": len(records)}
