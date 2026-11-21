import json
import os
import random
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from ops_agent.agent import INSTRUCTIONS, TOOLS
from ops_agent.database import create_database_engine
from ops_agent.main import app
from scripts.generate_sft_dataset import ident, refund_snapshot, replay
from scripts.generate_sft_v11 import FAMILIES, VERSION, audit, generate, normalize, scenarios, validate_distribution
from scripts.run_scenarios import isolated_database


def test_balanced_independent_template_assignments():
    cases = scenarios(20261121)
    assert len(cases) == 40
    assert cases == scenarios(20261121)
    assert cases != scenarios(20261122)
    for split in ("train", "validation"):
        selected = [case for case in cases if case.split == split]
        assert Counter(case.family for case in selected) == Counter({family: 5 for family in FAMILIES})
        assert Counter("propose" if case.family == "eligible_duplicate" else "no_refund"
                       for case in selected) == {"propose": 5, "no_refund": 15}
    assert {c.template for c in cases if c.split == "train"}.isdisjoint(
        c.template for c in cases if c.split == "validation"
    )
    with pytest.raises(ValueError):
        validate_distribution(Counter({"separate_invoices": 10, "already_refunded": 10}),
                              Counter({"no_refund": 20}), "validation")


def test_normalized_values_do_not_hide_identical_messages():
    a = [{"role": "user", "content": "Check 24f6312a-b37b-4832-92e4-f9ef1e3ba637 for CAD 123."}]
    b = [{"role": "user", "content": "Check 4b011769-7790-4619-bdf9-78fb3bc96a1a for USD 456."}]
    assert normalize(a) == normalize(b)
    b[0]["content"] = "Review the later capture."
    assert normalize(a) != normalize(b)


def test_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        generate(20261121, tmp_path)


def synthetic_export(tmp_path):
    rows = {"train": [], "validation": []}
    for case in scenarios(20261121):
        rng = random.Random(f"{case.seed}:{case.id}:layout")
        method = f"synthetic-instrument-{rng.randrange(1000, 10000)}"
        later = rng.randrange(1, 21)
        payments = []
        for label, days in (("a", 0), ("b", later)):
            payments.append({
                "id": str(ident(case, f"capture-{label}")),
                "invoice_id": str(ident(case, "invoice-b" if label == "b" and case.family == "separate_invoices" else "invoice-a")),
                "amount_cents": case.amount, "currency": case.currency, "payment_method": method,
                "status": "failed" if label == "a" and case.family == "failed_retry" else "captured",
                "captured_at": (datetime(2025, 6, 1, 10, tzinfo=timezone.utc) + timedelta(days=days)).isoformat().replace("+00:00", "Z"),
                "refunded": label == "b" and case.family == "already_refunded",
            })
        messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": case.request}]
        def append_call(name, call_id, arguments, result):
            messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": name, "arguments": json.dumps(arguments)}}]})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
        append_call("list_payments", "list-1", {}, {"payments": payments})
        if case.family == "eligible_duplicate":
            proposal = {"id": str(ident(case, "proposal")), "payment_id": payments[1]["id"],
                        "status": "pending", "amount_cents": case.amount, "currency": case.currency}
            append_call("propose_refund", "proposal-1", {"payment_id": payments[1]["id"]}, {"proposal": proposal})
            final = {"decision": "propose", "message": f"Pending proposal {proposal['id']} for {case.amount} {case.currency} cents."}
        else:
            final = {"decision": "no_refund", "message": "No additional refund proposal is warranted."}
        messages.append({"role": "assistant", "content": json.dumps(final)})
        rows[case.split].append({"dataset_version": VERSION, "seed": case.seed, "scenario_id": case.id,
                                 "template": case.template, "family": case.family, "split": case.split,
                                 "tools": TOOLS, "messages": messages})
    for split, records in rows.items():
        (tmp_path / f"{VERSION}-{split}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return rows


@pytest.mark.parametrize("tamper", ["pair", "grading_metadata", "decision"])
def test_export_audit_rejects_bad_messages(tmp_path, tamper):
    rows = synthetic_export(tmp_path)
    audit(tmp_path, 20261121)
    record = rows["validation"][0]
    if tamper == "pair":
        record["messages"][3]["tool_call_id"] = "wrong"
    elif tamper == "grading_metadata":
        result = json.loads(record["messages"][3]["content"])
        result["payments"][0]["expected_decision"] = "propose"
        record["messages"][3]["content"] = json.dumps(result)
    else:
        record["messages"][-1]["content"] = '{"decision":"no_refund","message":"None"}'
    (tmp_path / f"{VERSION}-validation.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows["validation"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit(tmp_path, 20261121)


def test_validation_positive_replays_with_no_new_refund():
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        pytest.skip("Disposable dev/test database opt-in required.")
    engine = create_database_engine()
    try:
        with isolated_database(engine) as isolated:
            app.state.session_factory = sessionmaker(isolated, expire_on_commit=False)
            try:
                with TestClient(app) as client:
                    case = next(c for c in scenarios(20261121)
                                if c.split == "validation" and c.family == "eligible_duplicate")
                    messages = replay(isolated, client, case)
                    result = json.loads(messages[-2]["content"])["proposal"]
                    assert result["payment_id"] == str(ident(case, "capture-b"))
                    assert (result["amount_cents"], result["currency"], result["status"]) == (
                        case.amount, case.currency, "pending")
                    assert refund_snapshot(isolated) == []
                    replay(isolated, client, case, messages)
                    assert refund_snapshot(isolated) == []
            finally:
                del app.state.session_factory
    finally:
        engine.dispose()
