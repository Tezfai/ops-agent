import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from ops_agent.database import create_database_engine
from ops_agent.main import app
from scripts.generate_sft_dataset import TEMPLATES, replay, scenarios
from scripts.run_scenarios import isolated_database


def test_seeded_generation_and_template_split():
    cases = scenarios(20261121)
    assert len(cases) == 40
    assert cases == scenarios(20261121)
    assert cases != scenarios(20261122)
    assert len({case.id for case in cases}) == 40
    train = {case.template for case in cases if case.split == "train"}
    validation = {case.template for case in cases if case.split == "validation"}
    assert train.isdisjoint(validation)
    assert {case.family for case in cases if case.split == "train"}.isdisjoint(
        {case.family for case in cases if case.split == "validation"}
    )
    for family in {family for _, family, _ in TEMPLATES}:
        assert sum(case.family == family for case in cases) == 10
    assert {case.family for case in cases} == {
        "eligible_duplicate", "separate_invoices", "failed_retry", "already_refunded"
    }


@pytest.mark.parametrize("tamper", ["payment", "amount", "proposal_id", "decision", "missing_call"])
def test_replay_rejects_tampered_demonstrations(tamper):
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        pytest.skip("Disposable dev/test database opt-in required.")
    engine = create_database_engine()
    try:
        with isolated_database(engine) as isolated:
            app.state.session_factory = sessionmaker(isolated, expire_on_commit=False)
            try:
                with TestClient(app) as client:
                    case = scenarios(20261121)[0]
                    messages = replay(isolated, client, case)
                    if tamper == "missing_call":
                        messages = messages[:4] + messages[6:]
                    elif tamper == "decision":
                        messages[-1]["content"] = '{"decision":"no_refund","message":"None."}'
                    else:
                        result = json.loads(messages[-2]["content"])
                        key = {"payment": "payment_id", "amount": "amount_cents", "proposal_id": "id"}[tamper]
                        result["proposal"][key] = str(uuid.uuid4()) if tamper == "proposal_id" else "wrong" if tamper == "payment" else 1
                        messages[-2]["content"] = json.dumps(result, sort_keys=True, separators=(",", ":"))
                    with pytest.raises(ValueError):
                        replay(isolated, client, case, messages)
            finally:
                del app.state.session_factory
    finally:
        engine.dispose()
