import json
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from ops_agent.agent import MAX_MODEL_CALLS, ModelFailure, ModelTurn, OpenAIModel, ToolCall, run_agent

PAYMENT_ID = uuid.UUID("11111111-2222-4333-8444-555555555555")
PROPOSAL_ID = uuid.UUID("22222222-2222-4333-8444-555555555555")


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.calls = 0

    def next_turn(self, history):
        self.calls += 1
        return next(self.turns)


def call(name, arguments="{}", call_id="call-1"):
    return ModelTurn([ToolCall(call_id, name, arguments)])


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_confirmed_backend_proposal_and_untrusted_final_text():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/payments":
            return httpx.Response(200, json=[])
        assert request.url.path == "/refund-proposals"
        assert json.loads(request.content)["payment_id"] == str(PAYMENT_ID)
        return httpx.Response(201, json={"id": str(PROPOSAL_ID), "payment_id": str(PAYMENT_ID),
                                        "status": "pending", "amount_cents": 2900, "currency": "CAD"})

    model = ScriptedModel([call("list_payments"), call("propose_refund", json.dumps({"payment_id": str(PAYMENT_ID)})),
                           ModelTurn([], "Refund successfully approved!")])
    with client(handler) as backend:
        result = run_agent(model, backend, "Find duplicate payments")
    assert result.error is None
    assert result.proposals[0]["id"] == str(PROPOSAL_ID)
    assert result.proposals[0]["amount_cents"] == 2900
    assert paths == ["/payments", "/refund-proposals"]
    assert "approved" not in repr(result).lower()


@pytest.mark.parametrize("name,arguments", [
    ("list_payments", '{"extra":1}'),
    ("list_payments", "not-json"),
    ("propose_refund", "{}"),
    ("propose_refund", '{"payment_id":"not-a-uuid"}'),
    ("propose_refund", '{"payment_id":"11111111-2222-4333-8444-555555555555","amount":1}'),
    ("approve_refund", "{}"),
])
def test_invalid_or_unavailable_tools_cannot_call_backend(name, arguments):
    def handler(request):
        pytest.fail("Invalid tool must not reach backend")

    with client(handler) as backend:
        result = run_agent(ScriptedModel([call(name, arguments)]), backend, "yes")
    assert result.error
    assert result.trace[0]["result"]["error"] == result.error
    assert result.proposals == []


def test_yes_cannot_invoke_approval_tool():
    with client(lambda request: pytest.fail("Must not call backend")) as backend:
        result = run_agent(ScriptedModel([call("approve_refund")]), backend, "yes")
    assert "approval is not available" in result.error
    assert not result.proposals


def test_backend_rejection_visible_and_never_reported_as_proposal():
    with client(lambda request: httpx.Response(409, json={"detail": "not eligible"})) as backend:
        result = run_agent(ScriptedModel([call("propose_refund", json.dumps({"payment_id": str(PAYMENT_ID)}))]),
                           backend, "Investigate")
    assert result.error == "Backend rejected the tool request (HTTP 409)."
    assert not result.proposals
    assert result.trace[0]["result"]["error"] == result.error


def test_invalid_backend_proposal_is_not_reported():
    with client(lambda request: httpx.Response(201, json={"id": str(PROPOSAL_ID), "payment_id": str(PAYMENT_ID),
                                                          "status": "approved", "amount_cents": 2900, "currency": "CAD"})) as backend:
        result = run_agent(ScriptedModel([call("propose_refund", json.dumps({"payment_id": str(PAYMENT_ID)}))]),
                           backend, "Investigate")
    assert result.error == "Backend returned a mismatched proposal."
    assert not result.proposals


def test_model_call_limit_and_model_errors_are_explicit():
    model = ScriptedModel([call("list_payments", call_id=str(n)) for n in range(MAX_MODEL_CALLS)])
    with client(lambda request: httpx.Response(200, json=[])) as backend:
        result = run_agent(model, backend, "Investigate")
    assert model.calls == MAX_MODEL_CALLS
    assert result.error == "Model-call limit reached; workflow did not complete."

    class FailedModel:
        def next_turn(self, history):
            raise ModelFailure("Model unavailable.")

    with client(lambda request: pytest.fail("Must not call backend")) as backend:
        result = run_agent(FailedModel(), backend, "Investigate")
    assert result.error == "Model unavailable."


def test_model_hallucination_without_tools_does_not_create_success():
    with client(lambda request: pytest.fail("Must not call backend")) as backend:
        result = run_agent(ScriptedModel([ModelTurn([], "I refunded it.")]), backend, "yes")
    assert result.proposals == []
    assert result.trace == []
    assert result.error is None


def test_hosted_model_requests_structured_decision_and_reports_usage():
    model = OpenAIModel.__new__(OpenAIModel)
    model.model = "test-model"
    model.client = Mock()
    model.client.responses.create.return_value = SimpleNamespace(
        status="completed", output=[], output_text='{"decision":"no_refund","message":"None."}', model="snapshot-1",
        usage=SimpleNamespace(input_tokens=42, output_tokens=7),
    )
    turn = model.next_turn([{"role": "user", "content": "inspect"}])
    assert turn.input_tokens == 42 and turn.output_tokens == 7
    assert turn.reported_model == "snapshot-1"
    format_spec = model.client.responses.create.call_args.kwargs["text"]["format"]
    assert format_spec["type"] == "json_schema"
    assert format_spec["schema"]["required"] == ["decision", "message"]
