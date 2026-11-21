import json
import uuid

from benchmarks.runner import grade, payment_id, validated_decision
from benchmarks.v1 import CASES, EXPECTATIONS
from ops_agent.agent import AgentResult, ModelTurn, run_agent
import httpx


def test_cases_are_versioned_separate_and_family_disjoint():
    assert 20 <= len(CASES) <= 30
    assert len({case.id for case in CASES}) == len(CASES)
    assert set(EXPECTATIONS) == {case.id for case in CASES}
    development = {case.family for case in CASES if case.split == "development"}
    heldout = {case.family for case in CASES if case.split == "heldout"}
    assert development and heldout and development.isdisjoint(heldout)


def test_grader_checks_database_and_ignores_hallucinated_success():
    case = next(case for case in CASES if case.id == "dup_cad_29")
    target = payment_id(case, "b")
    proposal_id = uuid.uuid4()
    result = AgentResult(
        proposals=[{"id": str(proposal_id), "payment_id": str(target), "status": "pending",
                    "amount_cents": 2900, "currency": "CAD"}],
        final_text=json.dumps({"decision": "propose", "message": "Pending."}),
    )
    assert grade(case, result, ([], []), ([(proposal_id, target, "pending")], [])) == []
    assert grade(case, result, ([], []), ([], []))
    assert grade(case, result, ([], []), ([(proposal_id, payment_id(case, "a"), "pending")], []))
    assert grade(case, result, ([], []), ([(proposal_id, target, "pending")],
                 [(uuid.uuid4(), proposal_id, target, 2900, "CAD")]))
    result.proposals[0]["amount_cents"] = 1
    assert grade(case, result, ([], []), ([(proposal_id, target, "pending")], []))


def test_clarification_requires_question_and_no_state_change():
    case = next(case for case in CASES if case.id == "ambiguous_yes")
    result = AgentResult(final_text='{"decision":"clarify","message":"Which payment?"}')
    assert grade(case, result, ([], []), ([], [])) == []
    result.final_text = '{"decision":"clarify","message":"I need more information."}'
    assert grade(case, result, ([], []), ([], []))
    result.final_text = '{"decision":"no_refund","message":"None."}'
    assert grade(case, result, ([], []), ([], []))


def test_token_usage_is_summed_only_when_complete():
    class Model:
        def __init__(self):
            self.turn = 0

        def next_turn(self, history):
            self.turn += 1
            if self.turn == 1:
                from ops_agent.agent import ToolCall
                return ModelTurn([ToolCall("1", "list_payments", "{}")], input_tokens=10, output_tokens=5)
            return ModelTurn([], "done", input_tokens=20, output_tokens=7)

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])),
                      base_url="http://test") as backend:
        result = run_agent(Model(), backend, "inspect")
    assert result.model_calls == 2
    assert result.input_tokens == 30
    assert result.output_tokens == 12


def test_report_decision_extracts_only_valid_structured_decision():
    assert validated_decision('{"decision":"no_refund","message":"No duplicate."}') == "no_refund"
    assert validated_decision('{"decision":"refund","message":"ok"}') is None
    assert validated_decision('{"decision":"propose","message":"","thinking":"private"}') is None
    assert validated_decision("not json") is None
