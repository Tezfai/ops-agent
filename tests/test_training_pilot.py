import json
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from ops_agent.agent import INSTRUCTIONS, TOOLS, AgentResult
from training.pilot import CONFIG, encode_turns, grade_result, pad_examples


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(CONFIG["checkpoint"], revision=CONFIG["revision"],
                                         use_fast=True, local_files_only=True)


def example(positive=True):
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": "Investigate the synthetic invoice."},
        {"role": "assistant", "tool_calls": [{"id": "l", "type": "function",
         "function": {"name": "list_payments", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "l", "content": '{"payments":["MASKED_TOOL_RESULT_MARKER"]}'},
    ]
    if positive:
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": "p", "type": "function",
             "function": {"name": "propose_refund", "arguments": '{"payment_id":"00000000-0000-0000-0000-000000000001"}'}}]},
            {"role": "tool", "tool_call_id": "p", "content": '{"proposal":"MASKED_PROPOSAL_RESULT_MARKER"}'},
        ])
    messages.append({"role": "assistant", "content": json.dumps({
        "decision": "propose" if positive else "no_refund", "message": "Synthetic decision."})})
    return {"scenario_id": "mini", "tools": TOOLS, "messages": messages,
            "expected_decision": "HIDDEN_GRADING_METADATA"}


@pytest.mark.parametrize("positive", [True, False])
def test_actual_template_masks_context_and_supervises_complete_turns(tokenizer, positive):
    encoded = encode_turns(tokenizer, example(positive))
    for turn in encoded:
        supervised = tokenizer.decode([token for token in turn["labels"] if token != -100])
        assert supervised == turn["target"]
        assert "MASKED_TOOL_RESULT_MARKER" not in supervised
        assert "MASKED_PROPOSAL_RESULT_MARKER" not in supervised
        assert "HIDDEN_GRADING_METADATA" not in tokenizer.decode(turn["input_ids"])
        if "MASKED_TOOL_RESULT_MARKER" in turn["masked_prefix"]:
            assert "MASKED_TOOL_RESULT_MARKER" in tokenizer.decode(turn["input_ids"])
            assert "MASKED_TOOL_RESULT_MARKER" not in supervised
    assert '"decision"' in encoded[-1]["target"]
    if positive:
        proposal = next(turn for turn in encoded if "propose_refund" in turn["target"])
        assert "00000000-0000-0000-0000-000000000001" in proposal["target"]
        assert all(token != -100 for token in proposal["labels"][-3:])


def test_rejects_truncation_instead_of_losing_final_decision(tokenizer, monkeypatch):
    monkeypatch.setitem(CONFIG, "max_sequence_tokens", 64)
    with pytest.raises(ValueError, match="refusing truncation"):
        encode_turns(tokenizer, example())


def test_padding_is_never_supervised():
    padded = pad_examples([
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [3], "attention_mask": [1], "labels": [3]},
    ], 0)
    assert padded["input_ids"] == [[1, 2], [3, 0]]
    assert padded["attention_mask"] == [[1, 1], [1, 0]]
    assert padded["labels"] == [[-100, 2], [3, -100]]


def test_fixture_helpers_do_not_import_benchmark_modules():
    code = (
        "import sys\n"
        "from scripts.generate_sft_v11 import scenarios\n"
        "print(len(scenarios(1)))\n"
        "print(any(name == 'benchmarks' or name.startswith('benchmarks.') for name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["40", "False"]


def test_grade_checks_persisted_target_and_no_new_refunds():
    payment, proposal = uuid.uuid4(), uuid.uuid4()
    case = SimpleNamespace(id="mini", family="eligible_duplicate", amount=2900, currency="CAD")
    result = AgentResult(trace=[{"tool": "list_payments"}],
                         proposals=[{"id": str(proposal), "amount_cents": 2900, "currency": "CAD"}],
                         final_text='{"decision":"propose","message":"Pending."}')
    after = {(proposal, payment, "pending")}
    assert grade_result(case, payment, set(), after, [], [], result)["passed"]
    assert not grade_result(case, uuid.uuid4(), set(), after, [], [], result)["passed"]
    assert not grade_result(case, payment, set(), after, [], [("new refund",)], result)["passed"]
    assert grade_result(case, None, set(), after, [], [], result)["false_proposal"]
