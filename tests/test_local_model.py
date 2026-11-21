import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from ops_agent.agent import INSTRUCTIONS, ModelFailure, run_agent
from ops_agent.local_model import LocalChatModel


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("SMALL_MODEL", "test-open-instruct")
    monkeypatch.setenv("SMALL_MODEL_REVISION", "test-revision")
    monkeypatch.setenv("SMALL_MODEL_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.delenv("SMALL_MODEL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SMALL_MODEL_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("SMALL_MODEL_DECISION_SCHEMA", raising=False)


def test_adapter_tool_exchange_and_no_hardcoded_tool_choice(configured):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "type": "function",
                 "function": {"name": "list_payments", "arguments": "{}"}}
            ]}
            finish = "tool_calls"
        else:
            assert body["messages"][-1]["role"] == "tool"
            assert json.loads(body["messages"][-1]["content"])["payments"][0]["id"] == "synthetic"
            message = {"role": "assistant", "content": '{"decision":"no_refund","message":"One payment."}'}
            finish = "stop"
        return httpx.Response(200, json={
            "id": "response", "object": "chat.completion", "created": 1, "model": "test-open-instruct",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        })

    model = LocalChatModel()
    model.client.close()
    import openai
    model.client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-placeholder",
                                 max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        backend = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=[{"id": "synthetic"}])), base_url="http://test")
        with backend:
            result = run_agent(model, backend, "Inspect")
    finally:
        model.close()
    assert result.error is None
    assert json.loads(result.final_text)["decision"] == "no_refund"
    assert result.input_tokens == 20 and result.output_tokens == 6
    assert requests[0]["messages"][0]["content"] == INSTRUCTIONS
    assert requests[0]["tool_choice"] == "auto"
    assert requests[0]["max_tokens"] == 500
    assert "reasoning_effort" not in requests[0]
    assert "response_format" not in requests[0]
    assert "response_format" not in requests[1]
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["list_payments", "propose_refund"]


@pytest.mark.parametrize("endpoint", ["https://example.com/v1", "http://example.com/v1",
                                      "http://user:pass@127.0.0.1:8000/v1",
                                      "http://127.0.0.1:8000/v1?key=x"])
def test_adapter_rejects_nonlocal_or_credentialed_urls(configured, monkeypatch, endpoint):
    monkeypatch.setenv("SMALL_MODEL_BASE_URL", endpoint)
    with pytest.raises(ModelFailure):
        LocalChatModel()


def test_adapter_rejects_incomplete_response(configured):
    with patch("ops_agent.local_model.openai.OpenAI") as factory:
        model = LocalChatModel()
        factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length",
                                     message=SimpleNamespace(content=None, tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=500))
        with pytest.raises(ModelFailure, match=r"finish_reason=length, max_tokens=500, prompt_tokens=42, "
                                                r"completion_tokens=500, has_text=False, has_tool_calls=False"):
            model.next_turn([{"role": "user", "content": "hello"}])
        model.close()


def test_incomplete_response_does_not_expose_untrusted_reason_or_content(configured):
    with patch("ops_agent.local_model.openai.OpenAI") as factory:
        model = LocalChatModel()
        factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="sensitive data",
                                     message=SimpleNamespace(content="sensitive text", tool_calls=[object()]))],
            usage=None)
        with pytest.raises(ModelFailure) as error:
            model.next_turn([{"role": "user", "content": "hello"}])
        assert "finish_reason=other_or_missing" in str(error.value)
        assert "has_text=True, has_tool_calls=True" in str(error.value)
        assert "sensitive" not in str(error.value)
        model.close()


def test_configured_token_limit_is_sent_and_reported(configured, monkeypatch):
    monkeypatch.setenv("SMALL_MODEL_MAX_TOKENS", "1024")
    with patch("ops_agent.local_model.openai.OpenAI") as factory:
        model = LocalChatModel()
        factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length",
                                     message=SimpleNamespace(content=None, tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=1024))
        with pytest.raises(ModelFailure, match="finish_reason=length, max_tokens=1024"):
            model.next_turn([{"role": "user", "content": "hello"}])
        assert factory.return_value.chat.completions.create.call_args.kwargs["max_tokens"] == 1024
        assert model.settings["max_tokens"] == 1024
        model.close()


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "invalid"])
def test_invalid_token_limit_is_rejected(configured, monkeypatch, value):
    monkeypatch.setenv("SMALL_MODEL_MAX_TOKENS", value)
    with pytest.raises(ModelFailure, match="SMALL_MODEL_MAX_TOKENS must be a positive integer"):
        LocalChatModel()


def test_reasoning_effort_none_is_sent_and_recorded(configured, monkeypatch):
    monkeypatch.setenv("SMALL_MODEL_REASONING_EFFORT", "none")
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 1, "model": "test-open-instruct",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"decision":"no_refund","message":"None."}'}}],
        })

    model = LocalChatModel()
    model.client.close()
    import openai
    model.client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-placeholder",
                                 max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        model.next_turn([{"role": "user", "content": "inspect"}])
        assert model.settings["reasoning_effort"] == "none"
        assert sent[0]["reasoning_effort"] == "none"
    finally:
        model.close()


@pytest.mark.parametrize("value", ["", "low", "NONE", "true"])
def test_invalid_reasoning_effort_is_rejected(configured, monkeypatch, value):
    monkeypatch.setenv("SMALL_MODEL_REASONING_EFFORT", value)
    with pytest.raises(ModelFailure, match="SMALL_MODEL_REASONING_EFFORT must be unset or 'none'"):
        LocalChatModel()


def test_decision_schema_is_opt_in_after_tool_result(configured, monkeypatch):
    monkeypatch.setenv("SMALL_MODEL_DECISION_SCHEMA", "1")
    with patch("ops_agent.local_model.openai.OpenAI") as factory:
        model = LocalChatModel()
        factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop",
                                     message=SimpleNamespace(content="{}", tool_calls=[]))],
            usage=None, model="test-model")
        model.next_turn([{"role": "user", "content": "inspect"}])
        assert "response_format" not in factory.return_value.chat.completions.create.call_args.kwargs
        model.next_turn([{"role": "user", "content": "inspect"},
                         {"type": "function_call", "call_id": "c", "name": "list_payments", "arguments": "{}"},
                         {"type": "function_call_output", "call_id": "c", "output": '{"payments":[]}'}])
        kwargs = factory.return_value.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["json_schema"]["schema"]["required"] == ["decision", "message"]
        assert kwargs["tool_choice"] == "auto"
        assert model.settings["response_format"] == "JSON schema after tool result"
        model.close()
