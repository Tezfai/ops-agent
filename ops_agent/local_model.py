"""Adapter for a user-supplied local OpenAI-compatible chat-completions server."""

import os
from typing import Any
from urllib.parse import urlsplit

import httpx
import openai

from ops_agent.agent import DECISION_FORMAT, INSTRUCTIONS, TOOLS, ModelFailure, ModelTurn, ToolCall


def create_model():
    return LocalChatModel()


class LocalChatModel:
    def __init__(self) -> None:
        self.model = os.getenv("SMALL_MODEL", "").strip()
        self.revision = os.getenv("SMALL_MODEL_REVISION", "").strip()
        base_url = os.getenv("SMALL_MODEL_BASE_URL", "").strip()
        if not self.model or not self.revision or not base_url:
            raise ModelFailure("Set SMALL_MODEL, SMALL_MODEL_REVISION, and SMALL_MODEL_BASE_URL.")
        token_limit = os.getenv("SMALL_MODEL_MAX_TOKENS", "500")
        if not token_limit.isascii() or not token_limit.isdecimal():
            raise ModelFailure("SMALL_MODEL_MAX_TOKENS must be a positive integer.")
        self.max_tokens = int(token_limit)
        if self.max_tokens < 1:
            raise ModelFailure("SMALL_MODEL_MAX_TOKENS must be a positive integer.")
        self.reasoning_effort = os.getenv("SMALL_MODEL_REASONING_EFFORT")
        if self.reasoning_effort is not None and self.reasoning_effort != "none":
            raise ModelFailure("SMALL_MODEL_REASONING_EFFORT must be unset or 'none'.")
        self.decision_schema = os.getenv("SMALL_MODEL_DECISION_SCHEMA")
        if self.decision_schema is not None and self.decision_schema != "1":
            raise ModelFailure("SMALL_MODEL_DECISION_SCHEMA must be unset or '1'.")
        try:
            parsed = urlsplit(base_url)
            if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError
            _ = parsed.port
        except ValueError:
            raise ModelFailure("SMALL_MODEL_BASE_URL must be an HTTP loopback URL without credentials.") from None
        self.settings = {
            "adapter": "openai-compatible-chat-completions",
            "model_revision": self.revision,
            "endpoint_scope": "loopback",
            "max_tokens": self.max_tokens,
            "max_retries": 0,
            "timeout_seconds": 30,
            "response_format": "prompt-only JSON",
            "prompt_differences": "none",
        }
        if self.reasoning_effort is not None:
            self.settings["reasoning_effort"] = self.reasoning_effort
        if self.decision_schema:
            self.settings["response_format"] = "JSON schema after tool result"
        self.client = openai.OpenAI(
            base_url=base_url, api_key="local-placeholder", max_retries=0, timeout=30.0,
            http_client=httpx.Client(timeout=30.0, trust_env=False),
        )

    def close(self) -> None:
        self.client.close()

    def next_turn(self, history: list[dict[str, Any]]) -> ModelTurn:
        messages: list[dict[str, Any]] = [{"role": "system", "content": INSTRUCTIONS}]
        for item in history:
            if item.get("role") == "user":
                messages.append({"role": "user", "content": item["content"]})
            elif item.get("type") == "function_call":
                tool_call = {
                    "id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
                if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["tool_calls"].append(tool_call)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            elif item.get("type") == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
            else:
                raise ModelFailure("Unsupported agent conversation item.")
        tools = [{"type": "function", "function": {
            "name": tool["name"], "description": tool["description"], "parameters": tool["parameters"],
        }} for tool in TOOLS]
        extra_body = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort is not None else None
        try:
            request = {
                "model": self.model, "messages": messages, "tools": tools,
                "tool_choice": "auto", "max_tokens": self.max_tokens,
            }
            if extra_body is not None:
                request["extra_body"] = extra_body
            if self.decision_schema and any(item.get("type") == "function_call_output" for item in history):
                request["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": DECISION_FORMAT["name"], "schema": DECISION_FORMAT["schema"], "strict": True,
                }}
            response = self.client.chat.completions.create(**request)
        except openai.APIStatusError as exc:
            raise ModelFailure(f"Local model request failed (HTTP {exc.status_code}).") from None
        except openai.APIConnectionError:
            raise ModelFailure("Local model connection failed; check the loopback server.") from None
        except openai.APIError:
            raise ModelFailure("Local model request failed; check server compatibility.") from None
        if not response.choices:
            raise ModelFailure("Local model returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls"):
            known_reasons = {"length", "content_filter", "function_call"}
            reason = choice.finish_reason if choice.finish_reason in known_reasons else "other_or_missing"
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None
            has_text = bool(choice.message.content and choice.message.content.strip())
            has_tool_calls = bool(choice.message.tool_calls)
            raise ModelFailure(
                f"Local model response was incomplete (finish_reason={reason}, max_tokens={self.max_tokens}, "
                f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                f"has_text={has_text}, has_tool_calls={has_tool_calls})."
            )
        calls = []
        for call in choice.message.tool_calls or []:
            if call.type != "function":
                raise ModelFailure("Local model returned an unsupported tool call.")
            calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
        usage = response.usage
        return ModelTurn(calls, choice.message.content or "",
                         input_tokens=usage.prompt_tokens if usage else None,
                         output_tokens=usage.completion_tokens if usage else None,
                         reported_model=response.model)
