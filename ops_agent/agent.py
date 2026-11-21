"""Bounded model-assisted proposal workflow; approval is never an agent tool."""

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError

MAX_MODEL_CALLS = 4
MAX_TOOLS_PER_CALL = 4
INSTRUCTIONS = (
    "You investigate simulated duplicate charges. Use list_payments first. "
    "A duplicate must be a later captured payment on the same invoice with the same "
    "amount, currency, and payment method. Ask the backend to propose a refund only "
    "for an apparent duplicate. The backend decides eligibility. Never claim a refund "
    "is executed or approved; approval is separate. If the user only says yes, do not "
    "interpret this as approval or create a new proposal. Never propose for an already "
    "refunded payment. If the request is ambiguous, ask a clarifying question instead "
    "of choosing. Finish with a JSON object with exactly two fields: "
    '"decision" (one of "propose", "no_refund", or "clarify") and "message" '
    "(a short explanation or a specific clarifying question). "
    "Report propose only when the backend confirmed a proposal."
)
TOOLS = [
    {"type": "function", "name": "list_payments", "description": "List the demo customer's payments.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
     "strict": True},
    {"type": "function", "name": "propose_refund", "description": "Create a pending proposal for an eligible duplicate.",
     "parameters": {"type": "object", "properties": {"payment_id": {"type": "string"}},
                    "required": ["payment_id"], "additionalProperties": False}, "strict": True},
]
DECISION_FORMAT = {
    "type": "json_schema",
    "name": "refund_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["propose", "no_refund", "clarify"]},
            "message": {"type": "string"},
        },
        "required": ["decision", "message"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelTurn:
    calls: list[ToolCall]
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    reported_model: str | None = None


class Model(Protocol):
    def next_turn(self, history: list[dict[str, Any]]) -> ModelTurn: ...


class ModelFailure(Exception):
    pass


class OpenAIModel:
    def __init__(self) -> None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        if not key or not self.model:
            raise ModelFailure("OPENAI_API_KEY and OPENAI_MODEL must be configured.")
        self.client = openai.OpenAI(api_key=key, max_retries=0, timeout=30.0)

    def close(self) -> None:
        self.client.close()

    def next_turn(self, history: list[dict[str, Any]]) -> ModelTurn:
        try:
            response = self.client.responses.create(
                model=self.model, instructions=INSTRUCTIONS, input=history, tools=TOOLS,
                max_output_tokens=500, parallel_tool_calls=False, store=False,
                text={"format": DECISION_FORMAT},
            )
        except openai.APIStatusError as exc:
            raise ModelFailure(f"Model request failed (HTTP {exc.status_code}); check model access and quota.") from None
        except openai.APIConnectionError:
            raise ModelFailure("Model connection failed; check network access.") from None
        except openai.APIError:
            raise ModelFailure("Model request failed; check the model configuration.") from None
        if response.status != "completed":
            raise ModelFailure("Model response was incomplete.")
        calls = [
            ToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments)
            for item in response.output if item.type == "function_call"
        ]
        usage = response.usage
        return ModelTurn(calls=calls, text=response.output_text,
                         input_tokens=usage.input_tokens if usage else None,
                         output_tokens=usage.output_tokens if usage else None,
                         reported_model=response.model)


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposalArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payment_id: uuid.UUID


class SavedProposal(BaseModel):
    id: uuid.UUID
    payment_id: uuid.UUID
    status: str
    amount_cents: int
    currency: str


@dataclass
class AgentResult:
    trace: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None
    final_text: str = field(default="", repr=False)
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_calls: int = 0
    usage_incomplete: bool = field(default=False, repr=False)
    reported_models: list[str] = field(default_factory=list)


def backend_request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.RequestError:
        raise ModelFailure("Backend connection failed.") from None
    if not response.is_success:
        # The backend's status is useful; arbitrary response bodies are not trusted.
        raise ModelFailure(f"Backend rejected the tool request (HTTP {response.status_code}).")
    try:
        result = response.json()
    except ValueError:
        raise ModelFailure("Backend returned invalid JSON.") from None
    if not isinstance(result, (dict, list)):
        raise ModelFailure("Backend returned an unexpected response.")
    return result


def execute_tool(client: httpx.Client, call: ToolCall) -> dict[str, Any]:
    try:
        if call.name == "list_payments":
            NoArguments.model_validate_json(call.arguments)
            payments = backend_request(client, "GET", "/payments")
            if not isinstance(payments, list):
                raise ModelFailure("Backend returned an unexpected payment list.")
            return {"payments": payments}
        if call.name == "propose_refund":
            args = ProposalArguments.model_validate_json(call.arguments)
            proposal = backend_request(client, "POST", "/refund-proposals", json={"payment_id": str(args.payment_id)})
            if not isinstance(proposal, dict):
                raise ModelFailure("Backend returned an unexpected proposal.")
            try:
                saved = SavedProposal.model_validate(proposal)
            except ValidationError:
                raise ModelFailure("Backend returned an invalid proposal.") from None
            if saved.payment_id != args.payment_id or saved.status != "pending" or saved.amount_cents <= 0:
                raise ModelFailure("Backend returned a mismatched proposal.")
            return {"proposal": saved.model_dump(mode="json")}
        raise ModelFailure("Unknown tool requested; approval is not available.")
    except ValidationError:
        raise ModelFailure("Invalid tool arguments.") from None


def run_agent(model: Model, client: httpx.Client, message: str) -> AgentResult:
    result = AgentResult()
    history: list[dict[str, Any]] = [{"role": "user", "content": message}]
    for _ in range(MAX_MODEL_CALLS):
        try:
            turn = model.next_turn(history)
        except ModelFailure as exc:
            result.error = str(exc)
            result.error_kind = "model"
            return result
        result.model_calls += 1
        if turn.reported_model and turn.reported_model not in result.reported_models:
            result.reported_models.append(turn.reported_model)
        if turn.input_tokens is None or turn.output_tokens is None:
            result.usage_incomplete = True
            result.input_tokens = None
            result.output_tokens = None
        elif not result.usage_incomplete:
            result.input_tokens = (result.input_tokens or 0) + turn.input_tokens
            result.output_tokens = (result.output_tokens or 0) + turn.output_tokens
        if not turn.calls:
            if not turn.text.strip():
                result.error = "Model returned neither a tool call nor a response."
                result.error_kind = "model"
            else:
                result.final_text = turn.text
            return result
        if len(turn.calls) > MAX_TOOLS_PER_CALL:
            result.error = "Model exceeded the tool-call limit."
            result.error_kind = "model"
            return result
        for call in turn.calls:
            history.append({"type": "function_call", "call_id": call.call_id,
                            "name": call.name, "arguments": call.arguments})
            try:
                output = execute_tool(client, call)
                result.trace.append({"tool": call.name, "result": output})
                if "proposal" in output:
                    result.proposals.append(output["proposal"])
            except ModelFailure as exc:
                output = {"error": str(exc)}
                result.trace.append({"tool": call.name, "result": output})
                result.error = str(exc)
                result.error_kind = "tool"
            history.append({"type": "function_call_output", "call_id": call.call_id,
                            "output": json.dumps(output)})
        if result.error:
            return result
    result.error = "Model-call limit reached; workflow did not complete."
    result.error_kind = "model"
    return result
