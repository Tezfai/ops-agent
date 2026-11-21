"""Run the hosted model against the local simulated backend."""

import argparse
import json
import sys

import httpx

from ops_agent.agent import ModelFailure, OpenAIModel, run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulated refund proposal demo; never approves refunds.")
    parser.add_argument("message", help="Use simulated data only; the message is sent to OpenAI.")
    args = parser.parse_args()
    try:
        model = OpenAIModel()
    except ModelFailure as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0) as client:
            result = run_agent(model, client, args.message)
    finally:
        model.close()
    print(json.dumps({"trace": result.trace, "proposals": result.proposals, "error": result.error}, indent=2))
    try:
        final = json.loads(result.final_text)
    except (ValueError, TypeError):
        final = None
    if (isinstance(final, dict) and final.get("decision") == "clarify"
            and isinstance(final.get("message"), str) and "?" in final["message"] and not result.proposals):
        print("Clarification needed: Which payment should I investigate, and what action would you like?")
    print("No refund was approved or executed by the agent.")
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
