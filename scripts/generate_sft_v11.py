"""Generate and audit balanced, replayed synthetic SFT demonstrations."""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import dotenv_values
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from ops_agent.agent import INSTRUCTIONS, TOOLS, ModelFailure
from ops_agent.database import create_database_engine
from ops_agent.main import app

if __package__:
    from scripts.generate_sft_dataset import Scenario, ident, replay
    from scripts.run_scenarios import isolated_database
else:
    from generate_sft_dataset import Scenario, ident, replay
    from run_scenarios import isolated_database


VERSION = "synthetic-refund-sft-v1.1"
TEMPLATES = (
    ("duplicate_time", "eligible_duplicate", "train"),
    ("duplicate_by_id", "eligible_duplicate", "validation"),
    ("separate_invoice_ids", "separate_invoices", "train"),
    ("separate_payment_ids", "separate_invoices", "validation"),
    ("retry_disclosed", "failed_retry", "train"),
    ("retry_investigate", "failed_retry", "validation"),
    ("refund_disclosed", "already_refunded", "train"),
    ("refund_investigate", "already_refunded", "validation"),
)
FAMILIES = {family for _, family, _ in TEMPLATES}


def request_for(case: Scenario) -> str:
    invoice_a, invoice_b = str(ident(case, "invoice-a")), str(ident(case, "invoice-b"))
    a, b = str(ident(case, "capture-a")), str(ident(case, "capture-b"))
    requests = {
        "duplicate_time": "Review the later of my two captures and request a proposal if it qualifies.",
        "duplicate_by_id": f"Please check payment {b} for an extra capture and request a proposal if eligible.",
        "separate_invoice_ids": f"Check invoices {invoice_a} and {invoice_b} for any extra capture.",
        "separate_payment_ids": f"These equal payments {a} and {b} may be duplicates; investigate before proposing.",
        "retry_disclosed": "My earlier attempt failed before the later one succeeded. Check for an extra capture.",
        "retry_investigate": f"Review payment {b} and its preceding attempt for any extra successful capture.",
        "refund_disclosed": "The extra capture was returned already; check whether any further proposal is needed.",
        "refund_investigate": f"Review payment {b} for an outstanding duplicate and propose only if eligible.",
    }
    return requests[case.template]


def scenarios(seed: int) -> list[Scenario]:
    result = []
    for template, family, split in TEMPLATES:
        for variant in range(5):
            rng = random.Random(f"{VERSION}:{seed}:{template}:{variant}")
            case = Scenario(f"{template}-{variant + 1:02d}", template, family, split, "",
                            rng.randrange(201, 19999) * 7 + 3, rng.choice(("CAD", "USD", "EUR")),
                            seed, variant, VERSION)
            result.append(Scenario(**{**vars(case), "request": request_for(case)}))
    return result


def normalize(messages: list[dict]) -> str:
    """Mask substituted values while retaining reference relationships and structure."""
    identifiers: dict[str, str] = {}
    value = json.dumps(messages, sort_keys=True, separators=(",", ":"))

    def mask_uuid(match: re.Match) -> str:
        key = match.group(0)
        if key not in identifiers:
            identifiers[key] = f"<id{len(identifiers) + 1}>"
        return identifiers[key]

    value = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                   mask_uuid, value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\b", "<timestamp>", value)
    value = re.sub(r"synthetic-instrument-\d+", "<instrument>", value)
    value = re.sub(r"\b(?:CAD|USD|EUR)\b", "<currency>", value)
    return re.sub(r"(?<![A-Za-z])\d+(?![A-Za-z])", "<number>", value)


def validate_distribution(families: Counter, decisions: Counter, split: str) -> None:
    if families != Counter({family: 5 for family in FAMILIES}) or decisions != Counter({"propose": 5, "no_refund": 15}):
        raise ValueError(f"{split} family or decision counts are incorrect.")


def audit(output: Path, seed: int) -> dict:
    cases = {case.id: case for case in scenarios(seed)}
    split_templates: dict[str, set[str]] = {}
    normalized: dict[str, set[str]] = {}
    counts = {}
    secrets = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    for key, value in os.environ.items():
        if any(part in key.upper() for part in ("KEY", "SECRET", "TOKEN", "PASSWORD", "DATABASE_URL")):
            secrets[key] = value
    for split in ("train", "validation"):
        path = output / f"{VERSION}-{split}.jsonl"
        raw = path.read_text(encoding="utf-8")
        for key, secret in secrets.items():
            if (any(part in key.upper() for part in ("KEY", "SECRET", "TOKEN", "PASSWORD", "DATABASE_URL"))
                    and secret and len(secret) >= 8 and secret in raw):
                raise ValueError("A configured credential was found in the export.")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != 20:
            raise ValueError(f"{split} must contain exactly 20 records.")
        families: Counter = Counter()
        decisions: Counter = Counter()
        split_templates[split] = set()
        normalized[split] = set()
        seen: set[str] = set()
        for row in rows:
            case = cases[row["scenario_id"]]
            if (case.id in seen or case.split != split or row["split"] != split
                    or row["dataset_version"] != VERSION or row["seed"] != seed
                    or row["family"] != case.family or row["template"] != case.template
                    or row["tools"] != TOOLS):
                raise ValueError("Export metadata or tool definitions do not match the scenario.")
            seen.add(case.id)
            families[case.family] += 1
            split_templates[split].add(case.template)
            messages = row["messages"]
            if messages[:2] != [{"role": "system", "content": INSTRUCTIONS},
                                {"role": "user", "content": case.request}]:
                raise ValueError("Unexpected system or user message.")
            expected_calls = ["list_payments"] + (["propose_refund"] if case.family == "eligible_duplicate" else [])
            if len(messages) != 3 + 2 * len(expected_calls):
                raise ValueError("Unexpected message count.")
            proposal = None
            for index, name in enumerate(expected_calls):
                call_message, result_message = messages[2 + 2 * index:4 + 2 * index]
                if set(call_message) != {"role", "tool_calls"} or call_message["role"] != "assistant" or len(call_message["tool_calls"]) != 1:
                    raise ValueError("Invalid assistant tool-call message.")
                call = call_message["tool_calls"][0]
                expected_id = "list-1" if name == "list_payments" else "proposal-1"
                if (set(call) != {"id", "type", "function"} or call["id"] != expected_id
                        or call["type"] != "function" or set(call["function"]) != {"name", "arguments"}
                        or call["function"]["name"] != name
                        or result_message.keys() != {"role", "tool_call_id", "content"}
                        or result_message["role"] != "tool" or result_message["tool_call_id"] != expected_id):
                    raise ValueError("Invalid tool-call/result pairing.")
                args = json.loads(call["function"]["arguments"])
                result = json.loads(result_message["content"])
                if name == "list_payments":
                    if args != {} or set(result) != {"payments"} or len(result["payments"]) != 2:
                        raise ValueError("Invalid payment list result.")
                    payment_keys = {"id", "invoice_id", "amount_cents", "currency", "payment_method",
                                    "status", "captured_at", "refunded"}
                    payments = result["payments"]
                    if any(set(payment) != payment_keys for payment in payments):
                        raise ValueError("Unexpected payment fields in model-visible results.")
                    a, b = str(ident(case, "capture-a")), str(ident(case, "capture-b"))
                    if [payment["id"] for payment in payments] != [a, b]:
                        raise ValueError("Unexpected exported payment order or IDs.")
                    rng = random.Random(f"{case.seed}:{case.id}:layout")
                    method = f"synthetic-instrument-{rng.randrange(1000, 10000)}"
                    later = rng.randrange(1, 21)
                    for payment in payments:
                        invoice_name = "invoice-b" if case.family == "separate_invoices" and payment["id"] == b else "invoice-a"
                        expected_status = "failed" if case.family == "failed_retry" and payment["id"] == a else "captured"
                        timestamp = (datetime(2025, 6, 1, 10, tzinfo=timezone.utc)
                                     + timedelta(days=later if payment["id"] == b else 0)).isoformat().replace("+00:00", "Z")
                        if (payment["invoice_id"] != str(ident(case, invoice_name))
                                or payment["amount_cents"] != case.amount or payment["currency"] != case.currency
                                or payment["payment_method"] != method or payment["captured_at"] != timestamp
                                or payment["status"] != expected_status
                                or payment["refunded"] != (case.family == "already_refunded" and payment["id"] == b)):
                            raise ValueError("Incorrect exported payment fixture.")
                else:
                    target = str(ident(case, "capture-b"))
                    proposal = result["proposal"]
                    if (args != {"payment_id": target} or set(result) != {"proposal"}
                            or set(proposal) != {"id", "payment_id", "status", "amount_cents", "currency"}
                            or (proposal["payment_id"], proposal["status"], proposal["amount_cents"], proposal["currency"])
                            != (target, "pending", case.amount, case.currency)):
                        raise ValueError("Incorrect exported proposal.")
            final_message = messages[-1]
            if set(final_message) != {"role", "content"} or final_message["role"] != "assistant":
                raise ValueError("Invalid final message.")
            final = json.loads(final_message["content"])
            expected = "propose" if proposal else "no_refund"
            expected_message = (
                f"Pending proposal {proposal['id']} for {proposal['amount_cents']} {proposal['currency']} cents."
                if proposal else "No additional refund proposal is warranted."
            )
            if (set(final) != {"decision", "message"} or final["decision"] != expected
                    or final["message"] != expected_message):
                raise ValueError("Invalid exported final decision.")
            decisions[expected] += 1
            normalized[split].add(normalize(messages))
        validate_distribution(families, decisions, split)
        counts[split] = {"families": dict(families), "decisions": dict(decisions)}
    if split_templates["train"] & split_templates["validation"]:
        raise ValueError("A template appears in both splits.")
    if normalized["train"] & normalized["validation"]:
        raise ValueError("Normalized messages overlap across splits.")
    return {"counts": counts, "normalized_cross_split_copies": 0}


def generate(seed: int, output: Path) -> int:
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new directory.")
    engine = create_database_engine()
    accepted: dict[str, list[dict]] = {"train": [], "validation": []}
    rejected: Counter = Counter()
    try:
        with isolated_database(engine) as isolated:
            app.state.session_factory = sessionmaker(isolated, expire_on_commit=False)
            try:
                with TestClient(app) as client:
                    for case in scenarios(seed):
                        try:
                            messages = replay(isolated, client, case)
                            replay(isolated, client, case, messages)
                        except (ValueError, KeyError, ModelFailure) as exc:
                            rejected[(case.family, str(exc))] += 1
                            continue
                        accepted[case.split].append({
                            "dataset_version": VERSION, "seed": seed, "scenario_id": case.id,
                            "template": case.template, "family": case.family, "split": case.split,
                            "tools": TOOLS, "messages": messages,
                        })
            finally:
                del app.state.session_factory
    finally:
        engine.dispose()
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in accepted.items():
        with (output / f"{VERSION}-{split}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    summary = {"dataset_version": VERSION, "seed": seed,
               "accepted": {split: len(rows) for split, rows in accepted.items()},
               "rejections": [{"family": family, "reason": reason, "count": count}
                              for (family, reason), count in sorted(rejected.items())]}
    try:
        summary["audit"] = audit(output, seed)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        summary["audit_error"] = str(exc)
    with (output / f"{VERSION}-summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    return 1 if rejected or "audit_error" in summary else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", type=Path, default=Path("data") / VERSION)
    args = parser.parse_args()
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        print("Set OPS_AGENT_TEST_USE_DEV_DATABASE=1 only for a disposable database.", file=sys.stderr)
        return 1
    try:
        return generate(args.seed, args.output)
    except FileExistsError:
        print("Output directory already exists; choose a new directory.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Dataset generation failed ({type(exc).__name__}); check the database and output path.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
