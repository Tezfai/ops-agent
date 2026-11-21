"""Frozen Qwen3 LoRA pilot, without benchmark fixture imports."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from ops_agent.agent import INSTRUCTIONS, TOOLS, ModelFailure, ModelTurn, ToolCall, run_agent

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "training" / "pilot_config.json").read_text(encoding="utf-8"))
DATA = ROOT / "data" / "synthetic-refund-sft-v1.1"
TOOLS_CHAT = [{"type": "function", "function": {
    "name": item["name"], "description": item["description"], "parameters": item["parameters"],
}} for item in TOOLS]


def load_split(split):
    path = DATA / f"synthetic-refund-sft-v1.1-{split}.jsonl"
    raw = path.read_bytes()
    expected = CONFIG[f"{split}_sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{split} dataset hash does not match the frozen pilot.")
    records = [json.loads(line) for line in raw.splitlines()]
    if len(records) != 20 or any(row["split"] != split for row in records):
        raise ValueError(f"{split} dataset is not the expected 20-record split.")
    return records


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(CONFIG["checkpoint"], revision=CONFIG["revision"], use_fast=True, token=False)


def chat_messages(record):
    messages = []
    for msg in record["messages"]:
        role = msg["role"]
        if role in ("system", "user", "tool"):
            messages.append({"role": role, "content": msg["content"]})
        elif role == "assistant" and "tool_calls" in msg:
            calls = []
            for call in msg["tool_calls"]:
                calls.append({"type": "function", "function": {
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }})
            messages.append({"role": "assistant", "tool_calls": calls})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})
        else:
            raise ValueError("Unsupported demonstration message.")
    return messages


def encode_turns(tokenizer, record):
    """Render one complete teacher-forced prefix per assistant turn, without truncation."""
    messages = chat_messages(record)
    if messages[0] != {"role": "system", "content": INSTRUCTIONS} or record["tools"] != TOOLS:
        raise ValueError("Training instructions or tools differ from the agent.")
    if messages[-1]["role"] != "assistant" or "content" not in messages[-1]:
        raise ValueError("Final decision is missing.")
    decision = json.loads(messages[-1]["content"])
    if set(decision) != {"decision", "message"} or decision["decision"] not in ("propose", "no_refund", "clarify"):
        raise ValueError("Final decision has an invalid schema.")
    encoded = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tools=TOOLS_CHAT, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(messages[:i + 1], tools=TOOLS_CHAT, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prefix):
            raise ValueError("Chat template does not preserve the no-thinking assistant prefix.")
        tokenized = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        if len(ids) > CONFIG["max_sequence_tokens"]:
            raise ValueError(f"Trace {record['scenario_id']} exceeds the token limit; refusing truncation.")
        if any(start < len(prefix) < end for start, end in offsets):
            raise ValueError("Token boundary crosses the assistant prefix.")
        labels = [token if start >= len(prefix) and end > start else -100
                  for token, (start, end) in zip(ids, offsets)]
        if not any(label != -100 for label in labels):
            raise ValueError("Assistant turn has no supervised tokens.")
        target = full[len(prefix):]
        if "tool_calls" in message:
            for call in message["tool_calls"]:
                if call["function"]["name"] not in target or not isinstance(call["function"]["arguments"], dict):
                    raise ValueError("Tool call is not represented in target.")
        else:
            if message["content"] not in target:
                raise ValueError("Final decision is not represented in target.")
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
                        "kind": "tool_call" if "tool_calls" in message else "decision",
                        "target": target, "masked_prefix": prefix})
    return encoded


def inspect(tokenizer, records):
    turns = [(r, encode_turns(tokenizer, r)) for r in records]
    for family in ("eligible_duplicate", "failed_retry"):
        record, examples = next((r, e) for r, e in turns if r["family"] == family)
        print(f"INSPECT {record['scenario_id']}")
        for example in examples:
            print(f"  {example['kind']}: input={len(example['input_ids'])} "
                  f"supervised={sum(x != -100 for x in example['labels'])}")
            print("  supervised_text:", tokenizer.decode([x for x in example["labels"] if x != -100]))
    print(f"Checked {len(turns)} training traces, {sum(len(e) for _, e in turns)} assistant turns; no truncation.")
    return [e for _, examples in turns for e in examples]


def pad_examples(batch, pad_token_id):
    size = max(len(item["input_ids"]) for item in batch)
    return {key: [item[key] + [pad] * (size - len(item[key])) for item in batch]
            for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}


def train(tokenizer, records, output):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pilot requires a CUDA GPU with bfloat16 support.")
    free, _ = torch.cuda.mem_get_info()
    if free < 20 * 1024**3:
        raise RuntimeError("Pilot requires at least 20 GiB free GPU memory; no weights were loaded.")
    examples = inspect(tokenizer, records)
    set_seed(CONFIG["seed"])
    model = AutoModelForCausalLM.from_pretrained(CONFIG["checkpoint"], revision=CONFIG["revision"],
                                                  torch_dtype=torch.bfloat16, token=False).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=CONFIG["lora_rank"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_targets"], task_type=TaskType.CAUSAL_LM))

    def collate(batch):
        return {key: torch.tensor(value) for key, value in pad_examples(batch, tokenizer.pad_token_id).items()}

    args = TrainingArguments(output_dir=str(output / "trainer"), per_device_train_batch_size=1,
                             gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
                             num_train_epochs=CONFIG["epochs"], learning_rate=CONFIG["learning_rate"],
                             bf16=True, logging_steps=1, save_strategy="no", report_to=[],
                             seed=CONFIG["seed"], data_seed=CONFIG["seed"], remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate)
    trainer.train()
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "tokenizer")


class TransformerModel:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.model_name = CONFIG["checkpoint"]

    def next_turn(self, history):
        import torch
        messages = [{"role": "system", "content": INSTRUCTIONS}]
        for item in history:
            if item.get("role") == "user":
                messages.append({"role": "user", "content": item["content"]})
            elif item.get("type") == "function_call":
                call = {"type": "function", "function": {"name": item["name"],
                        "arguments": json.loads(item["arguments"])}}
                if messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["tool_calls"].append(call)
                else:
                    messages.append({"role": "assistant", "tool_calls": [call]})
            elif item.get("type") == "function_call_output":
                messages.append({"role": "tool", "content": item["output"]})
            else:
                raise ModelFailure("Unsupported agent history.")
        inputs = self.tokenizer.apply_chat_template(messages, tools=TOOLS_CHAT, add_generation_prompt=True,
                                                     enable_thinking=False, return_tensors="pt",
                                                     return_dict=True).to("cuda")
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"],
                                            do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        tokens = generated[0, inputs["input_ids"].shape[1]:]
        if len(tokens) == CONFIG["max_new_tokens"] and tokens[-1].item() != self.tokenizer.eos_token_id:
            raise ModelFailure("Inference truncated at the completion token cap.")
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        calls = []
        chunks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
        if chunks:
            if re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.DOTALL).strip():
                raise ModelFailure("Model mixed tool calls with unexpected text.")
            for index, chunk in enumerate(chunks):
                try:
                    parsed = json.loads(chunk)
                    if set(parsed) != {"name", "arguments"} or not isinstance(parsed["name"], str) or not isinstance(parsed["arguments"], dict):
                        raise ValueError
                except (ValueError, TypeError):
                    raise ModelFailure("Invalid generated tool call.") from None
                calls.append(ToolCall(f"call-{len(history)}-{index}", parsed["name"], json.dumps(parsed["arguments"])))
            return ModelTurn(calls)
        return ModelTurn([], text)


def grade_result(case, target, before, after, before_refunds, after_refunds, result):
    try:
        answer = json.loads(result.final_text)
        valid = (isinstance(answer, dict) and set(answer) == {"decision", "message"}
                 and answer["decision"] in ("propose", "no_refund", "clarify")
                 and isinstance(answer["message"], str) and bool(answer["message"].strip()))
    except (ValueError, TypeError):
        answer, valid = {}, False
    added = after - before
    proposal_ok = (len(added) == 1 and next(iter(added))[1:] == (target, "pending")
                   and len(result.proposals) == 1
                   and result.proposals[0]["amount_cents"] == case.amount
                   and result.proposals[0]["currency"] == case.currency
                   and result.proposals[0]["id"] == str(next(iter(added))[0]))
    expected = "propose" if target else "no_refund"
    ok = (not result.error and valid and answer["decision"] == expected
          and before.issubset(after) and after_refunds == before_refunds
          and any(t["tool"] == "list_payments" for t in result.trace)
          and (proposal_ok if target else not added))
    return {
        "scenario_id": case.id, "family": case.family, "passed": bool(ok),
        "positive": bool(target), "false_proposal": bool(not target and added),
        "invalid_decision": not valid, "truncation": bool(result.error and "truncated" in result.error),
        "error": result.error, "new_refunds": len(set(after_refunds) - set(before_refunds)),
        "tools": [t["tool"] for t in result.trace],
    }


def evaluate(tokenizer, records, output, loaded_model=None, additional_settings=None):
    import torch
    import peft
    import transformers
    from fastapi.testclient import TestClient
    from peft import PeftModel
    from sqlalchemy.orm import sessionmaker
    from transformers import AutoModelForCausalLM
    from ops_agent.database import create_database_engine
    from ops_agent.main import app
    from scripts.generate_sft_dataset import proposal_snapshot, refund_snapshot, reset_and_seed
    from scripts.generate_sft_v11 import scenarios
    from scripts.run_scenarios import isolated_database
    if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        raise RuntimeError("Disposable database opt-in is required for evaluation.")
    if not torch.cuda.is_available() or (loaded_model is None and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Evaluation requires a CUDA GPU with bfloat16 support.")
    cases = {case.id: case for case in scenarios(20261121) if case.split == "validation"}
    if loaded_model is None:
        model = AutoModelForCausalLM.from_pretrained(CONFIG["checkpoint"], revision=CONFIG["revision"],
                                                     torch_dtype=torch.bfloat16, token=False).to("cuda")
        model = PeftModel.from_pretrained(model, output / "adapter").eval()
    else:
        model = loaded_model.eval()
    runner = TransformerModel(model, tokenizer)
    results = {}
    engine = create_database_engine()
    try:
        with isolated_database(engine) as isolated:
            app.state.session_factory = sessionmaker(isolated, expire_on_commit=False)
            try:
                with TestClient(app) as client:
                    for mode in ("base", "tuned"):
                        results[mode] = []
                        with model.disable_adapter() if mode == "base" else contextlib.nullcontext():
                            for row in records:
                                case = cases[row["scenario_id"]]
                                if row["messages"][1]["content"] != case.request:
                                    raise ValueError("Validation request does not match its fixture.")
                                target, before_refunds = reset_and_seed(isolated, case)
                                before = proposal_snapshot(isolated)
                                result = run_agent(runner, client, case.request)
                                after = proposal_snapshot(isolated)
                                after_refunds = refund_snapshot(isolated)
                                results[mode].append(grade_result(case, target, before, after,
                                                                  before_refunds, after_refunds, result))
            finally:
                del app.state.session_factory
    finally:
        engine.dispose()
    adapter_file = output / "adapter" / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError("Pilot adapter weights were not saved in the expected format.")
    report = {
        "configuration": CONFIG,
        "adapter_sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "peft": peft.__version__, "cuda_device": torch.cuda.get_device_name(),
                    "enable_thinking": False,
                    "instructions_sha256": hashlib.sha256(INSTRUCTIONS.encode("utf-8")).hexdigest(),
                    "tools_sha256": hashlib.sha256(json.dumps(TOOLS_CHAT, sort_keys=True).encode("utf-8")).hexdigest()},
        "results": results,
    }
    if additional_settings is not None:
        report["qlora_settings"] = additional_settings
    for mode, rows in results.items():
        report[mode] = {
            "correct_pending_proposals": sum(r["passed"] and r["positive"] for r in rows),
            "correct_no_proposal": sum(r["passed"] and not r["positive"] for r in rows),
            "false_proposals": sum(r["false_proposal"] for r in rows),
            "invalid_decisions": sum(r["invalid_decision"] for r in rows),
            "truncations": sum(r["truncation"] for r in rows),
            "new_refunds": sum(r["new_refunds"] for r in rows),
        }
    report["fail_to_pass"] = [a["scenario_id"] for a, b in zip(results["base"], results["tuned"]) if not a["passed"] and b["passed"]]
    report["pass_to_fail"] = [a["scenario_id"] for a, b in zip(results["base"], results["tuned"]) if a["passed"] and not b["passed"]]
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("base", "tuned", "fail_to_pass", "pass_to_fail")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "all"))
    parser.add_argument("--output", type=Path, default=ROOT / "pilot-runs" / "qwen3-4b-v1.1")
    args = parser.parse_args()
    train_records, validation = load_split("train"), load_split("validation")
    if args.mode == "all" and args.output.exists():
        raise FileExistsError("Pilot output already exists; choose a fresh directory.")
    tok = tokenizer()
    if args.mode == "inspect":
        inspect(tok, train_records)
    else:
        if os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
            raise RuntimeError("Disposable database opt-in is required before the pilot.")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
        train(tok, train_records, args.output)
        evaluate(tok, validation, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError, FileExistsError)):
            message = str(exc)
            if not message.startswith(("Trace ", "Pilot ", "Output ", "Training ", "Final ", "Assistant ",
                                       "Chat ", "Token ", "Tool ", "Disposable ", "Evaluation ",
                                       "train dataset ", "validation dataset ", "Validation ")):
                message = "Check configuration and local inputs."
            print(f"Pilot stopped: {message}", file=sys.stderr)
        else:
            print(f"Pilot stopped ({type(exc).__name__}); check the environment and local inputs.", file=sys.stderr)
        raise SystemExit(1)
