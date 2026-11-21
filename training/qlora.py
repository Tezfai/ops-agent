"""Memory-gated, frozen-data QLoRA pilot for a local NVIDIA GPU."""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

from training.pilot import CONFIG, ROOT, encode_turns, evaluate, inspect, load_split, pad_examples, tokenizer

QLORA = json.loads((ROOT / "training" / "qlora_config.json").read_text(encoding="utf-8"))


def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def create_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback.")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=QLORA["quantization"],
                               bnb_4bit_use_double_quant=QLORA["double_quant"],
                               bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["checkpoint"], revision=CONFIG["revision"], token=False,
        quantization_config=quant, torch_dtype=torch.float16, device_map={"": 0})
    if not any(isinstance(m, bnb.nn.Linear4bit) for m in base.modules()):
        raise RuntimeError("No 4-bit layers were loaded.")
    if any(p.device.type != "cuda" for p in base.parameters()):
        raise RuntimeError("Model parameters are not entirely on CUDA.")
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LoraConfig(
        r=CONFIG["lora_rank"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_targets"], task_type=TaskType.CAUSAL_LM))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Trainable parameters are not exclusively LoRA parameters.")
    return model


def batch(examples, tok):
    return {key: torch.tensor(value, device="cuda")
            for key, value in pad_examples(examples, tok.pad_token_id).items()}


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def preflight(tok, examples):
    set_seed(CONFIG["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = create_model()
    model.train()
    longest = max(examples, key=lambda x: len(x["input_ids"]))
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]
    optim = bnb.optim.PagedAdamW8bit(params, lr=CONFIG["learning_rate"])
    result = model(**batch([longest], tok))
    loss = result.loss
    if not torch.isfinite(loss):
        raise RuntimeError("Preflight training loss is not finite.")
    loss.backward()
    optim.step()
    torch.cuda.synchronize()
    if not any(not torch.equal(p, old) for p, old in zip(params, initial)):
        raise RuntimeError("Preflight optimizer step did not change a LoRA parameter.")
    train_stats = {"tokens": len(longest["input_ids"]), "loss": float(loss.item()),
                   "lora_changed": True, **memory()}
    print("TRAINING_PREFLIGHT " + json.dumps(train_stats), flush=True)
    del result, loss, params, initial, optim
    del model
    clear()

    set_seed(CONFIG["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = create_model().eval()
    # The longest training prefix contains real synthetic tool calls and responses;
    # force the full cap rather than accepting an early EOS as a memory proof.
    prefix = longest["masked_prefix"]
    inputs = tok(prefix, return_tensors="pt", add_special_tokens=False).to("cuda")
    model.config.use_cache = True
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False,
                                   min_new_tokens=CONFIG["max_new_tokens"],
                                   max_new_tokens=CONFIG["max_new_tokens"],
                                   pad_token_id=tok.pad_token_id)
    length = generated.shape[1] - inputs["input_ids"].shape[1]
    infer_stats = {"context_tokens": inputs["input_ids"].shape[1], "generated_tokens": length, **memory()}
    print("INFERENCE_PREFLIGHT " + json.dumps(infer_stats), flush=True)
    if length != CONFIG["max_new_tokens"]:
        raise RuntimeError("Preflight did not exercise the full completion cap.")
    del generated, inputs
    del model
    clear()
    return {"training": train_stats, "inference": infer_stats}


def train(tok, examples, output):
    set_seed(CONFIG["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = create_model()

    def collate(items):
        return {key: torch.tensor(value) for key, value in pad_examples(items, tok.pad_token_id).items()}

    args = TrainingArguments(output_dir=str(output / "trainer"), per_device_train_batch_size=1,
                             gradient_accumulation_steps=QLORA["gradient_accumulation_steps"],
                             num_train_epochs=CONFIG["epochs"], learning_rate=CONFIG["learning_rate"],
                             fp16=True, logging_steps=1, save_strategy="no", report_to=[],
                             seed=CONFIG["seed"], data_seed=CONFIG["seed"], remove_unused_columns=False,
                             optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate)
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Training loss is not finite.")
    model.save_pretrained(output / "adapter")
    tok.save_pretrained(output / "tokenizer")
    stats = {"training_loss": result.training_loss, "global_steps": result.global_step, **memory()}
    print("TRAINING_COMPLETE " + json.dumps(stats), flush=True)
    del trainer
    model.config.use_cache = True
    return model, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "pilot-runs" / "qwen3-4b-qlora-v1.1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output directory already exists.")
    train_records = load_split("train")
    validation = load_split("validation")
    if not args.preflight_only and os.getenv("OPS_AGENT_TEST_USE_DEV_DATABASE") != "1":
        raise RuntimeError("Disposable database opt-in is required before training and evaluation.")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {"base": CONFIG, "qlora": QLORA, "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "bitsandbytes": bnb.__version__,
                "transformers": transformers.__version__, "peft": importlib.metadata.version("peft"),
                "accelerate": importlib.metadata.version("accelerate"),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    stage = "tokenization"
    try:
        tok = tokenizer()
        examples = inspect(tok, train_records)
        stage = "real-checkpoint preflight"
        stats = preflight(tok, examples)
        (args.output / "preflight.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        if args.preflight_only:
            return
        stage = "training"
        model, training_stats = train(tok, examples, args.output)
        (args.output / "training.json").write_text(json.dumps(training_stats, indent=2) + "\n", encoding="utf-8")
        stage = "matched evaluation"
        evaluate(tok, validation, args.output, loaded_model=model, additional_settings=settings)
    except Exception as exc:
        info = {"stage": stage, "exception_type": type(exc).__name__}
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            info["error"] = "CUDA out of memory"
        elif isinstance(exc, RuntimeError) and str(exc).startswith(("Preflight ", "CUDA ", "No 4-bit ", "Model parameters ", "Trainable ", "Training loss ")):
            info["error"] = str(exc)
        else:
            info["error"] = "Operation failed; details suppressed to avoid exposing credentials."
        if torch.cuda.is_available():
            info["memory"] = memory()
        (args.output / "failure.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print("PILOT_FAILURE " + json.dumps(info), file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
