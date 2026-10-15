"""S1: LoRA supervised fine-tuning of a small instruct model on the TSN
scheduling prompt/completion pairs built by build_sft_dataset.py.

Trains the model to predict `completion_text` (admission decisions +
offsets/queues) given `prompt_text` (topology + requested flows), using the
model's own chat template and masking the loss to the assistant turn only
(standard SFT — the model isn't scored on reproducing the prompt).

Needs a GPU. Run via scripts/cluster/submit_s1_train.sbatch on disi, not
locally (this repo's dev machine has no CUDA GPU) — see
slm_future/s1_lora/requirements-gpu.txt for the extra deps this needs on
top of the foundation-phase requirements.txt.

Usage:
    python -m slm_future.s1_lora.train_lora \
        --sft-data-dir slm_future/s1_lora/sft_data \
        --output-dir slm_future/s1_lora/checkpoints/run1 \
        --model-name Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_example(tokenizer, prompt_text: str, completion_text: str, max_len: int) -> dict:
    """Tokenize one (prompt, completion) pair via the model's chat template,
    masking every token before the assistant turn so loss is computed only
    on the completion (the scheduling decision), not the prompt."""
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text},
         {"role": "assistant", "content": completion_text}],
        tokenize=False, add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_len)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prefix_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prefix_len + input_ids[prefix_len:]
    labels = labels[:len(input_ids)]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT for S1 (TSN scheduling SLM)")
    ap.add_argument("--sft-data-dir", default="slm_future/s1_lora/sft_data")
    ap.add_argument("--output-dir", default="slm_future/s1_lora/checkpoints/run1")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {"train": f"{args.sft_data_dir}/train.jsonl"}
    val_path = Path(args.sft_data_dir) / "val.jsonl"
    has_val = val_path.exists() and val_path.stat().st_size > 0
    if has_val:
        data_files["validation"] = str(val_path)
    raw = load_dataset("json", data_files=data_files)

    def tokenize_fn(example):
        return build_example(tokenizer, example["prompt_text"], example["completion_text"], args.max_seq_len)

    tokenized = raw.map(tokenize_fn, remove_columns=raw["train"].column_names)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_val else None,
        data_collator=PadCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    with open(out_dir / "training_log.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model_name": args.model_name,
            "lora_config": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                             "target_modules": LORA_TARGET_MODULES},
            "log_history": trainer.state.log_history,
            "n_train_examples": len(tokenized["train"]),
            "n_val_examples": len(tokenized["validation"]) if has_val else 0,
        }, fh, indent=2)

    print(f"Done. Adapter saved to {out_dir / 'adapter'}, training log at {out_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
