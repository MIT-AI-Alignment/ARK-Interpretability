#!/usr/bin/env python
"""LoRA fine-tune Qwen2.5-1.5B-Instruct on the IOED calibration dataset.

Trains the model to use calibrated verbalized confidence: high on facts it knows,
low / IOED-acknowledging on deep mechanism questions, mid on appropriate queries.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH  = "/workspace/ARK-Interpretability/data/items/qwen25_ioed_math_mech_calibration.jsonl"
OUTPUT_DIR = "/workspace/ARK-Interpretability/adapters/qwen25_ioed_calibration"
SEED       = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Use 100 examples + 1 epoch for a smoke test")
    p.add_argument("--data",       default=DATA_PATH)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--epochs",     type=int,   default=2)
    p.add_argument("--batch",      type=int,   default=4)
    p.add_argument("--accum",      type=int,   default=2)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--lora_r",     type=int,   default=16)
    p.add_argument("--max_length", type=int,   default=1024)
    args = p.parse_args()

    print(f"Loading dataset from {args.data}")
    items = [json.loads(l) for l in open(args.data)]
    print(f"  {len(items)} examples loaded")

    random.seed(SEED)
    random.shuffle(items)
    if args.smoke:
        items = items[:100]
        args.epochs = 1
        print(f"  SMOKE MODE: using {len(items)} examples, 1 epoch")
    split = int(0.9 * len(items))

    def to_messages(ex):
        return {"messages": [
            {"role": "user",      "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]},
        ]}

    train_ds = Dataset.from_list([to_messages(ex) for ex in items[:split]])
    eval_ds  = Dataset.from_list([to_messages(ex) for ex in items[split:]])
    print(f"  train: {len(train_ds)}, eval: {len(eval_ds)}")

    print(f"Loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200 if not args.smoke else 50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_length=args.max_length,
        seed=SEED,
        report_to="none",
        assistant_only_loss=True,    # only the model's responses contribute to loss
        dataset_num_proc=4,
    )

    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Show trainable param count to confirm LoRA is wired correctly.
    trainer.model.print_trainable_parameters()

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    Path(args.output_dir, "training_meta.json").write_text(json.dumps({
        "model_id":        MODEL_ID,
        "data_path":       args.data,
        "num_train":       len(train_ds),
        "num_eval":        len(eval_ds),
        "epochs":          args.epochs,
        "lr":              args.lr,
        "lora_r":          args.lora_r,
        "lora_alpha":      args.lora_r * 2,
        "batch":           args.batch,
        "grad_accum":      args.accum,
        "max_length":      args.max_length,
        "assistant_only_loss": True,
    }, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
