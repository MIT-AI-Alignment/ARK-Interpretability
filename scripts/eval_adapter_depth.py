#!/usr/bin/env python
"""Run the D1-D4 depth pipeline on the LoRA-adapter Qwen, mirroring the baseline runs.

Reads the same first-10 items from mechanism_qa.jsonl and math.jsonl that the
baseline runs used. Saves two new result files prefixed `adapter_*` so they
can be compared side-by-side with the baseline `ioed_*` files.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

MECH_PATH   = Path("/workspace/ARK-Interpretability/notebooks/mechanism_qa.jsonl")
MATH_PATH   = Path("/workspace/ARK-Interpretability/data/items/math.jsonl")
RESULTS_DIR = Path("/workspace/ARK-Interpretability/results")
ADAPTER     = "/workspace/ARK-Interpretability/adapters/qwen25_ioed_calibration"
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading base model + adapter...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
model      = PeftModel.from_pretrained(base_model, ADAPTER)
print(f"Model on: {model.device}; adapter: {ADAPTER}")


def chat(messages, max_new_tokens=256, temperature=None, do_sample=True):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
    if do_sample and temperature is not None:
        kwargs["temperature"] = temperature
    outputs = model.generate(**inputs, **kwargs)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def parse_confidence_block(text):
    text = text.strip()
    for pattern in (
        r"CONFIDENCE\s*:\s*(\d{1,3})\s*$",
        r"CONFIDENCE\s*:\s*(\d{1,3})",
    ):
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return v, (text[:m.start()] + text[m.end():]).strip()
    for pattern in (
        r"confiden(?:ce|t)\s+(?:level|score|rating|value)\s*:?\s*\**\s*(\d{1,3})\s*%?",
        r"confiden(?:ce|t)\s*:\s*\**\s*(\d{1,3})\s*%?",
        r"(\d{1,3})\s*(?:%|percent)\s*confiden(?:ce|t)",
        r"confiden(?:ce|t)[\s\S]{0,200}?(\d{1,3})\s*%",
    ):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return v, text
    return None, text


MECH_DEPTH_PROMPTS = {
    1: """Question: {question}

Provide your final answer in one sentence summarizing the core mechanism. Do not explain the details.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your one-sentence answer correctly captures the mechanism.""",
    2: """Question: {question}

Provide your final answer as a step-by-step mechanism. Walk through what happens at each step.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your mechanism is correct.""",
    3: """Question: {question}

Provide your final answer as a step-by-step mechanism, AND for each step explain why that step is necessary — what underlying principle (physical, geometric, or logical) makes the step work, and what would go wrong if that principle were absent.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your mechanism and justifications are correct.""",
    4: """Question: {question}

Provide your final answer in two parts:
1. State the core mechanism in 1-2 sentences.
2. Describe 3-5 distinct ways this mechanism can fail or break down. For each failure mode, identify which underlying principle of the mechanism is being violated and explain why that violation produces the observed failure.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your failure-mode analysis correctly identifies the underlying mechanistic causes.""",
}

MATH_DEPTH_PROMPTS = {
    1: """Problem: {problem}

Solve this problem. Provide ONLY your final answer in simplest form. Do not show your work or explain your reasoning.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your answer is correct.""",
    2: """Problem: {problem}

Solve this problem step by step. Show all your work, but do not justify why each step is valid.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your solution is correct.""",
    3: """Problem: {problem}

Solve this problem step by step. For each step, explain why it is necessary or valid — what mathematical principle, theorem, identity, or rule justifies it, and what would go wrong if you skipped or changed that step.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your solution and justifications are correct.""",
    4: """Problem: {problem}

Provide your answer in two parts:
1. Solve the problem step by step.
2. Identify 3-5 distinct mistakes a student could plausibly make on this problem. For each mistake, explain which step or principle is being violated and what wrong answer it would produce.

End your response with exactly:
CONFIDENCE: <integer 0-100>

where the integer reflects how confident you are that your solution AND your failure-mode analysis are correct.""",
}

DEPTH_MAX_TOKENS_MECH = {1: 120, 2: 400, 3: 600, 4: 700}
DEPTH_MAX_TOKENS_MATH = {1:  80, 2: 500, 3: 800, 4: 900}


def run_mech_pipeline(item):
    out = {"item_id": item["item_id"], "topic_bucket": item["topic_bucket"],
           "split": item["split"], "question": item["question"]}
    for depth in (1, 2, 3, 4):
        prompt = MECH_DEPTH_PROMPTS[depth].format(question=item["question"])
        messages = [
            {"role": "system", "content": "You are a helpful and honest AI assistant."},
            {"role": "user", "content": prompt},
        ]
        raw = chat(messages, max_new_tokens=DEPTH_MAX_TOKENS_MECH[depth], temperature=0.7, do_sample=True)
        c, a = parse_confidence_block(raw)
        out[f"d{depth}_answer"]     = a
        out[f"d{depth}_raw"]        = raw
        out[f"d{depth}_confidence"] = c
    return out


def run_math_pipeline(item):
    out = {"item_id": item["item_id"], "subject": item["subject"], "level": item["level"],
           "split": item["split"], "problem": item["problem"], "answer": item["answer"]}
    for depth in (1, 2, 3, 4):
        prompt = MATH_DEPTH_PROMPTS[depth].format(problem=item["problem"])
        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Be precise and rigorous."},
            {"role": "user", "content": prompt},
        ]
        raw = chat(messages, max_new_tokens=DEPTH_MAX_TOKENS_MATH[depth], temperature=0.7, do_sample=True)
        c, a = parse_confidence_block(raw)
        out[f"d{depth}_solution"]   = a
        out[f"d{depth}_raw"]        = raw
        out[f"d{depth}_confidence"] = c
    return out


# === Run mechanism ===
mech_items = [json.loads(l) for l in MECH_PATH.open()][:10]
print(f"\nRunning mechanism pipeline on {len(mech_items)} items (adapter)...")
mech_results = []
for it in tqdm(mech_items):
    mech_results.append(run_mech_pipeline(it))

# === Run math ===
math_items = [json.loads(l) for l in MATH_PATH.open()][:10]
print(f"\nRunning math pipeline on {len(math_items)} items (adapter)...")
math_results = []
for it in tqdm(math_items):
    math_results.append(run_math_pipeline(it))

# === Save ===
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
mech_out = RESULTS_DIR / f"adapter_mechanism_run_{ts}.json"
math_out = RESULTS_DIR / f"adapter_math_run_{ts}.json"

def payload(results, dataset_path, pipeline):
    return {
        "run_metadata": {
            "timestamp":     ts,
            "model_id":      MODEL_ID,
            "adapter_path":  ADAPTER,
            "dataset_path":  str(dataset_path),
            "num_items":     len(results),
            "sample_filter": "first 10 items",
            "pipeline":      pipeline,
        },
        "items": results,
    }

mech_out.write_text(json.dumps(payload(mech_results, MECH_PATH, "depth_1_to_4_mech"), indent=2))
math_out.write_text(json.dumps(payload(math_results, MATH_PATH, "depth_1_to_4_math"), indent=2))
print(f"\nSaved:\n  {mech_out.name}\n  {math_out.name}")

# === Summary ===
def summarize(results, name):
    df = pd.DataFrame([{f"d{d}": r.get(f"d{d}_confidence") for d in (1,2,3,4)} for r in results])
    print(f"\n=== {name} confidence (n={len(results)}, valid per depth = {df.notna().sum().to_dict()}) ===")
    print(df.mean().round(2).to_string())

summarize(mech_results, "MECHANISM (adapter)")
summarize(math_results, "MATH (adapter)")
print("\nDone.")
