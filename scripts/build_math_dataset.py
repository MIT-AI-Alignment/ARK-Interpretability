"""Build data/items/math.jsonl as 100 L2 + 100 L3 + 60 L4 + 40 AIME (200 train / 100 eval).

Inputs:
  - data/items/math.jsonl as it stands (300 items, 100 each at MATH levels 2/3/4).
  - AIME 1983-2024 dataset (di-zhang-fdu/AIME_1983_2024) fetched from HuggingFace.

Output: data/items/math.jsonl rewritten with the new composition, deterministic given seeds.

Reproducibility: SHA256 of the result is recorded in data/items/HASHES.txt.
"""

import csv
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MATH_PATH = REPO / "data" / "items" / "math.jsonl"
HASHES_PATH = REPO / "data" / "items" / "HASHES.txt"
MECH_PATH = REPO / "data" / "items" / "mechanism_qa.jsonl"

AIME_CSV_URL = (
    "https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/"
    "resolve/main/AIME_Dataset_1983_2024.csv"
)

L4_KEEP_SEED = 0       # which 60 of 100 L4 items survive
AIME_SAMPLE_SEED = 1   # which 40 of ~63 AIME candidates are picked


def fetch_aime_csv() -> list[dict]:
    with urllib.request.urlopen(AIME_CSV_URL) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def filter_aime_candidates(rows: list[dict]) -> list[dict]:
    """Keep only 2022-2024 problems that are answerable from text alone with a clean integer answer."""
    out = []
    for r in rows:
        if int(r["Year"]) not in (2022, 2023, 2024):
            continue
        # Drop problems whose statement requires an Asymptote figure or LaTeX figure.
        if "[asy]" in r["Question"] or r"\begin{figure}" in r["Question"]:
            continue
        # Drop problems with non-integer or ambiguous answers.
        if not re.fullmatch(r"\d+", r["Answer"].strip()):
            continue
        out.append(r)
    return out


def sample_aime(candidates: list[dict]) -> list[dict]:
    """Stratified sample of 40 items: 14 from 2022, 14 from 2023, 12 from 2024 (all of 2024)."""
    rng = random.Random(AIME_SAMPLE_SEED)
    target = {2022: 14, 2023: 14, 2024: 12}
    chosen = []
    for year, n in target.items():
        pool = sorted(
            (r for r in candidates if int(r["Year"]) == year),
            key=lambda r: (r["Part"], int(r["Problem Number"])),
        )
        if len(pool) < n:
            raise RuntimeError(f"Only {len(pool)} candidates for year {year}, need {n}")
        chosen.extend(rng.sample(pool, n))
    chosen.sort(key=lambda r: (int(r["Year"]), r["Part"], int(r["Problem Number"])))
    assert len(chosen) == 40
    return chosen


def assign_aime_splits(items: list[dict]) -> list[dict]:
    """Within each year, deterministic split: ~2:1 train:eval."""
    splits = {2022: ("train", 9, 5), 2023: ("train", 9, 5), 2024: ("train", 8, 4)}
    out = []
    for year, (_, n_train, n_eval) in splits.items():
        year_items = [r for r in items if int(r["Year"]) == year]
        assert len(year_items) == n_train + n_eval, year
        for i, r in enumerate(year_items):
            r = dict(r)
            r["_split"] = "train" if i < n_train else "eval"
            out.append(r)
    return out


def aime_row_to_item(r: dict, item_id: str) -> dict:
    return {
        "item_id": item_id,
        "level": "AIME",
        "subject": f"AIME {r['Year']} {r['Part']}",
        "problem": r["Question"],
        "answer": r["Answer"].strip(),
        "split": r["_split"],
        "source": "AIME",
        "aime_year": int(r["Year"]),
        "aime_part": r["Part"],
        "aime_problem_number": int(r["Problem Number"]),
    }


def reduce_l4(existing_l4: list[dict]) -> list[dict]:
    """Keep 60 of 100 L4 items: 40 train (from 66) + 20 eval (from 34). Deterministic."""
    rng = random.Random(L4_KEEP_SEED)
    train = sorted([it for it in existing_l4 if it["split"] == "train"], key=lambda x: x["item_id"])
    eval_ = sorted([it for it in existing_l4 if it["split"] == "eval"], key=lambda x: x["item_id"])
    assert len(train) == 66 and len(eval_) == 34, (len(train), len(eval_))
    keep_train = sorted(rng.sample(train, 40), key=lambda x: x["item_id"])
    keep_eval = sorted(rng.sample(eval_, 20), key=lambda x: x["item_id"])
    return keep_train + keep_eval


def add_source_field(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        if "source" not in it:
            it = {**it, "source": "MATH"}
        out.append(it)
    return out


def main() -> None:
    with MATH_PATH.open() as f:
        existing = [json.loads(line) for line in f]
    assert len(existing) == 300

    by_level = {2: [], 3: [], 4: []}
    for it in existing:
        by_level[it["level"]].append(it)
    for lvl, items in by_level.items():
        assert len(items) == 100, (lvl, len(items))

    keep_l2 = sorted(by_level[2], key=lambda x: x["item_id"])
    keep_l3 = sorted(by_level[3], key=lambda x: x["item_id"])
    keep_l4 = sorted(reduce_l4(by_level[4]), key=lambda x: x["item_id"])

    aime_rows = fetch_aime_csv()
    candidates = filter_aime_candidates(aime_rows)
    print(f"AIME candidates after filtering: {len(candidates)}", file=sys.stderr)
    sampled = sample_aime(candidates)
    sampled = assign_aime_splits(sampled)
    aime_items = [
        aime_row_to_item(r, f"math_{301 + i:04d}") for i, r in enumerate(sampled)
    ]

    math_items = add_source_field(keep_l2 + keep_l3 + keep_l4)
    all_items = math_items + aime_items

    train_total = sum(1 for it in all_items if it["split"] == "train")
    eval_total = sum(1 for it in all_items if it["split"] == "eval")
    assert train_total == 200, train_total
    assert eval_total == 100, eval_total

    with MATH_PATH.open("w") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    breakdown = Counter((str(it["level"]), it["split"]) for it in all_items)
    print("Final composition:", file=sys.stderr)
    for k in sorted(breakdown):
        print(f"  level={k[0]:>4} split={k[1]:>5}  n={breakdown[k]}", file=sys.stderr)

    math_sha = hashlib.sha256(MATH_PATH.read_bytes()).hexdigest()
    mech_sha = hashlib.sha256(MECH_PATH.read_bytes()).hexdigest()
    HASHES_PATH.write_text(
        f"{math_sha}  math.jsonl\n{mech_sha}  mechanism_qa.jsonl\n"
    )
    print(f"\nSHA256 math.jsonl       {math_sha}", file=sys.stderr)
    print(f"SHA256 mechanism_qa.jsonl {mech_sha}", file=sys.stderr)


if __name__ == "__main__":
    main()
