"""
Load CounterFact dataset (NeelNanda/counterfact-tracing) and produce
short_conflict.jsonl for Stage 1 experiments.

Each output record:
{
  "id": str,
  "subject": str,
  "label_parametric": str,   # model's true stored answer
  "label_incontext": str,    # counterfactual answer injected in context
  "prompt_no_context": str,
  "prompt_with_context": str,
  "source": "counterfact"
}

Usage:
    python data/prepare_counterfact.py --n 200 --out data/output/short_conflict.jsonl
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def make_conflict_prompt(subject: str, true_obj: str, false_obj: str) -> dict:
    context = (
        f"According to recent records, {subject} is associated with {false_obj}."
    )
    question = f"Q: What is {subject} associated with?\nA:"
    return {
        "prompt_with_context": f"Context: {context}\n\n{question}",
        "prompt_no_context": question,
        "label_parametric": true_obj,
        "label_incontext": false_obj,
    }


def load_counterfact(n: int, seed: int = 42) -> list[dict]:
    print("[prepare_counterfact] Loading NeelNanda/counterfact-tracing ...")
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    print(f"  Dataset size: {len(ds)}")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n * 3, len(ds)))  # oversample → filter

    records = []
    for i in tqdm(indices, desc="Building conflict prompts"):
        row = ds[i]
        try:
            # Actual dataset structure has these fields directly
            subject = row["subject"].strip()
            true_obj = row["target_true"].strip()
            false_obj = row["target_false"].strip()
        except (KeyError, TypeError):
            continue

        if not subject or not true_obj or not false_obj:
            continue
        if true_obj.lower() == false_obj.lower():
            continue

        conflict = make_conflict_prompt(subject, true_obj, false_obj)
        record = {
            "id": f"cf_{i}",
            "subject": subject,
            "source": "counterfact",
            **conflict,
        }
        records.append(record)
        if len(records) >= n:
            break

    print(f"  Collected {len(records)} usable records (requested {n})")
    return records


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare_counterfact] Saved {len(records)} records → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="data/output/short_conflict.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_counterfact(args.n, seed=args.seed)
    save_jsonl(records, args.out)
