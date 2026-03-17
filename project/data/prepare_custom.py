"""
Build a custom conflict dataset (~100 items) from TriviaQA.
Only retains items where the subject+answer are high-frequency factual
knowledge that most LLMs are likely to have stored parametrically.

Output format (same as prepare_counterfact.py):
{
  "id": "custom_<i>",
  "subject": str,
  "label_parametric": str,
  "label_incontext": str,
  "prompt_no_context": str,
  "prompt_with_context": str,
  "source": "custom_triviaqа"
}

Usage:
    python data/prepare_custom.py --n 100 --out data/output/custom_conflict.jsonl
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm

# Distractor answers: plausible-sounding wrong answers for each domain
_DOMAIN_DISTRACTORS = {
    "country": ["Germany", "France", "Italy", "Spain", "Japan", "Brazil", "Canada", "Australia"],
    "city": ["Berlin", "Madrid", "Sydney", "Toronto", "Tokyo", "Cairo", "Mumbai", "Seoul"],
    "person": ["Albert Einstein", "Isaac Newton", "Charles Darwin", "Marie Curie", "Nikola Tesla"],
    "year": ["1945", "1969", "1989", "2001", "1776", "1492"],
    "generic": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
}


def pick_distractor(true_obj: str, subject: str) -> str:
    """Pick a plausible but wrong answer distinct from true_obj."""
    for domain, pool in _DOMAIN_DISTRACTORS.items():
        for item in pool:
            if item.lower() not in true_obj.lower() and item.lower() not in subject.lower():
                return item
    return "unknown_entity_42"


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


def load_triviaqа_conflicts(n: int, seed: int = 42) -> list[dict]:
    print("[prepare_custom] Loading TriviaQA (rc.wikipedia) ...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train")
    print(f"  Dataset size: {len(ds)}")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n * 5, len(ds)))

    records = []
    for i in tqdm(indices, desc="Building custom conflicts"):
        row = ds[i]
        question = row.get("question", "")
        answer = row.get("answer", {})
        true_obj = answer.get("value", "") if isinstance(answer, dict) else str(answer)
        aliases = answer.get("aliases", []) if isinstance(answer, dict) else []

        if not question or not true_obj:
            continue

        # Try to extract subject from question (rough heuristic: first proper noun phrase)
        words = question.rstrip("?").split()
        subject = " ".join(w for w in words[-4:] if w[0].isupper()) or words[-1]
        if not subject:
            continue

        false_obj = pick_distractor(true_obj, subject)
        if not false_obj:
            continue

        conflict = make_conflict_prompt(subject, true_obj, false_obj)
        record = {
            "id": f"custom_{len(records)}",
            "subject": subject,
            "question": question,
            "source": "custom_triviaqа",
            "aliases": aliases[:5],
            **conflict,
        }
        records.append(record)
        if len(records) >= n:
            break

    print(f"  Collected {len(records)} records (requested {n})")
    return records


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare_custom] Saved {len(records)} records → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="data/output/custom_conflict.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_triviaqа_conflicts(args.n, seed=args.seed)
    save_jsonl(records, args.out)
