"""
Process ConflictBank (NeurIPS 2024) for Stage 1 experiments.
https://github.com/zhaochen0110/conflictbank

ConflictBank has three conflict types:
  - misinformation
  - temporal_discrepancy
  - semantic_divergence

Expected local structure after cloning the repo:
  conflictbank/
    data/
      misinformation/       (claim-evidence pairs)
      temporal_discrepancy/
      semantic_divergence/
      qa/                   (QA pairs, 553K total)

Usage:
    python data/prepare_conflictbank.py \
        --bank_dir /path/to/conflictbank \
        --conflict_type misinformation \
        --n 200 \
        --out data/output/conflictbank_conflict.jsonl
"""

import argparse
import json
import os
import random

from tqdm import tqdm

CONFLICT_TYPES = ["misinformation", "temporal_discrepancy", "semantic_divergence"]


def load_conflictbank_qa(bank_dir: str, conflict_type: str, n: int, seed: int = 42) -> list[dict]:
    """
    Load QA pairs from ConflictBank local directory.
    Falls back to a synthetic demo if the directory doesn't exist.
    """
    qa_dir = os.path.join(bank_dir, "data", "qa")
    if not os.path.exists(qa_dir):
        print(
            f"[prepare_conflictbank] WARNING: {qa_dir} not found. "
            "Generating synthetic demo records instead."
        )
        return _make_synthetic_demo(conflict_type, n, seed)

    # ConflictBank QA files are typically named by conflict type
    candidates = []
    for fname in os.listdir(qa_dir):
        if conflict_type.lower() in fname.lower() and fname.endswith(".json"):
            candidates.append(os.path.join(qa_dir, fname))
    if not candidates:
        for fname in os.listdir(qa_dir):
            if fname.endswith(".json") or fname.endswith(".jsonl"):
                candidates.append(os.path.join(qa_dir, fname))

    records = []
    random.seed(seed)
    for fpath in sorted(candidates):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                subject = obj.get("subject") or obj.get("entity") or obj.get("topic", "")
                true_obj = obj.get("parametric_answer") or obj.get("true_answer") or obj.get("answer_true", "")
                false_obj = obj.get("conflict_answer") or obj.get("new_answer") or obj.get("answer_false", "")
                question = obj.get("question") or obj.get("query", "")
                if not (subject and true_obj and false_obj):
                    continue
                context = f"According to recent records, {false_obj} is the answer to: {question}"
                records.append({
                    "id": f"cb_{conflict_type}_{len(records)}",
                    "subject": subject,
                    "label_parametric": true_obj,
                    "label_incontext": false_obj,
                    "prompt_no_context": f"Q: {question}\nA:",
                    "prompt_with_context": f"Context: {context}\n\nQ: {question}\nA:",
                    "source": f"conflictbank_{conflict_type}",
                    "conflict_type": conflict_type,
                })
        if len(records) >= n:
            break

    random.shuffle(records)
    records = records[:n]
    print(f"[prepare_conflictbank] Loaded {len(records)} records from ConflictBank ({conflict_type})")
    return records


def _make_synthetic_demo(conflict_type: str, n: int, seed: int) -> list[dict]:
    """
    Synthetic stand-in records for testing without the actual ConflictBank data.
    Based on common factual knowledge that most LLMs know.
    """
    TEMPLATES = [
        ("Paris", "France", "Germany", "What country is Paris the capital of?"),
        ("Mount Everest", "Nepal/Tibet", "Canada", "Where is Mount Everest located?"),
        ("Albert Einstein", "Theory of Relativity", "Theory of Evolution", "What theory is Albert Einstein known for?"),
        ("The Eiffel Tower", "Paris", "London", "In which city is the Eiffel Tower located?"),
        ("William Shakespeare", "England", "France", "What country was William Shakespeare from?"),
        ("The Amazon River", "South America", "Africa", "On which continent is the Amazon River?"),
        ("Beethoven", "Germany", "Italy", "Which country was Beethoven from?"),
        ("The Great Wall", "China", "Japan", "Which country built the Great Wall?"),
        ("Hamlet", "Shakespeare", "Dickens", "Who wrote Hamlet?"),
        ("Apollo 11", "Moon", "Mars", "Where did Apollo 11 land?"),
    ]
    random.seed(seed)
    records = []
    for i in range(n):
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        subject, true_obj, false_obj, question = tmpl
        context = f"According to recent records, {subject} is associated with {false_obj}."
        records.append({
            "id": f"cb_synthetic_{conflict_type}_{i}",
            "subject": subject,
            "label_parametric": true_obj,
            "label_incontext": false_obj,
            "prompt_no_context": f"Q: {question}\nA:",
            "prompt_with_context": f"Context: {context}\n\nQ: {question}\nA:",
            "source": f"conflictbank_{conflict_type}_synthetic",
            "conflict_type": conflict_type,
        })
    return records


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare_conflictbank] Saved {len(records)} records → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank_dir", default="", help="Path to local conflictbank repo")
    parser.add_argument("--conflict_type", choices=CONFLICT_TYPES, default="misinformation")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="data/output/conflictbank_conflict.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_conflictbank_qa(args.bank_dir, args.conflict_type, args.n, args.seed)
    save_jsonl(records, args.out)
