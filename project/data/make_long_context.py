"""
Stage 2: Generate long-context conflict prompts by embedding a conflict
sentence inside distractor text at varying lengths and positions.

Design:
  - lengths:   500, 2000, 8000, 32000 tokens
  - positions: beginning, middle, end
  - per condition: 100 cases
  → 3 positions × 4 lengths = 12 conditions, ~1200 total

Distractor text is sourced from BookCorpus-like text (wikitext-103) trimmed
to the required token count.  If the dataset is unavailable a synthetic
filler is generated instead.

Output files:
  data/output/long_conflict_500.jsonl
  data/output/long_conflict_2k.jsonl
  data/output/long_conflict_8k.jsonl
  data/output/long_conflict_32k.jsonl

Usage:
    python data/make_long_context.py \
        --conflict_data data/output/short_conflict.jsonl \
        --lengths 500 2000 8000 32000 \
        --positions beginning middle end \
        --n_per_condition 100 \
        --out_dir data/output/
"""

import argparse
import json
import os
import random
import re

from tqdm import tqdm

try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Distractor text pool
# ---------------------------------------------------------------------------

_FILLER_SENTENCES = [
    "The history of modern science began in the 17th century with a series of groundbreaking discoveries.",
    "Researchers have long debated the role of environmental factors in shaping human behavior.",
    "The development of the internet transformed communication patterns across the globe.",
    "Ancient civilizations left behind remarkable artifacts that continue to fascinate archaeologists.",
    "Climate change poses one of the most significant challenges to contemporary societies.",
    "The principles of thermodynamics govern energy transfer in physical and chemical systems.",
    "Literature from the Romantic period emphasized emotion and individualism over reason.",
    "Economic theory has evolved considerably since the classical works of the 18th century.",
    "Advances in medical imaging technology have improved diagnostic accuracy substantially.",
    "The exploration of deep-sea environments has revealed extraordinary biodiversity.",
]


def _make_filler_text(target_chars: int, seed: int = 0) -> str:
    """
    Build a plausible filler paragraph of approximately `target_chars` chars
    by repeating sentences from _FILLER_SENTENCES with slight variation.
    """
    random.seed(seed)
    sentences = list(_FILLER_SENTENCES)
    text_parts = []
    total = 0
    idx = 0
    while total < target_chars:
        s = sentences[idx % len(sentences)]
        text_parts.append(s)
        total += len(s) + 1
        idx += 1
    return " ".join(text_parts)


def _get_wikitext_pool(n_chars: int = 200_000) -> str:
    """Try to load wikitext-103; fall back to synthetic filler."""
    if not _HF_AVAILABLE:
        return _make_filler_text(n_chars)
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:5%]")
        text = " ".join(row["text"] for row in ds if row["text"].strip())
        if len(text) >= n_chars:
            return text[:n_chars]
    except Exception as e:
        print(f"[make_long_context] Wikitext not available ({e}), using synthetic filler.")
    return _make_filler_text(n_chars)


# Approximate chars per token for English text (rough heuristic: ~4)
_CHARS_PER_TOKEN = 4


def _trim_to_tokens(text: str, n_tokens: int) -> str:
    target_chars = n_tokens * _CHARS_PER_TOKEN
    return text[:target_chars]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def make_long_conflict_prompt(
    subject: str,
    true_obj: str,
    false_obj: str,
    distractor_text: str,
    conflict_position: str,
    total_tokens: int,
) -> str:
    """
    Embed conflict sentence in distractor at the requested position.
    `distractor_text` is already pre-trimmed to roughly (total_tokens - overhead).
    """
    conflict_sentence = (
        f"According to recent records, {subject} is associated with {false_obj}."
    )
    question = f"Q: What is {subject} associated with?\nA:"

    if conflict_position == "beginning":
        return (
            f"Context: {conflict_sentence}\n\n"
            f"{distractor_text}\n\n"
            f"{question}"
        )
    elif conflict_position == "middle":
        half = len(distractor_text) // 2
        return (
            f"Context: {distractor_text[:half]}\n"
            f"{conflict_sentence}\n"
            f"{distractor_text[half:]}\n\n"
            f"{question}"
        )
    elif conflict_position == "end":
        return (
            f"Context: {distractor_text}\n\n"
            f"{conflict_sentence}\n\n"
            f"{question}"
        )
    else:
        raise ValueError(f"Unknown conflict_position: {conflict_position}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LENGTH_TAG = {
    500: "500",
    2000: "2k",
    8000: "8k",
    32000: "32k",
    64000: "64k",
    128000: "128k",
    256000: "256k",
}


def process(
    conflict_data_path: str,
    lengths: list[int],
    positions: list[str],
    n_per_condition: int,
    out_dir: str,
    seed: int,
):
    # Load conflict records
    records = []
    with open(conflict_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[make_long_context] Loaded {len(records)} conflict records from {conflict_data_path}")

    # Load distractor pool once
    max_tokens = max(lengths)
    pool_chars = max_tokens * _CHARS_PER_TOKEN * 2
    print(f"[make_long_context] Building distractor text pool ({pool_chars} chars) ...")
    pool_text = _get_wikitext_pool(pool_chars)
    print(f"  Pool text length: {len(pool_text)} chars")

    os.makedirs(out_dir, exist_ok=True)

    for length in lengths:
        tag = LENGTH_TAG.get(length, str(length))
        out_path = os.path.join(out_dir, f"long_conflict_{tag}.jsonl")
        out_records = []

        for position in positions:
            random.seed(seed)
            base_records = random.choices(records, k=n_per_condition)

            for j, rec in enumerate(tqdm(
                base_records,
                desc=f"  length={length}, position={position}",
                leave=False,
            )):
                # Use a different slice of pool per example to vary filler
                offset = (j * length * _CHARS_PER_TOKEN) % max(1, len(pool_text) - length * _CHARS_PER_TOKEN)
                distractor = pool_text[offset: offset + length * _CHARS_PER_TOKEN]
                distractor = _trim_to_tokens(distractor, length - 50)  # leave room for conflict + question

                prompt = make_long_conflict_prompt(
                    subject=rec["subject"],
                    true_obj=rec["label_parametric"],
                    false_obj=rec["label_incontext"],
                    distractor_text=distractor,
                    conflict_position=position,
                    total_tokens=length,
                )

                out_records.append({
                    "id": f"long_{tag}_{position}_{rec['id']}_{j}",
                    "source_id": rec["id"],
                    "subject": rec["subject"],
                    "label_parametric": rec["label_parametric"],
                    "label_incontext": rec["label_incontext"],
                    "conflict_position": position,
                    "target_tokens": length,
                    "prompt": prompt,
                    "prompt_no_context": rec["prompt_no_context"],
                    "source": rec.get("source", ""),
                })

        with open(out_path, "w") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[make_long_context] Saved {len(out_records)} records → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict_data", default="data/output/short_conflict.jsonl")
    parser.add_argument("--lengths", nargs="+", type=int, default=[500, 2000, 8000, 32000])
    parser.add_argument("--positions", nargs="+", default=["beginning", "middle", "end"])
    parser.add_argument("--n_per_condition", type=int, default=100)
    parser.add_argument("--out_dir", default="data/output/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process(
        conflict_data_path=args.conflict_data,
        lengths=args.lengths,
        positions=args.positions,
        n_per_condition=args.n_per_condition,
        out_dir=args.out_dir,
        seed=args.seed,
    )
