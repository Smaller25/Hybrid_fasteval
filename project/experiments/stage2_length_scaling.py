"""
Stage 2 — Context Length Scaling Experiment

For each model × length × position condition, measure:
  1. context_following_rate
  2. avg logit_gap

Expected results:
  - Hypothesis: Hybrid models show steeper decline in context_following_rate
    as context length increases vs. pure Transformer baselines.
  - Track whether OLMo Hybrid deviates from OLMo 3 (controlled pair).

Usage:
    python experiments/stage2_length_scaling.py \
        --models qwen3.5-4b nemotron-h-4b olmo-hybrid-7b olmo-3-7b \
        --data_dir data/output/ \
        --lengths 500 2k 8k 32k \
        --out results/length_scaling.json \
        [--n_per_condition 50]
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.load_model import MODEL_REGISTRY, load_model
from experiments.utils import (
    compute_logit_gap,
    generate_answer,
    load_jsonl,
    save_json,
    score_response,
)

LENGTH_TAGS = ["500", "2k", "8k", "32k"]


def evaluate_one_condition(
    model,
    tokenizer,
    records: list[dict],
    max_new_tokens: int = 20,
) -> dict:
    """Evaluate a single (length, position) condition."""
    results = []
    for rec in records:
        prompt = rec["prompt"]
        lp = rec["label_parametric"]
        li = rec["label_incontext"]

        resp = generate_answer(model, tokenizer, prompt, max_new_tokens)
        verdict = score_response(resp, lp, li)
        lg = compute_logit_gap(model, tokenizer, prompt, lp, li)

        results.append({
            "id": rec.get("id", ""),
            "verdict": verdict,
            "logit_gap": lg["logit_gap"],
        })

    counter = Counter(r["verdict"] for r in results)
    n = counter["incontext"] + counter["parametric"]
    cfr = counter["incontext"] / n if n > 0 else float("nan")

    gaps = [r["logit_gap"] for r in results if r["logit_gap"] == r["logit_gap"]]
    avg_gap = sum(gaps) / len(gaps) if gaps else float("nan")

    return {
        "n": len(results),
        "context_following_rate": cfr,
        "avg_logit_gap": avg_gap,
        "verdict_counts": dict(counter),
    }


def run_length_scaling(
    models: list[str],
    data_dir: str,
    lengths: list[str],
    n_per_condition: int,
    device_map: str,
) -> dict:
    all_results = {}

    for model_name in models:
        print(f"\n[stage2] Loading model: {model_name}")
        model, tokenizer = load_model(model_name, device_map=device_map)
        model_results = {}

        for length_tag in lengths:
            data_path = os.path.join(data_dir, f"long_conflict_{length_tag}.jsonl")
            if not os.path.exists(data_path):
                print(f"  [skip] {data_path} not found")
                continue

            records_all = load_jsonl(data_path)
            # Group by position
            by_position = defaultdict(list)
            for r in records_all:
                by_position[r.get("conflict_position", "unknown")].append(r)

            for position, pos_records in by_position.items():
                pos_records = pos_records[:n_per_condition]
                print(f"  length={length_tag}, position={position}, n={len(pos_records)}")
                condition_result = evaluate_one_condition(model, tokenizer, pos_records)
                key = f"{length_tag}_{position}"
                model_results[key] = {
                    "length": length_tag,
                    "position": position,
                    **condition_result,
                }

        all_results[model_name] = model_results

        # Free device memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return all_results


def print_summary(results: dict):
    print(f"\n{'='*70}")
    print(f"  STAGE 2 SUMMARY — Context Length Scaling")
    print(f"{'='*70}")
    for model_name, conditions in results.items():
        print(f"\n  [{model_name}]")
        # Sort by length tag
        for key in sorted(conditions.keys()):
            c = conditions[key]
            print(
                f"    {key:20s}  CFR={c['context_following_rate']:.3f}  "
                f"gap={c['avg_logit_gap']:+.3f}  n={c['n']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3.5-4b", "olmo-3-7b"],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_dir", default="data/output/")
    parser.add_argument("--lengths", nargs="+", default=LENGTH_TAGS)
    parser.add_argument("--n_per_condition", type=int, default=100)
    parser.add_argument("--out", default="results/length_scaling.json")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    results = run_length_scaling(
        models=args.models,
        data_dir=args.data_dir,
        lengths=args.lengths,
        n_per_condition=args.n_per_condition,
        device_map=args.device_map,
    )

    save_json(results, args.out)
    print_summary(results)
    print(f"\n[stage2] Saved → {args.out}")


if __name__ == "__main__":
    main()
