"""
Stage 1 — Behavioral Baseline

For each conflict record, measure:
  1. context_following_rate = #{incontext} / (#{incontext} + #{parametric})
  2. logit_gap per item

Optionally run a parametric knowledge verification pass first to filter out
items the model doesn't actually know (to avoid noisy signal).

Usage:
    python experiments/stage1_baseline.py \
        --model qwen3.5-4b \
        --data data/output/short_conflict.jsonl \
        --out results/baseline_qwen3.5-4b.json \
        [--verify_knowledge] \
        [--n 50]
"""

import argparse
import json
import os
import sys
from collections import Counter

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
    verify_parametric_knowledge,
)


def run_baseline(
    model,
    tokenizer,
    records: list[dict],
    verify_knowledge: bool = False,
    max_new_tokens: int = 20,
) -> dict:
    results = []
    skipped_no_knowledge = 0

    for rec in tqdm(records, desc="Behavioral baseline"):
        subject = rec["subject"]
        lp = rec["label_parametric"]
        li = rec["label_incontext"]

        if verify_knowledge:
            if not verify_parametric_knowledge(model, tokenizer, subject, lp):
                skipped_no_knowledge += 1
                continue

        # Run with context (conflict)
        resp_context = generate_answer(model, tokenizer, rec["prompt_with_context"], max_new_tokens)
        verdict_context = score_response(resp_context, lp, li)

        # Run without context (baseline)
        resp_no_context = generate_answer(model, tokenizer, rec["prompt_no_context"], max_new_tokens)
        verdict_no_context = score_response(resp_no_context, lp, li)

        # Logit gap for with-context
        lg = compute_logit_gap(model, tokenizer, rec["prompt_with_context"], lp, li)

        results.append({
            "id": rec.get("id", ""),
            "subject": subject,
            "label_parametric": lp,
            "label_incontext": li,
            "verdict_with_context": verdict_context,
            "verdict_no_context": verdict_no_context,
            "response_with_context": resp_context,
            "response_no_context": resp_no_context,
            **{f"logit_{k}": v for k, v in lg.items()},
        })

    # Aggregate metrics
    counter_ctx = Counter(r["verdict_with_context"] for r in results)
    counter_no_ctx = Counter(r["verdict_no_context"] for r in results)

    n_ctx = counter_ctx["incontext"] + counter_ctx["parametric"]
    cfr = counter_ctx["incontext"] / n_ctx if n_ctx > 0 else float("nan")

    logit_gaps = [r["logit_logit_gap"] for r in results]
    avg_gap = sum(logit_gaps) / len(logit_gaps) if logit_gaps else float("nan")

    summary = {
        "n_evaluated": len(results),
        "n_skipped_no_knowledge": skipped_no_knowledge,
        "context_following_rate": cfr,
        "avg_logit_gap": avg_gap,
        "verdict_with_context_counts": dict(counter_ctx),
        "verdict_no_context_counts": dict(counter_no_ctx),
    }

    return {"summary": summary, "per_item": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data", default="data/output/short_conflict.jsonl")
    parser.add_argument("--out", default="results/baseline.json")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N records")
    parser.add_argument("--verify_knowledge", action="store_true",
                        help="Skip items the model doesn't know parametrically")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device_map=args.device_map)

    records = load_jsonl(args.data)
    if args.n:
        records = records[: args.n]
    print(f"[stage1_baseline] Evaluating {len(records)} records with {args.model}")

    out = run_baseline(model, tokenizer, records, verify_knowledge=args.verify_knowledge)
    out["model"] = args.model
    out["data"] = args.data

    save_json(out, args.out)
    print(f"\n{'='*50}")
    print(f"  Model: {args.model}")
    print(f"  context_following_rate : {out['summary']['context_following_rate']:.3f}")
    print(f"  avg_logit_gap          : {out['summary']['avg_logit_gap']:.3f}")
    print(f"  n_evaluated            : {out['summary']['n_evaluated']}")
    print(f"{'='*50}")
    print(f"[stage1_baseline] Saved → {args.out}")


if __name__ == "__main__":
    main()
