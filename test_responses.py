#!/usr/bin/env python3
"""
실제 모델 응답 확인 스크립트
"other" 비율이 높은 이유를 파악하기 위해 실제 응답 예시를 출력
"""
import sys
import os
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "project"))
from models.load_model import load_model
from experiments.utils import generate_answer, score_response

def test_responses(model_name="qwen3.5-4b", data_path="data/output/long_conflict_500.jsonl", n_samples=10):
    """몇 가지 샘플에 대해 실제 응답 확인"""

    # 데이터 로드
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 랜덤 샘플링
    random.seed(42)
    samples = random.sample(records, min(n_samples, len(records)))

    print("=" * 80)
    print(f"Testing model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Samples: {len(samples)}")
    print("=" * 80)

    # 모델 로드
    model, tokenizer = load_model(model_name)

    # 각 샘플 테스트
    for i, rec in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}/{len(samples)}")
        print(f"{'='*80}")
        print(f"Subject: {rec['subject']}")
        print(f"Parametric (True): {rec['label_parametric']}")
        print(f"Incontext (False): {rec['label_incontext']}")
        print(f"Position: {rec.get('conflict_position', 'N/A')}")
        print(f"\nPrompt preview (first 500 chars):")
        print("-" * 80)
        print(rec['prompt'][:500] + "..." if len(rec['prompt']) > 500 else rec['prompt'])
        print("-" * 80)

        # 모델 응답 생성
        response = generate_answer(model, tokenizer, rec['prompt'], max_new_tokens=20)
        verdict = score_response(response, rec['label_parametric'], rec['label_incontext'])

        print(f"\n🤖 Model Response:")
        print(f">>> {response}")
        print(f"\n📊 Verdict: {verdict}")
        print(f"   (Expected: 'parametric' or 'incontext')")

        if verdict == "other":
            print(f"\n⚠️  Classified as 'other'!")
            print(f"   Why? Response doesn't clearly contain:")
            print(f"     - '{rec['label_parametric']}' (parametric)")
            print(f"     - '{rec['label_incontext']}' (incontext)")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--data", default="data/output/long_conflict_500.jsonl")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    test_responses(args.model, args.data, args.n)
