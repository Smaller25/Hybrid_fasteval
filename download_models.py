#!/usr/bin/env python3
"""
모델 사전 다운로드 스크립트
실험 전에 모델을 미리 캐싱하여 네트워크 이슈 방지
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 캐시 디렉토리 설정
os.environ['HF_HOME'] = os.path.expanduser('~/models_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/models_cache')

# 다운로드할 모델 목록 (hf_id, trust_remote_code)
models = [
    ("Qwen/Qwen3.5-4B", False),
    # 필요시 추가:
    # ("nvidia/Nemotron-H-4B-Instruct-128K", True),
    # ("allenai/OLMo-Hybrid-7B", True),
    # ("allenai/OLMo-3-7B", False),
]

print("=" * 70)
print("  Model Download Script")
print("=" * 70)
print(f"Cache directory: {os.environ['HF_HOME']}")
print(f"Models to download: {len(models)}")
print()

for idx, (model_id, trust_remote_code) in enumerate(models, 1):
    print(f"\n[{idx}/{len(models)}] Downloading: {model_id}")
    print("-" * 70)

    try:
        # Tokenizer 다운로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        print(f"  ✓ Tokenizer downloaded")

        # Model 다운로드 (CPU로만, 실제 로드 안 함)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
            trust_remote_code=trust_remote_code
        )
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  ✓ Model downloaded ({num_params:.2f}B parameters)")

        # 메모리 정리
        del model
        del tokenizer

        print(f"  ✓ {model_id} fully cached!")

    except Exception as e:
        print(f"  ✗ Error downloading {model_id}: {e}")
        continue

print("\n" + "=" * 70)
print("  All downloads completed!")
print("=" * 70)
