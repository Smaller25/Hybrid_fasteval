#!/usr/bin/env python3
"""
Interactive model download script for login node
(로그인 노드에서 직접 실행용 - 2시간 제한 주의)

Usage:
    cd /scratch/e1887a03/Hybrid_fasteval
    conda activate hybrid
    python slurm/download_models_interactive.py
"""

import os
import sys

# 환경변수 설정
WORK_DIR = "/scratch/e1887a03/Hybrid_fasteval"
os.environ["HF_HOME"] = f"{WORK_DIR}/models_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{WORK_DIR}/models_cache"

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 모델 목록
MODELS = {
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "nemotron-h-4b": "nvidia/Nemotron-H-4B-Instruct-128K",
    "olmo-hybrid-7b": "allenai/OLMo-Hybrid-7B",
    "olmo-3-7b": "allenai/OLMo-3-7B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "gemma-3-4b": "google/gemma-3-4b-it",
}

def download_model(model_name: str, hf_id: str):
    """단일 모델 다운로드"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name} ({hf_id})")
    print(f"{'='*60}")

    try:
        # 토크나이저 다운로드
        print(f"[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            trust_remote_code=True,
        )
        print(f"✓ Tokenizer cached")

        # 모델 가중치 다운로드
        print(f"[2/2] Downloading model weights...")
        snapshot_download(
            repo_id=hf_id,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"✓ Model weights cached")
        print(f"✓ {model_name} download complete!")
        return True

    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("Model Download Tool")
    print("=" * 60)
    print(f"Cache directory: {os.environ['HF_HOME']}")
    print()

    # 우선순위 모델
    priority = ["qwen3.5-4b", "nemotron-h-4b", "olmo-hybrid-7b", "olmo-3-7b"]

    print("Priority models (for main experiments):")
    for i, name in enumerate(priority, 1):
        print(f"  {i}. {name} ({MODELS[name]})")

    print("\nOptional models (for baseline comparisons):")
    optional = ["llama-3.2-3b", "gemma-3-4b"]
    for i, name in enumerate(optional, len(priority)+1):
        print(f"  {i}. {name} ({MODELS[name]})")

    print("\nOptions:")
    print("  a - Download all models")
    print("  p - Download priority models only (recommended)")
    print("  1-6 - Download specific model")
    print("  q - Quit")

    choice = input("\nYour choice: ").strip().lower()

    if choice == 'q':
        print("Cancelled.")
        return

    to_download = []
    if choice == 'a':
        to_download = list(MODELS.keys())
    elif choice == 'p':
        to_download = priority
    elif choice.isdigit() and 1 <= int(choice) <= len(MODELS):
        idx = int(choice) - 1
        all_models = priority + optional
        to_download = [all_models[idx]]
    else:
        print("Invalid choice.")
        return

    print(f"\nWill download {len(to_download)} model(s):")
    for name in to_download:
        print(f"  - {name}")

    confirm = input("\nProceed? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # 다운로드 실행
    success_count = 0
    for name in to_download:
        if download_model(name, MODELS[name]):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Download complete: {success_count}/{len(to_download)} successful")
    print("=" * 60)

if __name__ == "__main__":
    main()
