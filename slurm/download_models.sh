#!/bin/bash
#SBATCH -J download_models
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/download_models.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/download_models.e%j
#SBATCH --time 04:00:00
#SBATCH --comment pytorch

# ============================================================
# 모델 가중치 미리 다운로드 (실험 전 1회 실행)
# Usage: sbatch slurm/download_models.sh
# ============================================================

set -e

# 작업 디렉토리
WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "Pre-downloading Model Weights"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Miniconda 초기화 및 환경 활성화
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid

# 환경변수 설정
export HF_HOME=$WORK_DIR/models_cache
export TRANSFORMERS_CACHE=$WORK_DIR/models_cache

# Python으로 모델 다운로드
python3 << 'EOF'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 목록 (load_model.py의 MODEL_REGISTRY와 동일)
MODELS = {
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-4b-base": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "nemotron-h-4b": "nvidia/Nemotron-H-4B-Instruct-128K",
    "nemotron-h-8b": "nvidia/Nemotron-H-8B-Reasoning-128K",
    "olmo-hybrid-7b": "allenai/OLMo-Hybrid-7B",
    "olmo-3-7b": "allenai/OLMo-3-7B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "gemma-3-4b": "google/gemma-3-4b-it",
}

# 우선순위 모델 (실험에서 주로 사용)
PRIORITY_MODELS = [
    "qwen3.5-4b",
    "nemotron-h-4b",
    "olmo-hybrid-7b",
    "olmo-3-7b",
]

print("=" * 60)
print("Downloading priority models first...")
print("=" * 60)
print()

for model_name in PRIORITY_MODELS:
    hf_id = MODELS[model_name]
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

        # 모델 다운로드 (메모리에 로드하지 않고 캐시만)
        print(f"[2/2] Downloading model weights...")
        # download_only로 다운로드만 수행
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=hf_id,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"✓ Model weights cached")
        print(f"✓ {model_name} download complete!\n")

    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        print(f"  Continuing with next model...\n")
        continue

print("\n" + "=" * 60)
print("Optional models (baseline comparisons)...")
print("=" * 60)
print()

OPTIONAL_MODELS = ["llama-3.2-3b", "gemma-3-4b"]

for model_name in OPTIONAL_MODELS:
    hf_id = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name} ({hf_id})")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        print(f"✓ Tokenizer cached")

        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=hf_id,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"✓ Model weights cached")
        print(f"✓ {model_name} download complete!\n")

    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        print(f"  Continuing with next model...\n")
        continue

print("\n" + "=" * 60)
print("Download Summary")
print("=" * 60)

# 캐시 디렉토리 크기 확인
import subprocess
cache_size = subprocess.check_output(
    ["du", "-sh", os.environ["HF_HOME"]],
    text=True
).split()[0]

print(f"Cache directory: {os.environ['HF_HOME']}")
print(f"Total cache size: {cache_size}")
print(f"\nAll models ready for experiments!")

EOF

echo ""
echo "=========================================="
echo "Model Download Complete!"
echo "=========================================="
echo "Cache location: $WORK_DIR/models_cache"
ls -lh $WORK_DIR/models_cache/hub/ 2>/dev/null | head -20 || echo "Cache directory structure may vary"
echo ""
echo "End time: $(date)"
