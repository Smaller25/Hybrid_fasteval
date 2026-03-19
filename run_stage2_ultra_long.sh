#!/bin/bash
# ============================================================
# RunPod Long Context Experiment (64K-128K, 선택적 256K)
# 메모리 최적화 + 개별 길이 저장
# Usage: bash run_stage2_ultra_long.sh [model_name] [n_per_condition]
# ============================================================

set -e

MODEL=${1:-"qwen3.5-4b"}
N_PER_CONDITION=${2:-30}

echo "==========================================="
echo "  Long Context Experiment (RunPod)"
echo "==========================================="
echo "Model: $MODEL"
echo "Samples per condition: $N_PER_CONDITION"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo ""

# 환경변수 - 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export HF_HOME=~/models_cache
export TRANSFORMERS_CACHE=~/models_cache

# 결과 디렉토리 생성
mkdir -p results logs

# 현실적 범위: 64K, 128K (256K는 A100 80GB에서 Qwen만 선택적)
# 512K, 1M 제외 이유:
#   - Qwen native max: 262K (YaRN extrapolation은 공식 지원 밖)
#   - VRAM: 512K ≈ 90GB+, 1M ≈ 170GB+ (불가능)
LENGTHS=("64k" "128k")

for length in "${LENGTHS[@]}"; do
    echo ""
    echo "==========================================="
    echo "  Running Ultra-Long: $length"
    echo "==========================================="

    OUTPUT_FILE="results/stage2_${MODEL}_${length}_runpod.json"
    LOG_FILE="logs/stage2_${MODEL}_${length}.log"

    # GPU 메모리 확인
    echo "Before experiment:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    echo ""

    # 실험 실행
    python -u project/experiments/stage2_length_scaling.py \
        --models "$MODEL" \
        --data_dir data/output/ \
        --lengths $length \
        --out "$OUTPUT_FILE" \
        --n_per_condition $N_PER_CONDITION \
        2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Length $length completed"
        echo "  Results: $OUTPUT_FILE"

        # GPU 메모리 확인
        echo "After experiment:"
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    else
        echo "✗ Length $length failed (likely OOM)"
        echo "  Previous results are saved"
        echo "  Try reducing n_per_condition or use larger GPU"

        # 메모리 정보 저장
        nvidia-smi > logs/oom_${length}.log
        break
    fi

    echo ""
    echo "Waiting 30 seconds for memory cleanup..."
    sleep 30
done

echo ""
echo "==========================================="
echo "  Ultra-Long Context Experiment Complete!"
echo "==========================================="
echo "Results: results/stage2_${MODEL}_*_runpod.json"
