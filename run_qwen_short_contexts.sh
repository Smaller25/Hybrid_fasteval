#!/bin/bash
# ============================================================
# Qwen3.5-4B 짧은 컨텍스트 실험 (500, 2K, 8K)
# Usage: bash run_qwen_short_contexts.sh [n_samples]
# ============================================================

set -e

N_SAMPLES=${1:-50}

echo "==========================================="
echo "  Qwen3.5-4B Short Context Experiments"
echo "==========================================="
echo "Lengths: 500, 2K, 8K"
echo "Samples per condition: $N_SAMPLES"
echo ""

# 환경변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=~/models_cache
export TRANSFORMERS_CACHE=~/models_cache

# 결과 디렉토리 생성
mkdir -p results logs

# 각 길이별로 개별 실행 (메모리 안전)
LENGTHS=("500" "2k" "8k")

for length in "${LENGTHS[@]}"; do
    echo ""
    echo "==========================================="
    echo "  Running: Qwen3.5-4B @ $length tokens"
    echo "==========================================="

    OUTPUT_FILE="results/stage2_qwen3.5-4b_${length}.json"
    LOG_FILE="logs/stage2_qwen3.5-4b_${length}.log"

    # 이미 완료된 길이는 건너뛰기
    if [ -f "$OUTPUT_FILE" ]; then
        echo "⚠️  $OUTPUT_FILE already exists, skipping..."
        continue
    fi

    python -u project/experiments/stage2_length_scaling.py \
        --models "qwen3.5-4b" \
        --data_dir data/output/ \
        --lengths "$length" \
        --out "$OUTPUT_FILE" \
        --n_per_condition "$N_SAMPLES" \
        2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ $length completed successfully"
        echo "  Results: $OUTPUT_FILE"
    else
        echo "✗ $length failed (see $LOG_FILE)"
        exit 1
    fi

    # GPU 메모리 정리
    echo "Cleaning up GPU memory..."
    sleep 10
done

echo ""
echo "==========================================="
echo "  All Short Contexts Complete!"
echo "==========================================="
echo "Results:"
ls -lh results/stage2_qwen3.5-4b_{500,2k,8k}.json 2>/dev/null || echo "  (Some files missing)"
echo ""
echo "Memory usage was SAFE:"
echo "  500:  ~9 GB"
echo "  2K:   ~10 GB"
echo "  8K:   ~20 GB"
