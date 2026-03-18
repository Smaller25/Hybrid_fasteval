#!/bin/bash
#SBATCH -J stage2_1m
#SBATCH -p amd_a100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/stage2_1m.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/stage2_1m.e%j
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# KISTI Ultra-Long Context Experiment (512K, 1M)
# 메모리 최적화 + 개별 길이 저장
# Usage: sbatch kisti/stage2_ultra_long.sh [model_name]
# ============================================================

set -e

WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "  Stage 2: Ultra-Long Context (1M tokens)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo ""

# Miniconda 초기화 및 환경 활성화
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid

# Module 로드
module load gcc/10.2.0 cuda/12.8

# 환경변수 - 메모리 최적화
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$WORK_DIR/models_cache
export TRANSFORMERS_CACHE=$WORK_DIR/models_cache

# 메모리 최적화 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# 모델 선택
MODEL_NAME=${1:-"qwen3.5-4b"}

echo "Model: $MODEL_NAME"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo ""

# Ultra-long context 실험 (512K, 1M)
LENGTHS=("512k" "1m")

for length in "${LENGTHS[@]}"; do
    echo ""
    echo "==========================================="
    echo "  Running Ultra-Long: $length"
    echo "==========================================="

    OUTPUT_FILE="results/stage2_${MODEL_NAME}_${length}_kisti.json"
    LOG_FILE="logs/stage2_${MODEL_NAME}_${length}.log"

    # Length tag를 숫자로 변환
    if [ "$length" = "512k" ]; then
        LENGTH_NUM=512000
    elif [ "$length" = "1m" ]; then
        LENGTH_NUM=1000000
    fi

    # GPU 메모리 확인
    echo "Before experiment:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    echo ""

    # 실험 실행
    python -u project/experiments/stage2_length_scaling.py \
        --models "$MODEL_NAME" \
        --data_dir data/output/ \
        --lengths $LENGTH_NUM \
        --out "$OUTPUT_FILE" \
        --n_per_condition 30 \
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
        echo "  Try reducing n_per_condition or batch size"

        # 메모리 정보 저장
        nvidia-smi > logs/oom_${length}.log
        break
    fi

    echo ""
    echo "Waiting 30 seconds for memory cleanup..."
    sleep 30
done

echo ""
echo "=========================================="
echo "  Ultra-Long Context Experiment Complete!"
echo "=========================================="
echo "Results: results/stage2_${MODEL_NAME}_*_kisti.json"
echo "End time: $(date)"
