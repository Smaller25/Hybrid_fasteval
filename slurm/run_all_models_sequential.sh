#!/bin/bash
#SBATCH -J hybrid_all_seq
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/all_sequential.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/all_sequential.e%j
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# 모든 모델 순차 실행 (GPU 1개로 안전하게)
# 메모리 이슈가 걱정되거나 안정성을 원할 때 사용
# Usage: sbatch slurm/run_all_models_sequential.sh
# ============================================================

set -e

# 작업 디렉토리
WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "All Models Sequential Execution"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo ""

# Miniconda 초기화 및 환경 활성화
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid

# 환경변수 설정
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$WORK_DIR/models_cache
export TRANSFORMERS_CACHE=$WORK_DIR/models_cache

# 실험 타입 선택 (stage1 또는 stage2)
STAGE=${1:-"stage2"}
echo "Running: $STAGE"
echo ""

# 모델 리스트
MODELS=("qwen3.5-4b" "nemotron-h-4b" "olmo-hybrid-7b" "olmo-3-7b" "llama-3.2-3b" "gemma-3-4b")

# 순차 실행
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running $MODEL at $(date)"
    echo "=========================================="

    if [ "$STAGE" == "stage1" ]; then
        python3 -u project/experiments/stage1_baseline.py \
            --model "$MODEL" \
            --data_dir data/output/ \
            --out results/stage1_baseline_${MODEL}.json \
            --n_samples 100
    else
        python3 -u project/experiments/stage2_length_scaling.py \
            --models "$MODEL" \
            --data_dir data/output/ \
            --lengths 500 2k 8k 32k \
            --out results/stage2_length_${MODEL}.json \
            --n_per_condition 50
    fi

    echo "✓ $MODEL completed at $(date)"

    # GPU 메모리 정리
    sleep 5
done

echo ""
echo "=========================================="
echo "All Models Complete!"
echo "=========================================="
echo "End time: $(date)"

# 결과 요약
echo ""
echo "Results:"
for MODEL in "${MODELS[@]}"; do
    RESULT_FILE="results/${STAGE}_*_${MODEL}.json"
    if ls $RESULT_FILE 1> /dev/null 2>&1; then
        echo "  ✓ $MODEL"
    else
        echo "  ✗ $MODEL: FAILED"
    fi
done
