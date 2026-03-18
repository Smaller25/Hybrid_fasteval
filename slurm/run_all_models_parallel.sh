#!/bin/bash
#SBATCH -J hybrid_all_parallel
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH -o logs/all_parallel.o%j
#SBATCH -e logs/all_parallel.e%j
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:4
#SBATCH --comment pytorch

# ============================================================
# 모든 모델 병렬 실행 (GPU 4개 동시 활용)
# 각 모델을 별도 GPU에서 동시 실행하여 시간 절약
# Usage: sbatch slurm/run_all_models_parallel.sh
# ============================================================

set -e

# 작업 디렉토리
WORK_DIR=/scratch/$USER/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "All Models Parallel Execution"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo ""

# 환경 로드
module purge
module load gcc/10.2.0 cuda/11.8 cudnn/8.6.0
source venv/bin/activate

# 환경변수 설정
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$WORK_DIR/models_cache
export TRANSFORMERS_CACHE=$WORK_DIR/models_cache

# 실험 타입 선택 (stage1 또는 stage2)
STAGE=${1:-"stage2"}
echo "Running: $STAGE"
echo ""

# 모델 리스트
MODELS=("qwen3.5-4b" "nemotron-h-4b" "olmo-hybrid-7b" "olmo-3-7b")

# 병렬 실행 함수
run_model() {
    local MODEL=$1
    local GPU_ID=$2
    local STAGE=$3

    echo "[GPU $GPU_ID] Starting $MODEL at $(date)"

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    if [ "$STAGE" == "stage1" ]; then
        python3 -u project/experiments/stage1_baseline.py \
            --model "$MODEL" \
            --data_dir data/output/ \
            --out results/stage1_baseline_${MODEL}.json \
            --n_samples 100 \
            > logs/${MODEL}_gpu${GPU_ID}.log 2>&1
    else
        python3 -u project/experiments/stage2_length_scaling.py \
            --models "$MODEL" \
            --data_dir data/output/ \
            --lengths 500 2k 8k 32k \
            --out results/stage2_length_${MODEL}.json \
            --n_per_condition 50 \
            > logs/${MODEL}_gpu${GPU_ID}.log 2>&1
    fi

    echo "[GPU $GPU_ID] Finished $MODEL at $(date)"
}

# 백그라운드로 4개 모델 동시 실행
for i in "${!MODELS[@]}"; do
    run_model "${MODELS[$i]}" "$i" "$STAGE" &
done

# 모든 작업 완료 대기
wait

echo ""
echo "=========================================="
echo "All Models Complete!"
echo "=========================================="
echo "Check individual logs in logs/ directory"
echo "End time: $(date)"

# 결과 요약
echo ""
echo "Results:"
for MODEL in "${MODELS[@]}"; do
    RESULT_FILE="results/${STAGE}_*_${MODEL}.json"
    if [ -f $RESULT_FILE ]; then
        echo "  ✓ $MODEL: $RESULT_FILE"
    else
        echo "  ✗ $MODEL: FAILED"
    fi
done
