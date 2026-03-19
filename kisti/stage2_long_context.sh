#!/bin/bash
#SBATCH -J stage2_long
#SBATCH -p amd_a100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/stage2_long.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/stage2_long.e%j
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# KISTI Long Context Experiment (64K, 128K, 256K)
# Usage: sbatch kisti/stage2_long_context.sh [model_name]
# ============================================================

set -e

WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "  Stage 2: Long Context (KISTI A100)"
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

# 환경변수
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$WORK_DIR/models_cache
export TRANSFORMERS_CACHE=$WORK_DIR/models_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 모델 선택
MODEL_NAME=${1:-"qwen3.5-4b"}

echo "Model: $MODEL_NAME"
echo ""

# Flash Attention 확인
python -c "
try:
    from fla.ops.gla import fused_chunk_gla
    print('✓ Flash Linear Attention available')
except Exception as e:
    print(f'⚠️  Flash Linear Attention not available: {e}')
    print('Will use fallback (slower)')
"

echo ""
echo "=========================================="
echo "  Running Long Context Lengths"
echo "=========================================="

# 긴 context만 실행 (64K, 128K, 256K, 512K, 1M)
LENGTHS=("64k" "128k" "256k" "512k" "1m")

for length in "${LENGTHS[@]}"; do
    echo ""
    echo ">>> Length: $length"

    OUTPUT_FILE="results/stage2_${MODEL_NAME}_${length}_kisti.json"
    LOG_FILE="logs/stage2_${MODEL_NAME}_${length}.log"

    # Length tag를 숫자로 변환
    if [ "$length" = "64k" ]; then
        LENGTH_NUM=64000
    elif [ "$length" = "128k" ]; then
        LENGTH_NUM=128000
    elif [ "$length" = "256k" ]; then
        LENGTH_NUM=256000
    elif [ "$length" = "512k" ]; then
        LENGTH_NUM=512000
    elif [ "$length" = "1m" ]; then
        LENGTH_NUM=1000000
    fi

    python -u project/experiments/stage2_length_scaling.py \
        --models "$MODEL_NAME" \
        --data_dir data/output/ \
        --lengths $length \
        --out "$OUTPUT_FILE" \
        --n_per_condition 50 \
        2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Length $length completed"
    else
        echo "✗ Length $length failed"
        echo "Previous results are saved"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "  Long Context Experiment Complete!"
echo "=========================================="
echo "Results: results/stage2_${MODEL_NAME}_*_kisti.json"
echo "End time: $(date)"
