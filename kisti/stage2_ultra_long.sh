#!/bin/bash
#SBATCH -J stage2_256k
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/stage2_256k.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/stage2_256k.e%j
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# KISTI 256K Context Experiment (A100 80GB required)
# Qwen3.5-4B only - experimental range
# Usage: sbatch kisti/stage2_ultra_long.sh qwen3.5-4b
# ============================================================

set -e

WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "  Stage 2: 256K Context (Experimental)"
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

# 모델 선택 (Qwen3.5-4B만 권장)
MODEL_NAME=${1:-"qwen3.5-4b"}

echo "Model: $MODEL_NAME"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo ""
echo "⚠️  256K is experimental (Qwen3.5-4B only)"
echo "   Requires A100 80GB, ~45-55GB VRAM"
echo ""

# 256K 실험 (선택적)
# 512K, 1M 제외 이유:
#   - Qwen native max: 262K (YaRN extrapolation은 공식 지원 밖)
#   - VRAM: 512K ≈ 90GB+, 1M ≈ 170GB+ (불가능)
LENGTHS=("256k")

for length in "${LENGTHS[@]}"; do
    echo ""
    echo "==========================================="
    echo "  Running Experimental Length: $length"
    echo "==========================================="

    OUTPUT_FILE="results/stage2_${MODEL_NAME}_${length}_kisti.json"
    LOG_FILE="logs/stage2_${MODEL_NAME}_${length}.log"

    # GPU 메모리 확인
    echo "Before experiment:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    echo ""

    # 실험 실행
    python -u project/experiments/stage2_length_scaling.py \
        --models "$MODEL_NAME" \
        --data_dir data/output/ \
        --lengths $length \
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
echo "  256K Context Experiment Complete!"
echo "=========================================="
echo "Results: results/stage2_${MODEL_NAME}_256k_kisti.json"
echo "End time: $(date)"
