#!/bin/bash
#SBATCH -J stage2_a100
#SBATCH -p amd_a100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/stage2_a100.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/stage2_a100.e%j
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# Stage 2: A100 80GB에서 실행 (32K context 지원)
# Usage: sbatch slurm/stage2_a100.sh [model_name]
# ============================================================

set -e

# 작업 디렉토리
WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "Stage 2 on A100 80GB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
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

# 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 모델 선택
MODEL_NAME=${1:-"qwen3.5-4b"}

echo "Model: $MODEL_NAME"
echo "CUDA Device: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# 실험 실행 (32K 포함)
python -u project/experiments/stage2_length_scaling.py \
    --models "$MODEL_NAME" \
    --data_dir data/output/ \
    --lengths 500 2k 8k 32k \
    --out results/stage2_${MODEL_NAME}_a100.json \
    --n_per_condition 50

echo ""
echo "=========================================="
echo "Stage 2 Complete!"
echo "=========================================="
echo "Results: results/stage2_${MODEL_NAME}_a100.json"
echo "End time: $(date)"
