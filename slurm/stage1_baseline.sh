#!/bin/bash
#SBATCH -J stage1_baseline
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/stage1_baseline.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/stage1_baseline.e%j
#SBATCH --time 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# Stage 1: Baseline Experiment (단일 모델)
# Usage: sbatch slurm/stage1_baseline.sh [model_name]
# Example: sbatch slurm/stage1_baseline.sh qwen3.5-4b
# ============================================================

set -e

# 작업 디렉토리
WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "Stage 1 Baseline Experiment"
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

# 모델 선택 (인자로 받거나 기본값)
MODEL_NAME=${1:-"qwen3.5-4b"}

echo "Model: $MODEL_NAME"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# 실험 실행
python3 -u project/experiments/stage1_baseline.py \
    --model "$MODEL_NAME" \
    --data_dir data/output/ \
    --out results/stage1_baseline_${MODEL_NAME}.json \
    --n_samples 100

echo ""
echo "=========================================="
echo "Stage 1 Baseline Complete!"
echo "=========================================="
echo "Results saved to: results/stage1_baseline_${MODEL_NAME}.json"
echo "End time: $(date)"
