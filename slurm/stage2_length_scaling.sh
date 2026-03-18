#!/bin/bash
#SBATCH -J stage2_length
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/stage2_length.o%j
#SBATCH -e logs/stage2_length.e%j
#SBATCH --time 08:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# Stage 2: Length Scaling Experiment (단일 모델)
# Usage: sbatch slurm/stage2_length_scaling.sh [model_name]
# Example: sbatch slurm/stage2_length_scaling.sh qwen3.5-4b
# ============================================================

set -e

# 작업 디렉토리
WORK_DIR=/scratch/$USER/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "Stage 2 Length Scaling Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo ""

# 환경 로드
module purge
module load gcc/10.2.0 cuda/11.8 cudnn/8.6.0
source venv/bin/activate

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
python3 -u project/experiments/stage2_length_scaling.py \
    --models "$MODEL_NAME" \
    --data_dir data/output/ \
    --lengths 500 2k 8k 32k \
    --out results/stage2_length_${MODEL_NAME}.json \
    --n_per_condition 50

echo ""
echo "=========================================="
echo "Stage 2 Length Scaling Complete!"
echo "=========================================="
echo "Results saved to: results/stage2_length_${MODEL_NAME}.json"
echo "End time: $(date)"
