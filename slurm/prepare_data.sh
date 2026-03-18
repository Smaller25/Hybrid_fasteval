#!/bin/bash
#SBATCH -J prepare_data
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /scratch/e1887a03/Hybrid_fasteval/logs/prepare_data.o%j
#SBATCH -e /scratch/e1887a03/Hybrid_fasteval/logs/prepare_data.e%j
#SBATCH --time 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# 데이터 준비 스크립트
# Usage: sbatch slurm/prepare_data.sh
# ============================================================

set -e

# 작업 디렉토리
WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "Data Preparation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Miniconda 초기화
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh

# conda 환경 활성화
conda activate hybrid

# Python 및 패키지 확인
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# 환경변수 설정
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$WORK_DIR/models_cache
export TRANSFORMERS_CACHE=$WORK_DIR/models_cache

# 출력 디렉토리 생성
mkdir -p data/output

echo "=========================================="
echo "1. Preparing CounterFact dataset..."
echo "=========================================="
python -u project/data/prepare_counterfact.py

echo ""
echo "=========================================="
echo "2. Preparing ConflictBank dataset..."
echo "=========================================="
# ConflictBank 데이터 준비 (있는 경우)
if [ -f "project/data/prepare_conflictbank.py" ]; then
    python -u project/data/prepare_conflictbank.py
else
    echo "ConflictBank script not found, skipping..."
fi

echo ""
echo "=========================================="
echo "3. Creating long context variants..."
echo "=========================================="
python -u project/data/make_long_context.py

echo ""
echo "=========================================="
echo "Data Preparation Complete!"
echo "=========================================="
echo "Output directory: $WORK_DIR/data/output/"
ls -lh data/output/
echo ""
echo "End time: $(date)"
