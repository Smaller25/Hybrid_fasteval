#!/bin/bash
#SBATCH -J hybrid_setup
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/setup.o%j
#SBATCH -e logs/setup.e%j
#SBATCH --time 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

# ============================================================
# 환경 초기 설정 스크립트 (최초 1회만 실행)
# Usage: sbatch slurm/setup_env.sh
# ============================================================

set -e

echo "=========================================="
echo "Hybrid LLM Experiment - Environment Setup"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 작업 디렉토리 설정
WORK_DIR=/scratch/$USER/Hybrid_fasteval
HOME_DIR=/home01/$USER/Hybrid_fasteval

echo "Creating directories..."
mkdir -p $WORK_DIR/{data,results,logs,models_cache}
mkdir -p $HOME_DIR/logs

# 프로젝트 복사 (홈 디렉토리에서 스크래치로)
echo "Copying project to scratch..."
if [ -d "$HOME_DIR" ]; then
    rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
        $HOME_DIR/ $WORK_DIR/
else
    echo "WARNING: $HOME_DIR not found. Please clone the repo first!"
    exit 1
fi

cd $WORK_DIR

# CUDA 환경 설정
echo "Loading modules..."
module purge
module load gcc/10.2.0 cuda/11.8 cudnn/8.6.0

# Python 환경 확인
echo "Python version: $(python3 --version)"
echo "CUDA version: $(nvcc --version | grep release)"

# 가상환경 생성
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 11.8 기준)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
echo "Installing dependencies..."
pip install -r project/requirements.txt

# CUDA 사용 가능 확인
echo ""
echo "=========================================="
echo "Environment Check"
echo "=========================================="
python3 << EOF
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available!")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Virtual environment: $WORK_DIR/venv"
echo "Cache directory: $WORK_DIR/models_cache"
echo ""
echo "Next steps:"
echo "1. Run data preparation (if needed)"
echo "2. Submit Stage 1 or Stage 2 experiments"
echo ""
echo "End time: $(date)"
