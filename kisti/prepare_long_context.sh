#!/bin/bash
# ============================================================
# KISTI: 긴 context 데이터 생성 (128K, 256K)
# /scratch에서 실행 필요
# Usage: bash kisti/prepare_long_context.sh
# ============================================================

set -e

WORK_DIR=/scratch/e1887a03/Hybrid_fasteval
cd $WORK_DIR

echo "=========================================="
echo "  Long Context Data Preparation"
echo "=========================================="
echo "Working directory: $(pwd)"
echo ""

# Conda 환경 활성화
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid

# 환경변수
export HF_HOME=$WORK_DIR/models_cache
export TRANSFORMERS_CACHE=$WORK_DIR/models_cache

# 1. Short conflict 데이터 확인
if [ ! -f "data/output/short_conflict.jsonl" ]; then
    echo "[1/2] Generating short conflict data..."
    python project/data/prepare_counterfact.py \
        --n 200 \
        --out data/output/short_conflict.jsonl
else
    echo "[1/2] Short conflict data already exists"
fi

# 2. Long context 데이터 (기존 + 긴 버전 + 초장문)
echo "[2/2] Generating long context data..."

# 기존: 500, 2k, 8k, 32k (RunPod)
# 긴 버전: 64k, 128k, 256k (KISTI)
# 초장문: 512k, 1M (KISTI Ultra-long)
python project/data/make_long_context.py \
    --conflict_data data/output/short_conflict.jsonl \
    --lengths 500 2000 8000 32000 64000 128000 256000 512000 1000000 \
    --positions beginning middle end \
    --n_per_condition 30 \
    --out_dir data/output/

echo ""
echo "=========================================="
echo "  Data Preparation Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh data/output/long_conflict_*.jsonl
echo ""
echo "Note: 64k, 128k, 256k require significant memory!"
echo "Use A100 80GB for best results."
