#!/bin/bash
# ============================================================
# 데이터 준비 스크립트
# CounterFact → Long Context 데이터 생성
# ============================================================

set -e

echo "=========================================="
echo "  Data Preparation"
echo "=========================================="

# 디렉토리 생성
mkdir -p data/output results logs

# 1단계: Short conflict 데이터
echo "[1/2] Generating short conflict data..."
python project/data/prepare_counterfact.py \
    --n 200 \
    --out data/output/short_conflict.jsonl

# 2단계: Long context 데이터 (500, 2k, 8k, 32k)
echo "[2/2] Generating long context data..."
python project/data/make_long_context.py \
    --conflict_data data/output/short_conflict.jsonl \
    --lengths 500 2000 8000 32000 \
    --positions beginning middle end \
    --n_per_condition 50 \
    --out_dir data/output/

echo ""
echo "=========================================="
echo "  Data Preparation Complete!"
echo "=========================================="
ls -lh data/output/long_conflict_*.jsonl
