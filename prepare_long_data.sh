#!/bin/bash
# ============================================================
# RunPod: 긴 context 데이터 생성 (64K, 128K)
# prepare_data.sh 이후에 실행
# Usage: bash prepare_long_data.sh
# ============================================================

set -e

echo "=========================================="
echo "  Long Context Data Preparation"
echo "=========================================="
echo "Working directory: $(pwd)"
echo ""

# 1. Short conflict 데이터 확인
if [ ! -f "data/output/short_conflict.jsonl" ]; then
    echo "❌ Error: short_conflict.jsonl not found"
    echo "   Run prepare_data.sh first!"
    exit 1
fi

echo "✓ Short conflict data exists"
echo ""

# 2. Long context 데이터 (64K, 128K)
echo "Generating long context data (64K, 128K)..."
echo ""

python project/data/make_long_context.py \
    --conflict_data data/output/short_conflict.jsonl \
    --lengths 64000 128000 \
    --positions beginning middle end \
    --n_per_condition 30 \
    --out_dir data/output/

echo ""
echo "=========================================="
echo "  Data Preparation Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh data/output/long_conflict_64k.jsonl data/output/long_conflict_128k.jsonl 2>/dev/null || echo "⚠️  Some files may not exist"
echo ""
echo "Ready for ultra-long experiments!"
echo "Run: bash run_stage2_ultra_long.sh qwen3.5-4b 30"
