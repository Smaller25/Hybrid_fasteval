#!/bin/bash
# ============================================================
# RunPod 재시작 후 원클릭 복구 + 실험 시작
# Usage: bash quick_start.sh [model_name] [n_per_condition]
# Example: bash quick_start.sh qwen3.5-4b 50
# ============================================================

set -e

MODEL=${1:-"qwen3.5-4b"}
N_PER_CONDITION=${2:-50}

echo "============================================"
echo "  Quick Start: Hybrid_fasteval"
echo "============================================"
echo "Model: $MODEL"
echo "Samples per condition: $N_PER_CONDITION"
echo ""

# 1. 환경 설정
echo "Step 1: Setting up environment..."
bash setup_environment.sh

# 2. 데이터 준비
echo ""
echo "Step 2: Preparing data..."
bash prepare_data.sh

# 3. 실험 시작
echo ""
echo "Step 3: Starting experiment..."
chmod +x run_stage2_safe.sh
nohup bash run_stage2_safe.sh "$MODEL" "$N_PER_CONDITION" > logs/run.log 2>&1 &

PID=$!
echo $PID > logs/run.pid

echo ""
echo "============================================"
echo "  Experiment Started!"
echo "============================================"
echo "PID: $PID"
echo "Log: logs/run.log"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/run.log"
echo ""
echo "Stop experiment:"
echo "  kill $PID"
echo "============================================"
