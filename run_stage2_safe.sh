#!/bin/bash
# ============================================================
# Stage 2 안전 실행 스크립트 - 길이별 개별 저장
# 32K에서 터져도 500, 2K, 8K 결과는 안전하게 보존
# ============================================================

set -e

MODEL=${1:-"qwen3.5-4b"}
N_PER_CONDITION=${2:-50}

echo "==========================================="
echo "  Stage 2 Safe Execution (Length-wise)"
echo "==========================================="
echo "Model: $MODEL"
echo "Samples per condition: $N_PER_CONDITION"
echo ""

# 환경변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=~/models_cache
export TRANSFORMERS_CACHE=~/models_cache

# 결과 디렉토리 생성
mkdir -p results logs

# 각 길이별로 개별 실행
LENGTHS=("500" "2k" "8k" "32k")

for length in "${LENGTHS[@]}"; do
    echo ""
    echo "==========================================="
    echo "  Running length: $length"
    echo "==========================================="

    OUTPUT_FILE="results/stage2_${MODEL}_${length}.json"
    LOG_FILE="logs/stage2_${MODEL}_${length}.log"

    python -u project/experiments/stage2_length_scaling.py \
        --models "$MODEL" \
        --data_dir data/output/ \
        --lengths "$length" \
        --out "$OUTPUT_FILE" \
        --n_per_condition "$N_PER_CONDITION" \
        2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Length $length completed successfully"
        echo "  Results: $OUTPUT_FILE"
        echo "  Log: $LOG_FILE"
    else
        echo "✗ Length $length failed!"
        echo "  Previous results (500-${LENGTHS[$((idx-1))]}) are safely saved."
        exit 1
    fi
done

echo ""
echo "==========================================="
echo "  All lengths completed!"
echo "==========================================="
echo "Combining results..."

# 결과 합치기
python << 'EOF'
import json
import sys
from pathlib import Path

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-4b"
lengths = ["500", "2k", "8k", "32k"]

combined = {}
for length in lengths:
    result_file = Path(f"results/stage2_{model}_{length}.json")
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            # 모델별 결과를 combined에 병합
            for model_name, conditions in data.items():
                if model_name not in combined:
                    combined[model_name] = {}
                combined[model_name].update(conditions)

output_file = Path(f"results/stage2_{model}_combined.json")
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"✓ Combined results saved to: {output_file}")
EOF

python -c "import sys; sys.argv.append('$MODEL')" 2>/dev/null

echo ""
echo "==========================================="
echo "  Stage 2 Complete!"
echo "==========================================="
echo "Individual results: results/stage2_${MODEL}_*.json"
echo "Combined results: results/stage2_${MODEL}_combined.json"
