#!/bin/bash
# ============================================================
# LM Evaluation Harness - Standard + Recall-intensive Benchmarks
# Usage:
#   bash run_lm_eval.sh              # Standard + Recall-intensive
#   bash run_lm_eval.sh standard     # Standard only (faster)
#   bash run_lm_eval.sh recall       # Recall-intensive only
# ============================================================

set -e

MODE=${1:-"all"}

echo "==========================================="
echo "  LM Evaluation Harness"
echo "==========================================="
echo "Mode: $MODE"
echo ""

# Results directory
mkdir -p results/lm_eval

# Tasks to evaluate
# Standard benchmarks (general capabilities)
TASKS_STANDARD="hellaswag,ai2_arc,gsm8k,winogrande,truthfulqa_mc2"

# Recall-intensive / Long-context benchmarks (SCROLLS suite)
TASKS_RECALL="drop,scrolls_narrativeqa,scrolls_govreport,scrolls_qmsum,scrolls_qasper"

# Select tasks based on mode
case $MODE in
    "standard")
        TASKS=$TASKS_STANDARD
        echo "Tasks: Standard benchmarks only"
        ;;
    "recall")
        TASKS=$TASKS_RECALL
        echo "Tasks: Recall-intensive benchmarks only"
        ;;
    "all"|*)
        TASKS="$TASKS_STANDARD,$TASKS_RECALL"
        echo "Tasks: Standard + Recall-intensive"
        ;;
esac

echo "Benchmarks: $TASKS"
echo ""

# Models to evaluate
MODELS=(
    "Qwen/Qwen3.5-4B:qwen3.5-4b:false"
    "allenai/Olmo-Hybrid-7B:olmo-hybrid-7b:true"
    "allenai/Olmo-3-1025-7B:olmo-3-7b:false"
)

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r hf_id name trust_code <<< "$model_info"

    echo ""
    echo "==========================================="
    echo "  Evaluating: $name"
    echo "==========================================="
    echo "Model: $hf_id"
    echo "Trust remote code: $trust_code"
    echo ""

    # Build model args
    MODEL_ARGS="pretrained=$hf_id,dtype=bfloat16"
    if [ "$trust_code" = "true" ]; then
        MODEL_ARGS="$MODEL_ARGS,trust_remote_code=True"
    fi

    # Run evaluation
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks $TASKS \
        --batch_size auto \
        --output_path "results/lm_eval/${name}.json" \
        --log_samples

    if [ $? -eq 0 ]; then
        echo "✓ $name evaluation completed"
    else
        echo "✗ $name evaluation failed"
    fi

    # Memory cleanup
    echo "Waiting 30s for memory cleanup..."
    sleep 30
done

echo ""
echo "==========================================="
echo "  All Evaluations Complete!"
echo "==========================================="
echo "Results: results/lm_eval/*.json"
echo ""

# Summary
echo "Generating summary..."
python << 'EOF'
import json
import os
from pathlib import Path

results_dir = Path("results/lm_eval")
models = ["qwen3.5-4b", "olmo-hybrid-7b", "olmo-3-7b"]

print("\n" + "="*60)
print("  LM Eval Summary")
print("="*60)

for model in models:
    result_file = results_dir / f"{model}.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)

        print(f"\n{model.upper()}:")
        results = data.get("results", {})
        for task, metrics in results.items():
            if isinstance(metrics, dict):
                # Get main metric
                if "acc" in metrics:
                    print(f"  {task:20s}: {metrics['acc']:.4f}")
                elif "acc_norm" in metrics:
                    print(f"  {task:20s}: {metrics['acc_norm']:.4f}")
EOF
