#!/bin/bash
# ============================================================
# LM Evaluation - RULER (Long-context Retrieval)
# Tests: 4K, 8K, 16K, 32K, 64K, 128K contexts
# Usage: bash run_lm_eval_ruler.sh
# ============================================================

set -e

echo "==========================================="
echo "  RULER - Long-Context Retrieval"
echo "==========================================="
echo ""
echo "⚠️  WARNING: RULER requires significant VRAM!"
echo "   - 32K: ~40-50 GB"
echo "   - 64K: ~60-70 GB"
echo "   - 128K: ~80 GB (A100 80GB required)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Results directory
mkdir -p results/lm_eval

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
    echo "  RULER: $name"
    echo "==========================================="
    echo "Model: $hf_id"
    echo ""

    # Build model args
    MODEL_ARGS="pretrained=$hf_id,dtype=bfloat16"
    if [ "$trust_code" = "true" ]; then
        MODEL_ARGS="$MODEL_ARGS,trust_remote_code=True"
    fi

    # Run RULER evaluation
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks ruler \
        --batch_size 1 \
        --output_path "results/lm_eval/${name}_ruler.json" \
        --log_samples

    if [ $? -eq 0 ]; then
        echo "✓ $name RULER completed"
    else
        echo "✗ $name RULER failed (likely OOM)"
        echo "  Try with smaller context lengths"
    fi

    # Memory cleanup
    echo "Waiting 30s for memory cleanup..."
    sleep 30
done

echo ""
echo "==========================================="
echo "  RULER Evaluation Complete!"
echo "==========================================="
echo "Results: results/lm_eval/*_ruler.json"
echo ""

# Summary
python << 'EOF'
import json
from pathlib import Path

results_dir = Path("results/lm_eval")
models = ["qwen3.5-4b", "olmo-hybrid-7b", "olmo-3-7b"]

print("\n" + "="*60)
print("  RULER Summary (Long-Context Retrieval)")
print("="*60)

for model in models:
    result_file = results_dir / f"{model}_ruler.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)

        print(f"\n{model.upper()}:")
        results = data.get("results", {})

        # RULER typically has subtasks by context length
        ruler_tasks = {k: v for k, v in results.items() if 'ruler' in k.lower()}
        if ruler_tasks:
            for task, metrics in sorted(ruler_tasks.items()):
                if isinstance(metrics, dict) and 'acc' in metrics:
                    # Extract context length from task name
                    ctx_len = "unknown"
                    if "4k" in task.lower():
                        ctx_len = "4K"
                    elif "8k" in task.lower():
                        ctx_len = "8K"
                    elif "16k" in task.lower():
                        ctx_len = "16K"
                    elif "32k" in task.lower():
                        ctx_len = "32K"
                    elif "64k" in task.lower():
                        ctx_len = "64K"
                    elif "128k" in task.lower():
                        ctx_len = "128K"

                    print(f"  {ctx_len:8s}: {metrics['acc']:.4f}")
        else:
            # Overall RULER score
            if 'ruler' in results:
                print(f"  Overall: {results['ruler'].get('acc', 'N/A')}")
EOF
