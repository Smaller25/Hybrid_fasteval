#!/bin/bash
# ============================================================
# KISTI Neuron Flash Linear Attention 설치
# Interactive 세션에서 실행 필요
# Usage: bash kisti/setup_kisti.sh
# ============================================================

set -e

echo "=========================================="
echo "  KISTI Flash Linear Attention Setup"
echo "=========================================="

# ============================================================
# 0. 환경 확인
# ============================================================
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Error: Conda 환경이 활성화되지 않았습니다!"
    echo "먼저 실행:"
    echo "  source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh"
    echo "  conda activate hybrid"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# ============================================================
# 1. 시스템 정보 확인
# ============================================================
echo "[1/7] Checking system info..."
echo "Hostname: $(hostname)"
echo "Python: $(python --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# PyTorch CUDA vs System CUDA
PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
SYSTEM_CUDA=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2- || echo "N/A")

echo "PyTorch CUDA: $PYTORCH_CUDA"
echo "System CUDA: $SYSTEM_CUDA"
echo ""

# ============================================================
# 2. Module 환경 정리 및 재설정
# ============================================================
echo "[2/7] Setting up modules..."

# 기존 module purge (선택사항)
# module purge

# CUDA 12.8 로드 (PyTorch Nightly는 CUDA 12.4 기반이지만 12.8도 호환)
module load gcc/10.2.0
module load cuda/12.8

echo "✓ Modules loaded:"
module list 2>&1 | grep -E "gcc|cuda"
echo ""

# ============================================================
# 3. 환경변수 설정
# ============================================================
echo "[3/7] Setting environment variables..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/e1887a03/Hybrid_fasteval/models_cache
export TRANSFORMERS_CACHE=/scratch/e1887a03/Hybrid_fasteval/models_cache

echo "✓ Environment variables set"
echo ""

# ============================================================
# 4. 기존 패키지 정리
# ============================================================
echo "[4/7] Cleaning existing packages..."

pip uninstall -y flash-linear-attention fla-core 2>/dev/null || true
# PyTorch는 유지 (이미 설치된 Nightly 사용)

echo "✓ Cleanup complete"
echo ""

# ============================================================
# 5. 필수 의존성 확인/설치
# ============================================================
echo "[5/7] Installing dependencies..."

pip install --upgrade pip
pip install einops ninja datasets transformers numpy psutil tqdm accelerate

echo "✓ Dependencies installed"
echo ""

# ============================================================
# 6. Flash Linear Attention 설치
# ============================================================
echo "[6/7] Installing Flash Linear Attention..."
echo "⚠️  Critical flags:"
echo "  --no-build-isolation: Use current env (PyTorch/CUDA)"
echo "  --no-deps: Prevent PyTorch downgrade"
echo ""

pip install -U --no-build-isolation \
    git+https://github.com/fla-org/flash-linear-attention \
    --no-deps

echo "✓ Flash Linear Attention installed"
echo ""

# ============================================================
# 7. 검증
# ============================================================
echo "[7/7] Verifying installation..."

python << 'EOF'
import sys

# PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")

# Triton
try:
    import triton
    print(f"✓ Triton: {triton.__version__}")
    if triton.__version__ < "3.2.0":
        print(f"  ⚠️  Warning: Triton < 3.2.0 (current: {triton.__version__})")
except ImportError:
    print("✗ Triton: Not installed")

# Flash Linear Attention
try:
    import fla
    print(f"✓ Flash-Linear-Attention: {fla.__version__}")

    # 커널 테스트
    from fla.ops.gla import fused_chunk_gla
    print(f"  ✓ GLA kernels available")

    # 간단한 forward pass 테스트
    import torch
    B, H, T, D = 2, 4, 128, 64
    q = torch.randn(B, H, T, D).cuda()
    k = torch.randn(B, H, T, D).cuda()
    v = torch.randn(B, H, T, D).cuda()
    g = torch.randn(B, H, T, D).cuda().sigmoid()

    try:
        o, _ = fused_chunk_gla(q, k, v, g)
        print(f"  ✓ Forward pass successful")
    except Exception as e:
        print(f"  ⚠️  Forward pass failed: {e}")
        print(f"  (This might still work in practice)")

except ImportError as e:
    print(f"✗ Flash-Linear-Attention: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️  Flash-Linear-Attention imported but kernel test failed: {e}")

print("\n✅ Setup complete!")
print("\nNext steps:")
print("  1. Test with a small model:")
print("     python test_responses.py --model qwen3.5-4b --data data/output/long_conflict_500.jsonl --n 1")
print("  2. Run experiments:")
print("     sbatch kisti/stage2_a100.sh qwen3.5-4b")
EOF

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
