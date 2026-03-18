#!/bin/bash
# ============================================================
# Flash Linear Attention 올바른 설치 스크립트
# KISTI 환경에 맞게 조정됨
# ============================================================

set -e

echo "=========================================="
echo "Flash Linear Attention 설치"
echo "=========================================="

# Conda 환경 확인
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: Conda 환경이 활성화되지 않았습니다!"
    echo "실행 전: conda activate hybrid"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Python 및 PyTorch 버전 확인
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# CUDA 모듈 확인
echo "Loaded CUDA module:"
module list 2>&1 | grep cuda || echo "No CUDA module loaded"
echo ""

# 1. 기존 flash-linear-attention 제거
echo "=========================================="
echo "1. Removing old installations..."
echo "=========================================="
pip uninstall -y flash-linear-attention fla-core 2>/dev/null || true
echo ""

# 2. 필수 의존성 설치
echo "=========================================="
echo "2. Installing dependencies..."
echo "=========================================="
pip install einops ninja
echo ""

# 3. Flash Linear Attention 올바른 설치
echo "=========================================="
echo "3. Installing Flash Linear Attention..."
echo "=========================================="
echo "Using flags:"
echo "  --no-build-isolation: Use current env (PyTorch/CUDA)"
echo "  --no-deps: Prevent PyTorch downgrade"
echo ""

pip install -U --no-build-isolation \
    git+https://github.com/fla-org/flash-linear-attention \
    --no-deps

echo ""

# 4. 설치 확인
echo "=========================================="
echo "4. Verifying installation..."
echo "=========================================="
python << 'EOF'
import sys
try:
    import fla
    print(f"✓ fla version: {fla.__version__}")

    # 커널 사용 가능 여부 확인
    from fla.ops.gla import fused_chunk_gla
    print("✓ GLA kernels available")

    print("\n Installation successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Test with a small model"
echo "2. Run Stage 2 experiments"
