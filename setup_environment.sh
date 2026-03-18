#!/bin/bash
# ============================================================
# RunPod/Cloud GPU 환경 설정 스크립트
# 재시작 후 한 번에 환경 복구
# Usage: bash setup_environment.sh
# ============================================================

set -e

echo "=========================================="
echo "  Hybrid_fasteval Environment Setup"
echo "=========================================="
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo ""

# 환경변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=~/models_cache
export TRANSFORMERS_CACHE=~/models_cache

# ============================================================
# 1. 기존 패키지 정리
# ============================================================
echo "[1/6] Cleaning existing packages..."
pip uninstall -y torch torchvision torchaudio triton flash-linear-attention fla-core 2>/dev/null || true
pip cache purge

# ============================================================
# 2. PyTorch Nightly 설치 (torch 먼저!)
# ============================================================
echo "[2/6] Installing PyTorch Nightly..."
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

echo "[2.5/6] Installing torchvision/torchaudio (--no-deps)..."
pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps

# ============================================================
# 3. Triton 3.2.0
# ============================================================
echo "[3/6] Installing Triton 3.2.0..."
pip install triton==3.2.0

# ============================================================
# 4. Core Dependencies
# ============================================================
echo "[4/6] Installing core dependencies..."
pip install einops ninja datasets transformers numpy psutil tqdm

# ============================================================
# 5. Flash-Linear-Attention (제일 마지막!)
# ============================================================
echo "[5/6] Installing Flash-Linear-Attention..."
pip install -U --no-build-isolation \
    git+https://github.com/fla-org/flash-linear-attention \
    --no-deps

# ============================================================
# 6. 검증
# ============================================================
echo "[6/6] Verifying installation..."
python << 'EOF'
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

import triton
print(f"✓ Triton: {triton.__version__}")

import transformers
print(f"✓ Transformers: {transformers.__version__}")

try:
    import fla
    print(f"✓ Flash-Linear-Attention: {fla.__version__}")
except Exception as e:
    print(f"✗ Flash-Linear-Attention: {e}")
    exit(1)

print("\n✅ All packages installed successfully!")
EOF

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo "Next steps:"
echo "  1. Prepare data:  bash prepare_data.sh"
echo "  2. Run experiment: bash run_stage2_safe.sh qwen3.5-4b 50"
