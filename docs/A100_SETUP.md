# A100 80GB 환경 설정 가이드

## 문제 요약

### V100 32GB에서 발생한 이슈
1. **OOM at 32K context** - V100 32GB 메모리 부족
2. **Flash Linear Attention 미작동** - 잘못된 설치 방법
3. **느린 추론 속도** - Fallback 모드로 실행

### A100 80GB로 해결
- ✅ 80GB VRAM - 32K context 여유
- ✅ 올바른 라이브러리 설치
- ✅ 빠른 추론 (Flash Attention 지원)

---

## 📋 설정 순서

### 1단계: Flash Linear Attention 올바른 설치

```bash
# Interactive 세션 시작
srun -p amd_a100_4 --gres=gpu:1 --cpus-per-task=8 \
    --time=2:00:00 --comment pytorch --pty bash

# 작업 디렉토리로 이동
cd /scratch/e1887a03/Hybrid_fasteval

# Conda 환경 활성화
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid

# 설치 스크립트 실행
bash slurm/setup_flash_attn.sh
```

**중요:** 반드시 `--no-build-isolation --no-deps` 플래그 사용!

---

### 2단계: CUDA 버전 확인

```bash
# 시스템 CUDA
nvcc --version

# PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

**버전 불일치 시:**
```bash
# 사용 가능한 CUDA 모듈 확인
module avail cuda

# PyTorch와 맞는 버전 로드
module load cuda/12.8  # 또는 적절한 버전
```

---

### 3단계: 테스트 실행

```bash
# 작은 모델로 테스트 (500 tokens만)
python -u project/experiments/stage2_length_scaling.py \
    --models qwen3.5-4b \
    --data_dir data/output/ \
    --lengths 500 \
    --out results/test.json \
    --n_per_condition 10
```

**확인 사항:**
- ✅ Flash attention 작동 (로그에서 확인)
- ✅ CUDA OOM 없음
- ✅ 정상 완료

---

### 4단계: 전체 실험 실행

#### 옵션 A: Interactive 세션

```bash
# A100 interactive 세션
srun -p amd_a100_4 --gres=gpu:1 --cpus-per-task=16 \
    --time=12:00:00 --comment pytorch --pty bash

cd /scratch/e1887a03/Hybrid_fasteval
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/e1887a03/Hybrid_fasteval/models_cache
export TRANSFORMERS_CACHE=/scratch/e1887a03/Hybrid_fasteval/models_cache

# nohup으로 백그라운드 실행
nohup python -u project/experiments/stage2_length_scaling.py \
    --models qwen3.5-4b \
    --data_dir data/output/ \
    --lengths 500 2k 8k 32k \
    --out results/stage2_qwen3.5-4b_a100.json \
    --n_per_condition 50 \
    > logs/stage2_a100.log 2>&1 &

# 로그 확인
tail -f logs/stage2_a100.log
```

#### 옵션 B: sbatch 제출

```bash
# 스크립트 제출
sbatch slurm/stage2_a100.sh qwen3.5-4b

# 작업 상태 확인
squeue -u e1887a03

# 로그 확인
tail -f /scratch/e1887a03/Hybrid_fasteval/logs/stage2_a100.o*
```

---

## 🔍 트러블슈팅

### Flash Attention 여전히 안 됨

**증상:** 로그에 "The fast path is not available" 경고

**해결:**
```bash
# 1. 완전히 제거
pip uninstall -y flash-linear-attention fla-core
pip cache purge

# 2. CUDA 모듈 재로드
module purge
module load gcc/10.2.0 cuda/12.8

# 3. 재설치
pip install -U --no-build-isolation \
    git+https://github.com/fla-org/flash-linear-attention \
    --no-deps
```

### CUDA 버전 불일치

**증상:** RuntimeError: CUDA version mismatch

**해결:**
```bash
# PyTorch와 맞는 CUDA 로드
python -c "import torch; print(torch.version.cuda)"  # 예: 12.8
module load cuda/12.8
```

### A100 파티션 대기 시간 길음

**대안:**
- `amd_h100_2` 파티션 시도
- 새벽/아침 시간대에 제출
- 순차 실행으로 우선순위 높이기

---

## 📊 예상 실행 시간

| 모델 | GPU | 예상 시간 | 비고 |
|------|-----|----------|------|
| qwen3.5-4b | A100 80GB | 3-4시간 | Flash attn 사용 시 |
| nemotron-h-4b | A100 80GB | 3-4시간 | |
| olmo-hybrid-7b | A100 80GB | 5-6시간 | 더 큰 모델 |
| olmo-3-7b | A100 80GB | 5-6시간 | |

---

## 💰 비용

| 파티션 | 요율 | 6시간 비용 |
|--------|------|-----------|
| amd_a100_4 | 0.19951 | 1.20 |
| cas_v100_4 | 0.14263 | 0.86 |

**차이:** A100이 약 40% 비싸지만, OOM 없이 완료 보장

---

## ✅ 체크리스트

실행 전 확인:

- [ ] Flash Linear Attention 올바르게 설치됨
- [ ] CUDA 버전 일치 확인
- [ ] 테스트 실행 성공
- [ ] 데이터 준비 완료 (data/output/)
- [ ] 모델 캐시 완료 (models_cache/)
- [ ] 로그 디렉토리 생성 (logs/)
- [ ] 환경변수 설정 (PYTORCH_CUDA_ALLOC_CONF)

실행 중 모니터링:

- [ ] GPU 메모리 사용량 (nvidia-smi)
- [ ] 로그 파일 확인 (tail -f)
- [ ] Flash attention 작동 확인

완료 후:

- [ ] 결과 파일 확인 (results/)
- [ ] 홈 디렉토리로 백업
- [ ] GitHub에 커밋
