# KISTI Neuron 사용 가이드

## 🎯 목표

- **Flash Linear Attention 활성화** (RunPod보다 빠른 inference)
- **긴 context 실험** (64K, 128K, 256K)
- **A100 80GB 활용**

---

## 📋 사전 준비

### 1. SSH 접속

```bash
ssh e1887a03@neuron.ksc.re.kr
```

### 2. 프로젝트 업데이트

```bash
# 홈에서 최신 코드 pull
cd /home01/e1887a03/Hybrid_fasteval
LD_LIBRARY_PATH="" git pull origin main

# 작업 디렉토리로 복사
rsync -av --exclude='.git' \
    /home01/e1887a03/Hybrid_fasteval/ \
    /scratch/e1887a03/Hybrid_fasteval/

cd /scratch/e1887a03/Hybrid_fasteval
```

---

## 🔧 Interactive 세션에서 환경 설정

### A100 Interactive 세션 시작

```bash
srun -p amd_a100_4 --gres=gpu:1 --cpus-per-task=8 \
    --time=2:00:00 --comment pytorch --pty bash
```

대기가 너무 길면 V100로 먼저 테스트:
```bash
srun -p cas_v100_4 --gres=gpu:1 --cpus-per-task=8 \
    --time=2:00:00 --comment pytorch --pty bash
```

### 환경 설정 실행

```bash
cd /scratch/e1887a03/Hybrid_fasteval

# Conda 환경 활성화
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid

# Flash Linear Attention 설치
bash kisti/setup_kisti.sh
```

**예상 시간**: ~10분

**중요**:
- ✅ "✓ Forward pass successful" 나오면 성공
- ⚠️ 경고 있어도 "Setup complete" 나오면 OK

---

## 📊 데이터 준비

### 짧은 context (500-32K) - 이미 있으면 skip

```bash
bash prepare_data.sh
```

### 긴 context (64K-256K) 추가 생성

```bash
bash kisti/prepare_long_context.sh
```

**생성되는 파일**:
- `data/output/long_conflict_64k.jsonl`
- `data/output/long_conflict_128k.jsonl`
- `data/output/long_conflict_256k.jsonl`

---

## 🚀 실험 실행

### 옵션 A: Interactive 세션에서 직접 실행

```bash
# A100 세션 확보
srun -p amd_a100_4 --gres=gpu:1 --cpus-per-task=16 \
    --time=12:00:00 --comment pytorch --pty bash

cd /scratch/e1887a03/Hybrid_fasteval
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
conda activate hybrid
module load gcc/10.2.0 cuda/12.8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/e1887a03/Hybrid_fasteval/models_cache
export TRANSFORMERS_CACHE=/scratch/e1887a03/Hybrid_fasteval/models_cache

# nohup으로 백그라운드 실행
nohup bash run_stage2_safe.sh qwen3.5-4b 50 > logs/run_kisti.log 2>&1 &

# 로그 확인
tail -f logs/run_kisti.log
```

### 옵션 B: sbatch 제출

```bash
# 기본 실험 (500-32K)
sbatch slurm/stage2_a100.sh qwen3.5-4b

# 긴 context 실험 (64K-256K)
sbatch kisti/stage2_long_context.sh qwen3.5-4b

# 작업 상태 확인
squeue -u e1887a03

# 로그 확인
tail -f /scratch/e1887a03/Hybrid_fasteval/logs/stage2_*.o*
```

---

## 🔍 Flash Attention 작동 확인

### 로그에서 확인

**성공**:
```
✓ Flash Linear Attention available
```

**실패** (fallback):
```
The fast path is not available because one of the required library is not installed.
```

### 수동 테스트

```bash
python -c "
from fla.ops.gla import fused_chunk_gla
import torch

B, H, T, D = 2, 4, 1024, 64
q = torch.randn(B, H, T, D).cuda()
k = torch.randn(B, H, T, D).cuda()
v = torch.randn(B, H, T, D).cuda()
g = torch.randn(B, H, T, D).cuda().sigmoid()

o, _ = fused_chunk_gla(q, k, v, g)
print('✓ Flash attention working!')
"
```

---

## 🐛 문제 해결

### Flash Attention 여전히 안 됨

```bash
# 1. 완전 제거
pip uninstall -y flash-linear-attention fla-core
pip cache purge

# 2. CUDA module 재로드
module purge
module load gcc/10.2.0 cuda/12.8

# 3. 재설치
bash kisti/setup_kisti.sh
```

### CUDA 버전 불일치

```bash
# PyTorch CUDA 확인
python -c "import torch; print(torch.version.cuda)"

# 맞는 CUDA module 로드
module load cuda/12.8  # 또는 적절한 버전
```

### OOM (Out of Memory)

**128K, 256K에서 발생 가능**

해결:
1. `n_per_condition` 줄이기 (50 → 30)
2. Gradient checkpointing 활성화 (코드 수정 필요)
3. 더 작은 모델 사용

---

## 📊 예상 실행 시간

| Context | Flash ON | Flash OFF | GPU |
|---------|----------|-----------|-----|
| 32K | 2-3시간 | 6-8시간 | A100 |
| 64K | 4-5시간 | 12-15시간 | A100 |
| 128K | 8-10시간 | 20-24시간 | A100 |
| 256K | 16-20시간 | 40-48시간+ | A100 |

**Flash Attention 활성화 시 3-4배 빠름!**

---

## 💾 결과 백업

### Scratch → Home 백업

```bash
# 스크래치는 15일 미접근 시 삭제됨!
rsync -av /scratch/e1887a03/Hybrid_fasteval/results/ \
    /home01/e1887a03/Hybrid_fasteval/results_backup/
```

### GitHub 백업

```bash
cd /home01/e1887a03/Hybrid_fasteval
git checkout -b kisti-long-context-results

# 결과만 추가 (.gitignore 수정 필요)
git add results/stage2_*_kisti.json
git commit -m "Add KISTI long context results (64K-256K)"

LD_LIBRARY_PATH="" git push origin kisti-long-context-results
```

---

## 📝 체크리스트

**환경 설정**:
- [ ] Flash Linear Attention 설치 완료
- [ ] Forward pass 테스트 성공
- [ ] CUDA 버전 일치 확인

**데이터 준비**:
- [ ] Short conflict 데이터 생성 완료
- [ ] Long context 데이터 (64K-256K) 생성 완료

**실험 실행**:
- [ ] Interactive 세션 확보 또는 sbatch 제출
- [ ] Flash Attention 작동 확인 (로그)
- [ ] 결과 파일 생성 확인

**백업**:
- [ ] Scratch → Home 복사
- [ ] GitHub 업로드

---

## 💡 Tips

1. **Flash Attention 중요**: 없으면 128K+ 실험이 매우 느림
2. **Scratch 15일 규칙**: 주기적으로 Home으로 백업
3. **대기 시간**: A100 혼잡 시 새벽/아침 제출
4. **Git 명령어**: 항상 `LD_LIBRARY_PATH=""` 붙이기

---

**문제 발생 시**: 로그 파일 확인 (`logs/*.e<JOB_ID>`)
