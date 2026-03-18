# SLURM 스크립트 사용 가이드

KISTI Neuron 시스템에서 Hybrid LLM 실험을 실행하기 위한 SLURM 배치 스크립트 모음입니다.

## 📁 디렉토리 구조

```
/home01/$USER/Hybrid_fasteval/     # 홈 디렉토리 (코드 원본)
/scratch/$USER/Hybrid_fasteval/    # 스크래치 (실제 작업)
  ├── data/                        # 데이터셋
  ├── results/                     # 실험 결과
  ├── logs/                        # 작업 로그
  ├── models_cache/                # HuggingFace 모델 캐시
  └── venv/                        # Python 가상환경
```

## 🚀 사용 순서

### 1️⃣ 초기 설정 (최초 1회만)

```bash
# KISTI 로그인 노드에서
cd /home01/$USER
git clone https://github.com/Smaller25/Hybrid_fasteval.git
cd Hybrid_fasteval

# logs 디렉토리 생성
mkdir -p logs

# 환경 설정 작업 제출
sbatch slurm/setup_env.sh

# 작업 상태 확인
squeue -u $USER
```

### 2️⃣ 데이터 준비 (필요시)

```bash
# 데이터 준비 스크립트 실행
cd /scratch/$USER/Hybrid_fasteval
source venv/bin/activate
python project/data/prepare_counterfact.py
python project/data/make_long_context.py
```

### 3️⃣ 실험 실행

#### 옵션 A: 단일 모델 테스트 (빠른 검증)

```bash
# Stage 1 실험
sbatch slurm/stage1_baseline.sh qwen3.5-4b

# Stage 2 실험
sbatch slurm/stage2_length_scaling.sh qwen3.5-4b
```

#### 옵션 B: 모든 모델 병렬 실행 (⭐ 권장 - 가장 빠름)

```bash
# GPU 4개를 동시 사용하여 시간 절약
# Stage 2 실험 (기본값)
sbatch slurm/run_all_models_parallel.sh

# Stage 1 실험
sbatch slurm/run_all_models_parallel.sh stage1
```

**장점:**
- 4개 모델을 동시에 실행하여 4배 빠름
- GPU 4개를 모두 활용하여 비용 효율적

**단점:**
- 메모리 관리가 중요 (각 GPU가 독립적)

#### 옵션 C: 모든 모델 순차 실행 (안전)

```bash
# GPU 1개로 안전하게 순차 실행
# Stage 2 실험 (기본값)
sbatch slurm/run_all_models_sequential.sh

# Stage 1 실험
sbatch slurm/run_all_models_sequential.sh stage1
```

**장점:**
- 안정적이고 메모리 이슈 없음
- GPU 1개만 사용하여 비용 절감

**단점:**
- 시간이 오래 걸림 (모델 개수 × 실험 시간)

## 📊 작업 모니터링

```bash
# 작업 상태 확인
squeue -u $USER

# 작업 상세 정보
scontrol show job <JOB_ID>

# 로그 실시간 확인
tail -f logs/all_parallel.o<JOB_ID>
tail -f logs/qwen3.5-4b_gpu0.log

# 작업 취소
scancel <JOB_ID>

# GPU 사용량 확인 (실행 중인 노드에서)
ssh <NODE_NAME>
nvidia-smi
```

## 📂 결과 확인

```bash
# 결과 파일 위치
ls -lh /scratch/$USER/Hybrid_fasteval/results/

# 결과를 홈 디렉토리로 백업
rsync -av /scratch/$USER/Hybrid_fasteval/results/ \
         /home01/$USER/Hybrid_fasteval/results/
```

## ⚙️ SLURM 옵션 설명

| 옵션 | 의미 | 값 |
|------|------|-----|
| `-J` | 작업 이름 | hybrid_* |
| `-p` | 파티션 | cas_v100_4 |
| `--nodes` | 노드 개수 | 1 |
| `--gres=gpu:N` | GPU 개수 | 1 또는 4 |
| `--cpus-per-task` | CPU 코어 수 | 5-8 |
| `--time` | 최대 실행 시간 | HH:MM:SS |
| `-o` | 표준 출력 로그 | logs/*.o%j |
| `-e` | 에러 로그 | logs/*.e%j |

## 🔧 트러블슈팅

### CUDA 버전 문제
```bash
# 사용 가능한 CUDA 버전 확인
module avail cuda

# setup_env.sh에서 적절한 버전 선택
module load cuda/11.8  # 또는 cuda/12.1 등
```

### 메모리 부족 (OOM)
```bash
# 배치 크기 줄이기 (코드 수정 필요)
# 또는 더 작은 모델로 테스트
sbatch slurm/stage2_length_scaling.sh qwen3.5-2b
```

### 데이터 자동 삭제 방지
```bash
# 스크래치는 15일 미접근 시 삭제됨
# 주기적으로 파일 접근
touch /scratch/$USER/Hybrid_fasteval/results/*

# 중요한 결과는 홈 디렉토리로 백업!
```

## 💰 비용 예상

| 실행 방식 | GPU 사용 | 예상 시간 | 비용 (요율 × 시간) |
|-----------|----------|-----------|-------------------|
| 단일 모델 | 1개 | ~2시간 | 0.14263 × 2 |
| 병렬 4모델 | 4개 | ~3시간 | 0.14263 × 3 |
| 순차 6모델 | 1개 | ~12시간 | 0.14263 × 12 |

**권장:** 병렬 실행이 가장 효율적!

## 📝 추가 팁

1. **테스트는 짧게:** 먼저 `--time 00:30:00`로 짧게 테스트
2. **로그 확인:** 실행 중간에 로그 확인으로 문제 조기 발견
3. **결과 백업:** 스크래치는 자동 삭제되므로 주기적 백업 필수
4. **모델 캐시:** 첫 실행 시 모델 다운로드로 시간 소요 (약 30분)

## 🆘 문제 발생 시

1. 로그 파일 확인: `logs/*.e<JOB_ID>`
2. CUDA 메모리: `nvidia-smi` 확인
3. 환경 재설정: `sbatch slurm/setup_env.sh` 재실행
