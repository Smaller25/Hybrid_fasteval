# 🚀 빠른 시작 가이드 (기존 conda 환경 사용)

이미 conda 환경이 설정되어 있으므로 setup_env.sh는 건너뛰고 바로 실험을 시작할 수 있습니다.

## 현재 환경 정보
- **Conda 환경**: `hybrid`
- **프로젝트 위치**: `/home01/e1887a03/Hybrid_fasteval`
- **Miniconda**: `/apps/applications/Miniconda/25.11.1`

---

## 📋 실행 순서

### 1단계: 스크래치 디렉토리 준비

```bash
# 스크래치에 작업 디렉토리 생성
mkdir -p /scratch/e1887a03/Hybrid_fasteval/{data/output,results,logs,models_cache}

# 프로젝트 파일 복사 (홈 → 스크래치)
cd /home01/e1887a03/Hybrid_fasteval
rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    ./ /scratch/e1887a03/Hybrid_fasteval/
```

### 2단계: 데이터 준비 (최초 1회)

```bash
# 데이터 준비 작업 제출
sbatch slurm/prepare_data.sh

# 작업 상태 확인
squeue -u e1887a03

# 완료 후 데이터 확인
ls -lh /scratch/e1887a03/Hybrid_fasteval/data/output/
```

### 3단계: 실험 실행

#### 옵션 A: 단일 모델 테스트 (빠른 검증)

```bash
# Stage 2 실험 (1개 모델, GPU 1개)
sbatch slurm/stage2_length_scaling.sh qwen3.5-4b

# 로그 확인
tail -f /scratch/e1887a03/Hybrid_fasteval/logs/stage2_length.o*
```

#### 옵션 B: 모든 모델 병렬 실행 ⭐ 권장

```bash
# GPU 4개로 4개 모델 동시 실행 (가장 빠름!)
sbatch slurm/run_all_models_parallel.sh

# 또는 Stage 1 실험
sbatch slurm/run_all_models_parallel.sh stage1

# 개별 모델 로그 확인
tail -f /scratch/e1887a03/Hybrid_fasteval/logs/qwen3.5-4b_gpu0.log
```

#### 옵션 C: 모든 모델 순차 실행 (안전)

```bash
# GPU 1개로 순차 실행
sbatch slurm/run_all_models_sequential.sh

# 전체 로그 확인
tail -f /scratch/e1887a03/Hybrid_fasteval/logs/all_sequential.o*
```

### 4단계: 결과 확인 및 백업

```bash
# 결과 파일 확인
ls -lh /scratch/e1887a03/Hybrid_fasteval/results/

# 결과를 홈 디렉토리로 백업 (중요!)
rsync -av /scratch/e1887a03/Hybrid_fasteval/results/ \
         /home01/e1887a03/Hybrid_fasteval/results/

# 스크래치는 15일 후 자동 삭제되므로 주기적 백업 필수!
```

---

## 🔍 작업 모니터링 명령어

```bash
# 내 작업 목록
squeue -u e1887a03

# 작업 상세 정보
scontrol show job <JOB_ID>

# 작업 취소
scancel <JOB_ID>

# 실시간 로그 확인
tail -f /scratch/e1887a03/Hybrid_fasteval/logs/*.o<JOB_ID>

# GPU 사용량 확인 (실행 중인 노드에서)
ssh <NODE_NAME>
nvidia-smi -l 1
```

---

## ⚡ 가장 빠른 시작 (3줄 요약)

```bash
# 1. 데이터 준비
sbatch slurm/prepare_data.sh

# 2. (데이터 준비 완료 후) 모든 모델 병렬 실행
sbatch slurm/run_all_models_parallel.sh

# 3. 결과 백업
rsync -av /scratch/e1887a03/Hybrid_fasteval/results/ \
         /home01/e1887a03/Hybrid_fasteval/results/
```

---

## 💡 팁

- **첫 실행**: 모델 다운로드로 30분~1시간 걸릴 수 있음 (자동 캐시됨)
- **로그 위치**: `/scratch/e1887a03/Hybrid_fasteval/logs/`
- **결과 위치**: `/scratch/e1887a03/Hybrid_fasteval/results/`
- **문제 발생 시**: `logs/*.e<JOB_ID>` 에러 로그 확인
