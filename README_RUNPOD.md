# RunPod 사용 가이드

## 🚀 빠른 시작 (재시작 후)

### 원클릭 복구 + 실험 시작
```bash
cd /workspace/Hybrid_fasteval
git pull origin main
bash quick_start.sh qwen3.5-4b 50
```

이게 끝입니다! 환경 설치 → 데이터 준비 → 실험 시작이 자동으로 진행됩니다.

---

## 📋 단계별 실행 (커스터마이징 필요 시)

### 1. 환경 설정만
```bash
bash setup_environment.sh
```

**설치되는 것**:
- PyTorch Nightly (CUDA 12.4)
- Triton 3.2.0
- Transformers, Datasets 등
- Flash-Linear-Attention

**소요 시간**: ~5-10분

### 2. 데이터 준비만
```bash
bash prepare_data.sh
```

**생성되는 파일**:
- `data/output/short_conflict.jsonl` (200 records)
- `data/output/long_conflict_500.jsonl`
- `data/output/long_conflict_2k.jsonl`
- `data/output/long_conflict_8k.jsonl`
- `data/output/long_conflict_32k.jsonl`

**소요 시간**: ~2-3분

### 3. 실험 실행
```bash
bash run_stage2_safe.sh qwen3.5-4b 50
```

**예상 시간**: ~4시간 (A100 80GB 기준)

---

## 🔍 진행 상황 확인

```bash
# 실시간 로그
tail -f logs/run.log

# GPU 사용률
watch -n 1 nvidia-smi

# 결과 파일 확인
ls -lh results/

# 특정 길이 완료 여부
cat results/stage2_qwen3.5-4b_500.json
```

---

## 🛠️ 문제 해결

### GPU 인식 안 됨
```bash
# CUDA 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# GPU 확인
nvidia-smi
```

만약 False이거나 에러면 **RunPod pod 재시작** 필요

### 실험 중단하고 싶을 때
```bash
# PID 확인
cat logs/run.pid

# 중단
kill $(cat logs/run.pid)
```

### 특정 길이만 다시 실행
```bash
python -u project/experiments/stage2_length_scaling.py \
    --models qwen3.5-4b \
    --data_dir data/output/ \
    --lengths 32k \
    --out results/stage2_qwen3.5-4b_32k_retry.json \
    --n_per_condition 50
```

---

## 📊 결과 다운로드

### 방법 1: RunPod 웹 UI
File Browser → `/workspace/Hybrid_fasteval/results/` → Download

### 방법 2: GitHub에 백업
```bash
git checkout -b experiment-results
git add results/
git commit -m "Add Stage 2 results - qwen3.5-4b"
git push origin experiment-results
```

---

## ⚙️ 환경 변수

실험 전에 자동으로 설정되지만, 수동으로 설정하려면:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=~/models_cache
export TRANSFORMERS_CACHE=~/models_cache
```

---

## 📝 주요 파일

| 파일 | 용도 |
|------|------|
| `quick_start.sh` | 원클릭 복구 + 실험 시작 |
| `setup_environment.sh` | 환경 설치 |
| `prepare_data.sh` | 데이터 생성 |
| `run_stage2_safe.sh` | 실험 실행 (길이별 저장) |
| `test_responses.py` | 모델 응답 확인 (디버깅) |
| `download_models.py` | 모델 사전 다운로드 |

---

## 💡 Tips

1. **백그라운드 실행**: `nohup`으로 실행하면 SSH 연결 끊어져도 계속 실행
2. **중간 저장**: `run_stage2_safe.sh`는 길이별로 저장하므로 32k에서 터져도 안전
3. **로그 확인**: 실험 시작 후 5-10분 뒤 로그 확인해서 정상 작동 확인
4. **모델 캐싱**: 처음 실행 시 모델 다운로드 시간 포함 (~10분 추가)

---

**문제 발생 시**: GitHub Issues 또는 로그 파일 공유
