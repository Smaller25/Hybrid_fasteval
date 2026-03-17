# Hybrid LLM ICL Limitation: Knowledge Conflict as a Mechanistic Probe

> **Claude Code 전달용 문서**  
> 이 README는 연구 배경, 실험 설계, 코드 구조, 참고 레퍼런스를 포함합니다.  
> 모든 실험은 2-Stage로 진행합니다.

---

## 연구 한 줄 요약

Hybrid LLM (GDN/Mamba 기반)의 ICL 약점을 Knowledge Conflict를 controlled probe로 활용해 mechanistically 분석하고, 이를 기반으로 context length scaling 실험을 수행한다.

---

## 배경 및 동기

### Hybrid 모델 ICL 성능 — 현황

최신 hybrid 모델들은 RULER 등 long-context benchmark에서 pure TF와 대등하거나 더 좋다고 보고한다:

| 모델 | RULER (128K) | MMLU-Pro | 특이사항 |
|---|---|---|---|
| Nemotron-H-8B-Instruct | ~84% | 보고 안 함 | Qwen2.5-7B에 근접 |
| Nemotron 3 Nano (30B-A3B) | 87.5%@64K, 70.56%@512K | 78.3% | Qwen3-30B의 80.9%보다 낮음 |
| OLMo Hybrid 7B | 85.0%@64K | 경쟁력 | Extended reasoning에서 손실 있음 (팀 인정) |
| Qwen3.5-4B | 보고 안 함 (GDN 3:1) | — | Small series 최초 공개 |

**선택적 보고 (selective reporting) 주의:**
- Nemotron-H paper: Transformer baseline이 이긴 7개 task 수치 미공개
- Nemotron-H / Nemotron 3 Nano: NQ, TriviaQA, Phonebook, Based benchmark 전부 미보고
- OLMo Hybrid: extended reasoning에서 "substantial losses"라고 언급했으나 수치 불명확

### Pure SSM의 확인된 약점 (중요 baseline)

NVIDIA의 8B controlled study (Waleffe et al., 2024):
- Pure Mamba/Mamba-2: 5-shot MMLU에서 Transformer 대비 ~15점 낮음
- Phonebook lookup, in-context recall: training token 늘려도 gap 유지
- **Mamba-2-Hybrid (4 attention layers)**: 동일 데이터 기준 Transformer를 12개 task 전부에서 능가

→ 즉 "hybrid가 약하다"가 아니라 **"pure SSM이 약하고, hybrid는 attention layer 덕에 보완된다"**가 현재 정확한 이해.

### 핵심 질문: 그렇다면 어떤 조건에서 약점이 드러나는가?

**Knowledge Conflict = ICL의 Hard Test**

ICL 능력에는 두 레벨이 있다:
- 쉬운 ICL: context에 없던 패턴을 따라가기 (few-shot)
- **어려운 ICL: 모델이 강하게 아는 사실과 context가 정면 충돌할 때 context를 따라가기**

Knowledge conflict는 두 번째를 테스트한다. 이게 hybrid의 약점을 드러낼 수 있는 이유:
- GDN/Mamba의 recurrent state가 parametric knowledge를 압축 저장
- 이 state가 attention의 in-context retrieval 신호를 **passive하게 decay**시킬 가능성
- Context length가 늘수록 이 효과가 누적될 수 있음

### 선행 연구와 빈틈

**이미 알려진 것 (TF 기준):**
- Attention head가 "memory head"(parametric)와 "context head"(in-context)로 분화 — Jin et al., ACL 2024
- JUICE (2025): memory/context head가 exclusive하지 않고 superposition으로 존재
- Logit lens로 layer별 prediction 추이 추적 가능 — nostalgebraist 2020

**진짜 빈틈:**
1. **Hybrid에서 memory/context head 분화가 동일하게 존재하는가?** → 아무도 안 봤다
2. **Long context에 conflict가 묻혀 있을 때 hybrid vs TF의 행동이 달라지는가?** → benchmark 자체가 없다
3. **Linear 레이어가 conflict resolution에서 active interference를 하는가, passive인가?** → 미확인

---

## 실험 설계

### 대상 모델

#### Hybrid (핵심 실험 대상)

| 모델 | HuggingFace ID | 특징 |
|---|---|---|
| **Qwen3.5-4B** | `Qwen/Qwen3.5-4B` | GDN 3:1, 262K ctx, 파일럿 1순위 |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | VRAM 부족 시 대안 |
| **Nemotron-H-4B** | `nvidia/Nemotron-H-4B-Instruct-128K` | Mamba2 dominant, attn 4개만, pruned from 8B |
| Nemotron-H-8B | `nvidia/Nemotron-H-8B-Reasoning-128K` | 8B scale |
| **OLMo Hybrid 7B** | `allenai/OLMo-Hybrid-7B` | OLMo 3와 동일 데이터 → controlled comparison 핵심 |

아키텍처 구조:
```
Qwen3.5:    [GDN][GDN][GDN][ATTN][GDN][GDN][GDN][ATTN]...  (75% linear, 25% attn)
Nemotron-H: [M2][M2]...[ATTN]...[M2][M2]...[ATTN]...       (92% linear, 8% attn = 4 layers total)
OLMo Hybrid:[GDN][GDN][GDN][ATTN][GDN][GDN][GDN][ATTN]...  (3:1, SWA 교체)
```

#### Full TF Baseline

| 모델 | HuggingFace ID | 역할 |
|---|---|---|
| **OLMo 3 7B** | `allenai/OLMo-3-7B` | OLMo Hybrid와 paired (동일 데이터), 가장 중요한 controlled baseline |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B` | 사이즈 matched baseline |
| Gemma 3 4B | `google/gemma-3-4b-it` | 사이즈 matched baseline |

---

### Stage 1: Short-Context Mechanistic 분석

**목적**: signal 확인 + linear 레이어의 conflict resolution 역할 파악

#### 1-1. 데이터 구성

**CounterFact** (공개 데이터):
```python
from datasets import load_dataset
cf = load_dataset("NeelNanda/counterfact-tracing", split="train")
# 필드: requested_rewrite.subject, target_true.str, target_new.str, prompt
```

**ConflictBank** (NeurIPS 2024):
```python
# https://github.com/zhaochen0110/conflictbank
# 7.45M claim-evidence pairs, 553K QA pairs
# 세 타입: misinformation / temporal discrepancy / semantic divergence
```

**자체 구성** (100~200개, TriviaQA 기반):
```python
def make_conflict_prompt(subject, true_obj, false_obj):
    context = f"According to recent records, {subject} is associated with {false_obj}."
    question = f"Question: What is {subject} associated with?\nAnswer:"
    return {
        "prompt_with_context": f"Context: {context}\n\n{question}",
        "prompt_no_context": question,
        "label_parametric": true_obj,   # 모델의 실제 알고 있는 답
        "label_incontext": false_obj,   # context의 counterfactual 답
    }
```

#### 1-2. Behavioral Baseline

```python
def score_response(response: str, label_parametric: str, label_incontext: str) -> str:
    resp = response.lower().strip()
    if label_incontext.lower() in resp:
        return "incontext"
    elif label_parametric.lower() in resp:
        return "parametric"
    return "other"

# 측정 지표
# context_following_rate = incontext / (incontext + parametric)
# logit_gap = logit(incontext_token) - logit(parametric_token)
```

#### 1-3. Logit Lens (Layer별 Prediction 추이)

```python
def logit_lens_analysis(model, tokenizer, prompt, label_a, label_b):
    """
    각 레이어의 hidden state를 vocabulary로 project해서
    layer를 따라 prediction이 어떻게 변하는지 추적.
    flip point (parametric↔incontext)가 어느 레이어 타입인지 확인.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # (num_layers+1, batch, seq, hidden)
    lm_head = model.lm_head
    norm = getattr(model.model, 'norm', None)
    
    tok_a = tokenizer.encode(" " + label_a, add_special_tokens=False)[0]
    tok_b = tokenizer.encode(" " + label_b, add_special_tokens=False)[0]
    
    layer_probs = []
    for h in hidden_states:
        h_last = h[0, -1, :]
        if norm is not None:
            h_last = norm(h_last.unsqueeze(0)).squeeze(0)
        logits = lm_head(h_last)
        probs = torch.softmax(logits, dim=-1)
        layer_probs.append({
            "p_parametric": probs[tok_a].item(),
            "p_incontext": probs[tok_b].item(),
        })
    return layer_probs
```

**분석 포인트**: flip point가 attention 레이어 직후에 발생하는가 vs. linear 레이어 직후에 발생하는가.

#### 1-4. Activation Patching (Causal Mediation)

Skip보다 더 정밀한 인과 분석. "clean" run (conflict 없음)의 hidden state를 "corrupted" run (conflict 있음)에 patch했을 때 prediction이 변하는지 측정.

```python
def activation_patch(model, clean_prompt, corrupted_prompt, 
                     tokenizer, label_parametric, label_incontext,
                     layer_idx: int, layer_type: str):
    """
    clean: no conflict context
    corrupted: conflict context (counterfactual)
    
    corrupted run에서 특정 레이어의 hidden state를 
    clean run의 것으로 교체했을 때 prediction 변화를 측정.
    
    layer_type: "attention", "linear", "ffn"
    """
    # clean run에서 hidden state 저장
    clean_hs = {}
    def save_hook(module, input, output, name):
        clean_hs[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
    
    # patching 후 corrupted run에서 prediction 변화
    # → linear 레이어 patch가 prediction을 parametric으로 이동시키면:
    #   linear 레이어가 parametric knowledge 유지에 기여한다는 증거
```

---

### Stage 2: Context Length Scaling

**목적**: conflict가 long context에 묻혀 있을 때 hybrid vs TF 행동 차이 관찰  
**→ 이 자체 benchmark이 연구의 contribution**

#### 2-1. 데이터 구성 (자체 생성)

동일 conflict를 점점 긴 distractor context에 묻는 방식:

```python
def make_long_conflict_prompt(
    subject, true_obj, false_obj,
    distractor_text: str,    # 관련 없는 filler text
    conflict_position: str,  # "beginning", "middle", "end"
    total_length: int        # 목표 토큰 수
):
    conflict_sentence = f"According to recent records, {subject} is associated with {false_obj}."
    
    if conflict_position == "beginning":
        return f"Context: {conflict_sentence}\n\n{distractor_text}\n\nQ: {question}"
    elif conflict_position == "middle":
        half = len(distractor_text) // 2
        return f"Context: {distractor_text[:half]}\n{conflict_sentence}\n{distractor_text[half:]}\n\nQ: {question}"
    elif conflict_position == "end":
        return f"Context: {distractor_text}\n\n{conflict_sentence}\n\nQ: {question}"
```

**길이 설정**: 500 → 2K → 8K → 32K 토큰  
**Conflict position**: beginning, middle, end (3가지) × 4가지 길이 = 12 조건  
**반복**: 각 100 케이스 → 총 1200 케이스

#### 2-2. 측정

```python
# 각 조건에서 측정:
# 1. context_following_rate
# 2. logit_gap (확신도)
# 3. (Stage 1에서 찾은 key layers의) hidden state 변화

# 핵심 가설:
# Hybrid: context length 늘수록 context_following_rate 하락 속도가 TF보다 빠름
# TF: 상대적으로 유지
```

---

## 예상 결과 및 판단 기준

| 가설 | 지지 패턴 | 반박 패턴 | 의미 |
|---|---|---|---|
| Linear 레이어 = parametric bias amplifier | Conflict에서 linear hidden state에 parametric token representation 증폭 | 차이 없음 | ICL limitation의 mechanistic 설명 |
| Context length↑ → Hybrid가 더 많이 parametric으로 회귀 | Short gap ≈ 0, long gap 확대 | Short부터 gap 있거나 없음 유지 | Linear state의 누적 decay 효과 |
| Attention skip → context following 폭락 | 양 모델 모두 즉각 폭락 | Hybrid에서 덜 폭락 | Attention의 in-context retrieval 전담 확인 |
| OLMo Hybrid vs OLMo 3 behavioral gap 있음 | Short conflict에서도 차이 | 동일 행동 | Architecture 차이가 conflict resolution에 영향 |

**연구가 보여줄 수 있는 것들:**
1. (positive) Linear 레이어가 conflict에서 active interference → ICL limitation 설명 가능
2. (positive) Long context에서만 gap 나타남 → "언제 hybrid가 약한가"의 조건 특정
3. (negative도 OK) Short-context에서 gap이 없음 → "RULER에서 대등한 이유"를 설명, long context 실험에서 gap 찾기

---

## 디렉토리 구조

```
project/
├── RESEARCH_README.md          ← 이 파일
├── data/
│   ├── prepare_counterfact.py  # CounterFact 로드 및 conflict prompt 생성
│   ├── prepare_conflictbank.py # ConflictBank 처리
│   ├── prepare_custom.py       # TriviaQA 기반 자체 데이터 100개 생성
│   ├── make_long_context.py    # Stage 2: distractor 추가, 길이별 생성
│   └── output/
│       ├── short_conflict.jsonl
│       └── long_conflict_{500,2k,8k,32k}.jsonl
│
├── models/
│   ├── load_model.py           # 모델별 로드 유틸 (trust_remote_code, dtype 등)
│   └── layer_utils.py          # 레이어 타입 매핑 (attention vs linear vs ffn)
│
├── experiments/
│   ├── stage1_baseline.py      # Behavioral baseline 측정
│   ├── stage1_logit_lens.py    # Layer-by-layer prediction 추이
│   ├── stage1_patching.py      # Activation patching (causal mediation)
│   └── stage2_length_scaling.py # Context length별 성능 측정
│
├── analysis/
│   ├── plot_logit_lens.py      # Layer별 p(parametric) vs p(incontext) 시각화
│   ├── plot_length_scaling.py  # Context length vs context_following_rate
│   └── flip_point_stats.py     # Flip point가 attn vs linear 레이어인지 통계
│
├── results/                    # 실험 결과 JSON, CSV
└── requirements.txt
```

---

## 환경 설정

```bash
pip install transformers accelerate datasets torch
pip install transformer_lens   # logit lens용
pip install matplotlib seaborn pandas tqdm
pip install mamba-ssm causal-conv1d  # Nemotron-H용 (CUDA 필수)
pip install lm-eval             # 외부 벤치마크용 (Stage 2 이후)
```

---

## 모델 로드

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen3.5-4B
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

# Nemotron-H-4B (trust_remote_code 필요)
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-H-4B-Instruct-128K",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# OLMo Hybrid 7B
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-Hybrid-7B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# OLMo 3 7B (controlled TF baseline)
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-3-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

---

## 레이어 타입 매핑

각 모델의 레이어 타입을 정확히 파악하는 게 중요하다. 아래 코드로 먼저 확인할 것:

```python
def inspect_layer_types(model):
    """모델의 레이어 구조 출력"""
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if any(k in module_type.lower() for k in ['attention', 'mamba', 'gdn', 'deltanet', 'ssm']):
            print(f"{name}: {module_type}")

# Qwen3.5-4B 예상 구조 (확인 필요)
# model.model.layers[i].self_attn  → full attention (every 4th layer)
# model.model.layers[i].gdn        → Gated DeltaNet

# Nemotron-H-4B 예상 구조
# model.model.layers[i].self_attn  → 4개 attention layer (위치 확인 필요)
# model.model.layers[i].mamba      → Mamba-2 layers

# OLMo Hybrid
# 레이어 구조 model.config에서 확인
# model.config.layer_types 또는 inspect로 확인

LAYER_TYPE_MAP = {
    # 실제 실행 후 채울 것
    # "qwen3.5-4b": [...],  # "attn" or "linear" 리스트, 레이어 순서대로
    # "nemotron-h-4b": [...],
    # "olmo-hybrid-7b": [...],
}
```

---

## 핵심 참고 레퍼런스

### 모델 페이지

| 모델 | HuggingFace | 주의사항 |
|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | Thinking mode 기본 off (small series), `enable_thinking=true` 옵션 있음 |
| Qwen3.5-4B-Base | https://huggingface.co/Qwen/Qwen3.5-4B-Base | Base model (연구용 권장) |
| Nemotron-H-4B-Instruct | https://huggingface.co/nvidia/Nemotron-H-4B-Instruct-128K | `trust_remote_code=True` 필수 |
| Nemotron-H-4B-Base | https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K | 8K ctx base |
| Nemotron-H-8B-Base | https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K | |
| OLMo Hybrid 7B | https://huggingface.co/allenai/OLMo-Hybrid-7B | |
| OLMo 3 7B | https://huggingface.co/allenai/OLMo-3-7B | controlled TF baseline |
| Llama 3.2 3B | https://huggingface.co/meta-llama/Llama-3.2-3B | |
| Gemma 3 4B | https://huggingface.co/google/gemma-3-4b-it | |

### 핵심 논문

**Architecture:**
- Nemotron-H: https://arxiv.org/abs/2504.03624
- OLMo Hybrid: https://allenai.org/papers/olmo-hybrid
- OLMo 3: https://arxiv.org/abs/2512.13961
- Gated Delta Networks (GDN): Yang et al. 2025 (Qwen3.5, OLMo Hybrid 기반)
- Mamba-2 Hybrid empirical study: https://arxiv.org/abs/2406.07887 (NVIDIA, Waleffe et al.)
  → Pure SSM vs Hybrid vs TF의 가장 controlled 비교. **필독**

**Knowledge Conflict:**
- Jin et al. ACL 2024 (PH3) - memory head / context head: https://arxiv.org/abs/2402.18154
  → **이 연구의 가장 직접적인 선행 연구. 코드**: https://github.com/jinzhuoran/MConflict/
- JUICE 2025 (CP superposition): https://arxiv.org/abs/2503.10996
  → Jin et al. 반박, head가 superposition으로 memory+context 동시 encode
- Knowledge Conflicts Survey EMNLP 2024: https://arxiv.org/abs/2403.08319
- KCR (long-context conflict 2025): https://arxiv.org/abs/2508.01273

**Benchmark:**
- RULER: https://arxiv.org/abs/2404.06654
- CounterFact: https://rome.baulab.info/ (ROME 논문 부록)
  - HuggingFace: `NeelNanda/counterfact-tracing`
- ConflictBank NeurIPS 2024: https://arxiv.org/abs/2408.12076
  - GitHub: https://github.com/zhaochen0110/conflictbank
- MemoTrap: https://huggingface.co/datasets/google/IFEval (일부 포함)
- Based benchmark (recall): Arora et al. 2024, https://arxiv.org/abs/2402.18510

**Interpretability:**
- Logit Lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- TransformerLens (tuned lens 포함): https://github.com/TransformerLensOrg/TransformerLens
- ROME (parametric knowledge 위치): https://arxiv.org/abs/2202.05262
- Activation Patching best practices: https://arxiv.org/abs/2309.16042

### 코드 참고

```python
# TransformerLens 설치 및 사용
pip install transformer_lens

# 사용 예시 (logit lens)
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")
_, cache = model.run_with_cache(tokens)
# cache["blocks.5.hook_resid_post"] → layer 5 이후 residual stream

# 주의: Qwen3.5, Nemotron-H는 TransformerLens 미지원 가능성 있음
# → 직접 hook 구현 필요 (아래 참고)

# 직접 hook으로 hidden state 추출
hidden_states_by_layer = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        hidden_states_by_layer[layer_idx] = hs.detach()
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    h = layer.register_forward_hook(make_hook(i))
    hooks.append(h)

with torch.no_grad():
    model(**inputs)

for h in hooks:
    h.remove()
```

---

## 실험 실행 순서

### Day 1-2: 환경 설정 + 데이터 구성
```bash
# 1. 모델 로드 테스트 (각 모델이 올라가는지 확인)
python models/load_model.py --model qwen3.5-4b

# 2. 레이어 타입 확인
python models/layer_utils.py --model qwen3.5-4b

# 3. 데이터 생성
python data/prepare_counterfact.py --n 200 --out data/output/short_conflict.jsonl
python data/prepare_custom.py --n 100 --out data/output/custom_conflict.jsonl
```

### Day 3-4: Stage 1 Behavioral Baseline
```bash
python experiments/stage1_baseline.py \
    --model qwen3.5-4b \
    --data data/output/short_conflict.jsonl \
    --out results/baseline_qwen3.5-4b.json

# 판단: context_following_rate 차이가 모델 간에 있는가?
# → 차이 없으면 데이터 조정 (더 강한 parametric knowledge 필요)
```

### Day 5-7: Stage 1 Logit Lens
```bash
python experiments/stage1_logit_lens.py \
    --model qwen3.5-4b \
    --data data/output/short_conflict.jsonl \
    --n_samples 50 \
    --out results/logit_lens_qwen3.5-4b.json

python analysis/plot_logit_lens.py \
    --input results/logit_lens_qwen3.5-4b.json \
    --layer_types models/layer_type_maps/qwen3.5-4b.json
```

### Day 8-10: Stage 1 Activation Patching + OLMo pair
```bash
# OLMo pair 투입
python experiments/stage1_baseline.py --model olmo-hybrid-7b ...
python experiments/stage1_baseline.py --model olmo-3-7b ...

# Patching
python experiments/stage1_patching.py \
    --model qwen3.5-4b \
    --layer_types models/layer_type_maps/qwen3.5-4b.json \
    --patch_target linear  # "attention", "linear", "ffn" 순으로
```

### Day 11+: Stage 2 Context Length Scaling
```bash
python data/make_long_context.py \
    --conflict_data data/output/short_conflict.jsonl \
    --lengths 500 2000 8000 32000 \
    --positions beginning middle end \
    --out data/output/

python experiments/stage2_length_scaling.py \
    --models qwen3.5-4b nemotron-h-4b olmo-hybrid-7b olmo-3-7b llama-3.2-3b \
    --data_dir data/output/ \
    --out results/length_scaling.json
```

---

## 주의사항

### 모델별 구현 이슈

1. **Nemotron-H-4B**: `trust_remote_code=True` 필수. Mamba 레이어가 일반 TF와 다른 output 형태 반환 → hook에서 output 구조 확인 필요
2. **OLMo Hybrid**: Gated DeltaNet 레이어의 hidden state shape 확인 필요. output이 tuple일 수 있음
3. **Qwen3.5 thinking mode**: Small series(4B)는 thinking 기본 off. 일관성을 위해 thinking off로 실험
4. **TransformerLens 호환**: Qwen3.5, Nemotron-H는 TransformerLens 미지원 가능성 높음 → 직접 hook 구현 사용
5. **logit lens 구현**: 모델마다 final norm의 위치가 다름 (`model.model.norm` vs `model.norm` 등) → 직접 확인 필요

### 실험 유효성 체크

```python
# Sanity check: 모델이 실제로 conflict fact를 parametric으로 알고 있는지 확인
def verify_parametric_knowledge(model, tokenizer, subject, true_obj):
    """conflict 실험 전, 모델이 true_obj를 실제로 알고 있는지 확인"""
    prompt = f"Q: What is {subject} associated with?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return true_obj.lower() in response.lower()

# 이 확인을 통과한 케이스만 conflict 실험에 사용!
# 모델이 모르는 사실로 conflict를 만들면 signal이 없음
```

---

## 현재 날짜 기준 최신 현황 (2026-03-17)

- Qwen3.5 Small series (0.8B~9B): 2026-03-02 출시, 아직 community evaluation 부족
- OLMo Hybrid: 2026-03 출시, 아직 third-party 재현 실험 없음
- Nemotron 3 Super (120B): 2026-03 출시 (파일럿 대상 아님)
- KCR (long-context conflict): 2025-08, 아직 미정착 benchmark

---

*Last updated: 2026-03-17*
