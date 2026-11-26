# Qwen3-8B + Attention Tracker 실험

이 프로젝트는 Qwen3-8B에서 Attention Tracker의 Important Head를 새로 찾고, **Deepset test** 및 **BIPIA test**에서 프롬프트 인젝션 탐지 성능을 검증합니다. 원 논문의 탐지 수식 및 공개 코드에 기반하되, 새 모델과 BIPIA 형식에 맞게 직접 구현한 **후속 실험 코드**입니다.

## 1. 실행 순서

```
run_experiment.py
    ↓
QwenAttentionModel            모델 / 토크나이저 / attention hook 초기화
    ↓
SyntheticHeadDataset          벤치마크와 무관한 정상-공격 30쌍 준비
    ↓
AttentionTracker.fit_heads    Important Head 선정 (기본 k=4)
    ↓
DeepsetDataset / BIPIADataset 각각 로드
    ↓
ExperimentRunner             Focus Score 기록 및 AUROC 계산
    ↓
results/                     지표 JSON과 개별 샘플 결과 JSONL
```

## 2. 설치

CUDA GPU가 있는 Linux/Colab 환경을 권장합니다. PyTorch는 해당 GPU에 맞는 CUDA 버전으로 미리 설치합니다.

```bash
pip install -r requirements.txt
mkdir -p external
git clone https://github.com/microsoft/BIPIA.git external/BIPIA
pip install -e external/BIPIA
```

## 3. 실행

```bash
# 기본: Deepset test 전체 + BIPIA email/table/code 소규모 표본
python run_experiment.py --benchmark both \
  --bipia-repo external/BIPIA

# Deepset과 BIPIA 개별 샘플의 원본 모델 응답까지 생성 (ASR 자동 판정은 하지 않음)
python run_experiment.py --benchmark both \
  --bipia-repo external/BIPIA --generate-responses

# Head 선택 결과를 새로 구해야 할 때
python run_experiment.py --k 3 --force-refit

# 저장된 토큰 수로 그래프 생성
python make_plot.py
```

## 4. 주요 파라미터

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--benchmark` | both | deepset/bipia/both |
| `--model-id` | Qwen/Qwen3-8B | 모델 식별자 |
| `--k` | 4 | Head 선정 시 표준편차 보정 강도 |
| `--max-input-tokens` | 1024 | 입력 길이 상한. 넘어가면 **기록 후 제외**, 자르지 않음 |
| `--bipia-tasks` | email,table,code | BIPIA 태스크 |
| `--bipia-max-contexts` | 3 | 태스크별 원본 컨텍스트 수; 0=전체 |
| `--bipia-max-variants-per-attack` | 2 | 공격 유형별 샘플 수; 0=전체 |
| `--bipia-max-pairs` | 40 | 태스크별 정상/공격 짝 수; 0=전체 |
| `--generate-responses` | 꺼짐 | Deepset과 BIPIA의 원본 모델 응답 생성 옵션 (`run_probedelta.py`에도 적용) |

BIPIA의 WebQA(qa), Summarization(abstract)은 저작권 문제 때문에 데이터 생성이 추가로 필요합니다. 경로 및 생성 방법은 공식 `external/BIPIA/benchmark/README.md`를 참고하세요.

## 5. 데이터와 지표

Deepset에서는 Focus Score로 정상/공격을 구별하는 **AUROC**를 계산합니다. 낮은 Focus Score가 공격에 가까우므로 AUROC에는 음수 Focus Score를 사용합니다. 원 논문과 마찬가지로 시스템 instruction을 `Say xxxxxx`로 고정하며, 원본 test 데이터를 사용합니다.

BIPIA에서는 공식 `AutoPIABuilder`가 삽입한 컨텍스트를 유지하고, 동일한 소스 컨텍스트에 삽입 전/후 버전을 생성해 정상/공격 Focus Score와 AUROC를 평가합니다. 공식 프롬프트 구조대로 정상 또는 공격 컨텍스트를 시스템 메시지에, 질문을 사용자 메시지에 둡니다. Focus Score는 시스템 메시지에서 컨텍스트 앞부분과 원래 사용자 메시지 전체를 신뢰할 수 있는 지시사항으로, 외부 컨텍스트를 데이터로 취급합니다. code 태스크의 사용자 메시지에 포함된 코드와 오류 내용도 이 가정에 포함됩니다. ProbeDelta가 사용자 메시지 뒤에 추가하는 probe는 데이터 범위에 넣습니다. Attention Tracker의 범위 해석을 확장한 구현이므로 다른 구현과 결과를 비교할 때 이 차이를 고려해야 합니다.

**주의:** BIPIA의 AUROC는 공격 문자열 포함 여부에 대한 점수 순위 지표이며 공식 ASR(공격 성공률)이 아닙니다. 실제 ASR은 모델의 응답을 추가로 판정해야 합니다. `--generate-responses`는 Deepset과 BIPIA의 원본 응답만 저장하며 자동 성공 판정은 하지 않습니다.

## 6. 결과 위치와 읽는 방법

- `results/tracker_qwen3_8b.json`: Important Head 좌표와 k. 기존 체크포인트의 threshold 값은 무시
- `results/deepset_metrics.json`: Deepset 성능 지표
- `results/deepset_predictions.jsonl`: 개별 정상/공격 프롬프트의 Focus Score, `formatted_text`, 선택적 `baseline_response`
- `results/bipia_email_metrics.json` 등: 태스크별 성능 지표
- `results/bipia_email_predictions.jsonl` 등: Focus Score, `formatted_text`, 공격 유형, 위치, 원본 데이터, 선택적 원본 모델 응답

`formatted_text`는 system/user 역할 구분과 assistant 생성 시작 부분까지 포함한, 토큰화 전 채팅 템플릿 입력 문자열입니다.
`instruction_token_count`는 Focus Score에서 지시문으로 취급한 범위의 토큰 수이고, `total_token_count`는 assistant 생성 시작 부분을 포함한 원본 입력 전체의 토큰 수입니다. ProbeDelta도 probe를 추가하기 전 원본 입력을 셉니다.

모든 예측 행은 정상·공격 여부와 관계없이 `attack_success: null`로 저장합니다. 나중에 판정할 미판정 필드이며, 기존 AUROC는 `attack_success`가 아닌 `label`을 사용합니다.

`make_plot.py`는 저장된 토큰 수를 직접 사용하므로 토크나이저나 head 체크포인트가 필요하지 않습니다. 기존 예측 파일에 두 필드가 없으면 실험을 다시 실행한 후 그래프를 생성해야 합니다.
- `results/bipia_email_skipped.jsonl` 등: 길이가 길어 평가에서 제외된 샘플
- `results/summary.json`: 벤치마크별 지표 종합

논문 비교 시 반드시 **중요한 head의 개수, 평가 샘플 개수, 입력 길이로 인한 제외 샘플 수, BIPIA task별 결과**를 함께 보고하세요. 이전 threshold 기반 결과와 구분되는 출력 디렉터리를 사용하세요.

## 7. 알려진 제한

- 실제 Qwen3-8B 가중치를 로딩해 실행하는 GPU 실험은 이 코드 작성 환경에서 수행하지 못했습니다. 모델 및 벤치마크 다운로드가 필요합니다.
- Attention Tracker 공개 구현의 `normalize_sum` 수식과 head 선택 수식에 기반하지만, Qwen3/새 데이터 형식을 대상으로 한 확장입니다. 논문 수치의 완전 재현을 의미하지 않습니다.
- 기본 BIPIA 실행은 **작은 표본**입니다. 전체 벤치마크로 평가하려면 세 가지 `--bipia-max-*` 값을 0으로 바꾸고 GPU 메모리/입력 길이 상한을 조정하세요.
- 8B 모델의 attention 행렬을 계산하므로 충분한 VRAM이 필요합니다. 24GB 이상 권장하며, 긴 BIPIA 문서는 더 많은 메모리를 요구할 수 있습니다.
