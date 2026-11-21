# WADA 기준 AI 약물 도핑 위험성 예측 서비스 (`doping_classifier`)

의약품 낱알 사진 한 장으로 WADA 금지목록 기준 3단계 도핑 위험도를 즉시 판별하고, 판단 근거(Grad-CAM 히트맵)와 등급별 맞춤형 안전 복용 가이드를 제공하는 로컬 AI 비전 안심 체크 서비스입니다. (인트로 화면 없이 단일 페이지 웹 서비스로 바로 진입합니다.)

---

## 1. 핵심 기능 및 정책

- **3단계 WADA 위험도 분류**: `High Risk` (복용 금지) / `Caution` (주의 - 전문가 확인 필요) / `Safe` (안전 - 허용 의약품)
- **보수적 판정 규칙 (위음성 최소화)**: `High Risk` 예측 확률이 **35% 이상**일 경우 자동으로 `Caution` 이상 등급 경고를 활성화합니다.
- **Grad-CAM 기반 판단 근거 시각화**: CNN 백본(`EfficientNet-B0` / `ResNet50`)의 마지막 합성곱 블록(`conv_head` / `layer4[-1]`)에서 추출한 히트맵을 원본 위에 합성하고, 알약 본체·각인·분할선·윤곽 집중도를 설명합니다.
- **Zero-Retention 메모리 추론**: 사용자가 업로드한 이미지는 RAM에서만 전처리 및 추론을 수행하고 디스크에 저장하지 않으며, 임시 업로드 파일도 즉시 파기합니다.
- **예외 방어 및 고정 면책 고지**: 미입력 경고, 비정상 파일(`.txt`, `.pdf` 등) 차단, 하단 고정 면책 조항 상시 노출.

---

## 2. 디렉토리 구조

```text
doping_classifier/
├── data/
│   ├── train/              # 학습용 낱알 이미지 (High_Risk, Caution, Safe)
│   ├── val/                # 검증용 낱알 이미지
│   ├── test/               # 테스트용 낱알 이미지
│   └── samples/            # UI 테스트용 샘플 3종 이미지
├── models/
│   ├── best_model.pt       # 학습 완료된 최적 모델 가중치
│   └── metrics.json        # 검증·테스트셋 평가 지표 및 Confusion Matrix
├── src/
│   ├── config.py           # 클래스 정의, 임계치, 가이드, 면책 문구 상수
│   ├── dataset.py          # 메모리 이미지 검증, Albumentations 전처리, 데이터셋
│   ├── model.py            # timm 기반 3-Class 분류 모델 (EfficientNet-B0 / ResNet50)
│   ├── train.py            # Recall 중심 학습 및 평가 루프
│   ├── gradcam.py          # Grad-CAM 히트맵 생성 및 활성화 영역 설명
│   ├── inference.py        # Zero-Retention 공유 추론 파이프라인
│   └── predict.py          # CLI 단일 이미지 추론 스크립트
├── app.py                  # 로컬 실행용 Gradio 단일 페이지 웹 UI
├── requirements.txt        # 의존성 패키지 목록
└── README.md
```

---

## 3. 모델 성능 지표 (Test Set Evaluation)

- **백본 아키텍처**: `timm` `efficientnet_b0` (입력 규격 `224x224`)
- **평가 데이터셋**: MVP 낱알 데이터셋 (`train`: 180장, `val`: 45장, `test`: 60장 — 클래스당 20장)
  > ※ 실제 서비스 고도화 시 식약처(MFDS) 낱알 식별 공공 데이터를 `data/{train,val,test}/{High_Risk,Caution,Safe}` 구조로 배치한 뒤 `python -m src.train`으로 재학습할 수 있습니다.

| 지표 | 목표 기준 (PRD) | 측정 결과 (Test Set) |
| :--- | :---: | :---: |
| **Overall Accuracy** | ≥ 85.0% | **100.00%** |
| **High Risk Recall** | ≥ 92.0% | **100.00%** |
| **단일 이미지 추론 + Grad-CAM 속도** | ≤ 3.0초 | **~0.5초 (CPU 기준)** |

### Confusion Matrix (Test Set, N=60)

| 실제 \ 예측 | High Risk | Caution | Safe |
| :--- | :---: | :---: | :---: |
| **High Risk** | **20** | 0 | 0 |
| **Caution** | 0 | **20** | 0 |
| **Safe** | 0 | 0 | **20** |

---

## 4. 설치 및 실행 가이드

### 4.1. 가상환경 실행

```powershell
cd doping_classifier
.\.venv\Scripts\Activate.ps1
```

### 4.2. Gradio 웹 UI 실행

```powershell
.\.venv\Scripts\python.exe app.py
```

실행 후 브라우저에서 `http://127.0.0.1:7860` 으로 접속합니다.

### 4.3. CLI 단일 이미지 추론

```powershell
.\.venv\Scripts\python.exe -m src.predict --image .\data\samples\sample_high_risk.png
```

### 4.4. 모델 재학습

```powershell
.\.venv\Scripts\python.exe -m src.train --backbone efficientnet_b0 --epochs 12
```

---

## 5. 면책 조항 (Disclaimer)

> 본 결과는 AI 예측 모델의 보조 분석 결과로, 실제 WADA/KADA 도핑 규정 준수 여부를 최종 보장하지 않습니다. 실제 복용 전 반드시 KADA(한국도핑방지위원회) 의약품 검색 서비스 또는 스포츠 의학 전문의와 상의하십시오.
