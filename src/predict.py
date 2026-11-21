"""CLI single-image inference script with zero-retention validation and fixed disclaimer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import CLASS_META, DEFAULT_WEIGHTS_PATH, DISCLAIMER_TEXT
from src.dataset import InvalidImageFormatError
from src.inference import DopingPredictor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WADA 기준 약물 도핑 위험성 예측 CLI 추론 스크립트"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="분석할 의약품 낱알 이미지 경로 (JPG, PNG, WEBP)",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="학습된 모델 가중치 경로 (.pt)",
    )
    args = parser.parse_args()

    try:
        predictor = DopingPredictor(weights_path=Path(args.weights))
        result = predictor.predict_from_path(args.image)
    except InvalidImageFormatError as exc:
        print(f"[입력 오류] {exc}", file=sys.stderr)
        print(f"\n[면책 조항]\n{DISCLAIMER_TEXT}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"[파일 오류] {exc}", file=sys.stderr)
        print(f"\n[면책 조항]\n{DISCLAIMER_TEXT}", file=sys.stderr)
        return 1

    meta = CLASS_META[result.final_class]
    print("=" * 68)
    print(f" 최종 판정 등급 : {meta['title']}")
    if result.escalated:
        print(f" 보수적 판정    : {result.escalation_reason}")
    print(f" 모델 백본      : {result.backbone} (추론 시간: {result.elapsed_ms:.1f} ms)")
    print("-" * 68)
    print(" [클래스별 Softmax 예측 확률]")
    for cls_name, prob in result.probabilities.items():
        bar = "█" * int(round(prob * 25))
        print(f"  - {cls_name:<10s}: {prob * 100:6.2f}%  {bar}")
    print("-" * 68)
    print(f" [Grad-CAM 판단 근거]\n  {result.cam_explanation}")
    print("-" * 68)
    print(f" [맞춤형 복용 주의 가이드]\n{result.guide_text}")
    print("=" * 68)
    print(f"[면책 조항]\n{result.disclaimer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
