"""Zero-retention in-memory inference pipeline shared by Gradio UI and CLI."""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config import (
    CLASS_META,
    CLASS_NAMES,
    DEFAULT_WEIGHTS_PATH,
    DISCLAIMER_TEXT,
    HIGH_RISK_CONSERVATIVE_THRESHOLD,
    MAPPED_PILL_CSV_PATH,
    resolve_wada_alternative_entry,
)
from src.dataset import get_eval_transforms, numpy_to_tensor, validate_and_decode_bytes
from src.gradcam import generate_gradcam_overlay
from src.model import load_trained_model

_ITEM_SEQ_CACHE: Optional[Dict[str, Tuple[str, str, str]]] = None
_ITEM_SEQ_FILENAME_RE = re.compile(r"^(\d{6,12})(?:_|\.)")


def _load_mapped_item_index() -> Dict[str, Tuple[str, str, str]]:
    """Lazily load ITEM_SEQ -> (item_name, wada_codes, risk_level) from MFDS mapping CSV if present."""
    global _ITEM_SEQ_CACHE
    if _ITEM_SEQ_CACHE is not None:
        return _ITEM_SEQ_CACHE

    index: Dict[str, Tuple[str, str, str]] = {}
    if MAPPED_PILL_CSV_PATH.exists():
        try:
            with MAPPED_PILL_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    item_seq = (row.get("ITEM_SEQ") or "").strip()
                    wada_codes = (row.get("WADA_CODES") or "").strip()
                    risk_level = (row.get("RISK_LEVEL") or "").strip()
                    item_name = (row.get("ITEM_NAME") or "").strip()
                    if item_seq and wada_codes and risk_level in ("High Risk", "Caution"):
                        index[item_seq] = (item_name, wada_codes, risk_level)
        except OSError:
            index = {}

    _ITEM_SEQ_CACHE = index
    return _ITEM_SEQ_CACHE


def _lookup_wada_code_from_filename(filename: Optional[str]) -> Tuple[Optional[str], str]:
    """Identify WADA code and MFDS item name if the filename carries a mapped ITEM_SEQ prefix."""
    if not filename:
        return None, ""
    match = _ITEM_SEQ_FILENAME_RE.match(Path(filename).name)
    if not match:
        return None, ""
    item_seq = match.group(1)
    entry = _load_mapped_item_index().get(item_seq)
    if not entry:
        return None, ""
    item_name, wada_codes, _ = entry
    primary_code = wada_codes.split(",")[0].strip()
    return (primary_code or None), item_name


@dataclass
class PredictionResult:
    raw_class: str
    final_class: str
    escalated: bool
    escalation_reason: str
    probabilities: Dict[str, float]
    heatmap_image: np.ndarray
    cam_explanation: str
    guide_text: str
    wada_ref: str
    disclaimer: str
    elapsed_ms: float
    backbone: str
    is_trained_weights: bool
    wada_code: str = ""
    wada_category_name: str = ""
    prohibition_scope: str = ""
    recommendation_headline: str = ""
    alternatives_list: List[str] = field(default_factory=list)
    tue_notice: str = ""
    is_estimated_group: bool = True
    prescription_only: bool = False
    matched_item_name: str = ""


def apply_conservative_rule(probabilities: Dict[str, float]) -> tuple[str, str, bool, str]:
    """Escalate Safe -> Caution if High Risk similarity score >= 35% to minimize false negatives."""
    raw_class = max(probabilities, key=probabilities.get)
    high_risk_prob = probabilities["High Risk"]

    if raw_class == "Safe" and high_risk_prob >= HIGH_RISK_CONSERVATIVE_THRESHOLD:
        reason = (
            f"보수적 판정 규칙 활성화: High Risk 외형 유사도 점수({high_risk_prob * 100:.1f}%)가 "
            f"기준치({HIGH_RISK_CONSERVATIVE_THRESHOLD * 100:.0f}%) 이상으로 감지되어 "
            "위음성(도핑 양성 오분류) 방지를 위해 Caution 등급 이상으로 상향 경고합니다."
        )
        return raw_class, "Caution", True, reason

    if raw_class == "Caution" and high_risk_prob >= HIGH_RISK_CONSERVATIVE_THRESHOLD:
        reason = (
            f"보수적 경고 병기: High Risk 외형 유사도 점수({high_risk_prob * 100:.1f}%)가 "
            f"기준치({HIGH_RISK_CONSERVATIVE_THRESHOLD * 100:.0f}%) 이상 감지되었습니다."
        )
        return raw_class, "Caution", True, reason

    return raw_class, raw_class, False, ""


def build_wada_recommendation(
    final_class: str,
    filename: Optional[str] = None,
) -> Dict[str, object]:
    """Build WADA code metadata and safe alternative/TUE guidance for High Risk and Caution."""
    if final_class not in ("High Risk", "Caution"):
        return {
            "wada_code": "",
            "wada_category_name": "",
            "prohibition_scope": "",
            "recommendation_headline": "",
            "alternatives_list": [],
            "tue_notice": "",
            "is_estimated_group": False,
            "prescription_only": False,
            "matched_item_name": "",
        }

    detected_code, matched_item_name = _lookup_wada_code_from_filename(filename)
    resolved_code, info = resolve_wada_alternative_entry(detected_code, final_class)
    is_estimated = detected_code is None
    category_name = str(info["category_name"])
    prohibition_scope = str(info["prohibition_scope"])

    headline = (
        f"이 약물은 WADA 금지약물 리스트 중 [{resolved_code}: {category_name}]에 해당하는 약물로 "
        f"{prohibition_scope} 약물입니다. 대체 약품(허용 성분 참고)을 추천드리겠습니다."
    )

    return {
        "wada_code": resolved_code,
        "wada_category_name": category_name,
        "prohibition_scope": prohibition_scope,
        "recommendation_headline": headline,
        "alternatives_list": list(info["alternatives"]),
        "tue_notice": str(info["tue_notice"]),
        "is_estimated_group": is_estimated,
        "prescription_only": bool(info.get("prescription_only", False)),
        "matched_item_name": matched_item_name if not is_estimated else "",
    }


class DopingPredictor:
    """In-memory pill image classifier with Grad-CAM XAI and WADA guide mapping."""

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.model_meta = load_trained_model(
            weights_path=weights_path,
            device=self.device,
        )
        self.backbone = self.model_meta["backbone"]
        self.is_trained_weights = bool(self.model_meta["is_trained"])
        self.eval_transform = get_eval_transforms()

    def predict_from_bytes(
        self, raw_bytes: bytes, filename: Optional[str] = None
    ) -> PredictionResult:
        """Run validation, classification, conservative threshold check, and Grad-CAM in RAM."""
        start_time = time.perf_counter()

        rgb_img = validate_and_decode_bytes(raw_bytes, filename=filename)
        transformed = self.eval_transform(image=rgb_img)["image"]
        input_tensor = numpy_to_tensor(transformed).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs_tensor = torch.softmax(logits, dim=1)[0].cpu()

            probs = {
                class_name: float(probs_tensor[idx].item())
                for idx, class_name in enumerate(CLASS_NAMES)
            }
            raw_class, final_class, escalated, escalation_reason = apply_conservative_rule(probs)

            cam_target_idx = (
                CLASS_NAMES.index("High Risk")
                if escalated
                else CLASS_NAMES.index(raw_class)
            )
            heatmap_image, cam_explanation = generate_gradcam_overlay(
                model=self.model,
                input_tensor=input_tensor,
                original_rgb=rgb_img,
                target_class_idx=cam_target_idx,
                backbone=self.backbone,
            )

            meta = CLASS_META[final_class]
            wada_rec = build_wada_recommendation(final_class, filename=filename)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return PredictionResult(
                raw_class=raw_class,
                final_class=final_class,
                escalated=escalated,
                escalation_reason=escalation_reason,
                probabilities=probs,
                heatmap_image=heatmap_image,
                cam_explanation=cam_explanation,
                guide_text=meta["guide"],
                wada_ref=meta["wada_ref"],
                disclaimer=DISCLAIMER_TEXT,
                elapsed_ms=elapsed_ms,
                backbone=self.backbone,
                is_trained_weights=self.is_trained_weights,
                wada_code=str(wada_rec["wada_code"]),
                wada_category_name=str(wada_rec["wada_category_name"]),
                prohibition_scope=str(wada_rec["prohibition_scope"]),
                recommendation_headline=str(wada_rec["recommendation_headline"]),
                alternatives_list=list(wada_rec["alternatives_list"]),
                tue_notice=str(wada_rec["tue_notice"]),
                is_estimated_group=bool(wada_rec["is_estimated_group"]),
                prescription_only=bool(wada_rec["prescription_only"]),
                matched_item_name=str(wada_rec["matched_item_name"]),
            )
        finally:
            # Explicit zero-retention cleanup of image tensors and pixel arrays in RAM.
            del input_tensor
            del rgb_img
            del transformed

    def predict_from_path(self, image_path: str | Path) -> PredictionResult:
        """Read image file into memory, run inference, and avoid keeping copies."""
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
        raw_bytes = path.read_bytes()
        try:
            return self.predict_from_bytes(raw_bytes, filename=path.name)
        finally:
            del raw_bytes
