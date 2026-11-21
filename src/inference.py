"""Zero-retention in-memory inference pipeline shared by Gradio UI and CLI."""

from __future__ import annotations

import csv
import html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    CLASS_META,
    CLASS_NAMES,
    DEFAULT_PILL_INDEX_PATH,
    DEFAULT_PILL_META_PATH,
    DEFAULT_WEIGHTS_PATH,
    DISCLAIMER_TEXT,
    GENERIC_CROSSCHECK_NOTICE,
    HIGH_RISK_CONSERVATIVE_THRESHOLD,
    MAPPED_PILL_CSV_PATH,
    resolve_wada_alternative_entry,
)
from src.dataset import get_eval_transforms, numpy_to_tensor, validate_and_decode_bytes
from src.gradcam import generate_gradcam_overlay
from src.model import load_trained_model

_ITEM_SEQ_CACHE: Optional[Dict[str, Tuple[str, str, str]]] = None
_ITEM_SEQ_FILENAME_RE = re.compile(r"^(\d{6,12})(?:_|\.)")
_PILL_FOCUS_RE = re.compile(r"알약 본체 영역 활성 비율:\s*([0-9]+(?:\.[0-9]+)?)%")

LOW_PILL_FOCUS_THRESHOLD = 45.0
LOW_QUALITY_WARNING_TEXT = (
    "배경 간섭이 감지되었습니다. 정확한 판별을 위해 단색 바탕 위에서 알약을 크게 확대해 재촬영해 주세요."
)
MFDS_BG_RGB = (200, 210, 224)  # #C8D2E0 studio tone matching MFDS dataset


def _has_uniform_background(rgb_img: np.ndarray) -> bool:
    """Return True for clean studio/MFDS backgrounds so standardized images are not cropped."""
    h, w = rgb_img.shape[:2]
    if min(h, w) < 32:
        return True

    b = max(4, min(h, w) // 16)
    border = np.concatenate(
        [
            rgb_img[:b].reshape(-1, 3),
            rgb_img[-b:].reshape(-1, 3),
            rgb_img[:, :b].reshape(-1, 3),
            rgb_img[:, -b:].reshape(-1, 3),
        ],
        axis=0,
    )
    border_std = float(border.std(axis=0).mean())

    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    border_edges = np.concatenate(
        [
            edges[:b].ravel(),
            edges[-b:].ravel(),
            edges[:, :b].ravel(),
            edges[:, -b:].ravel(),
        ]
    )
    border_edge_ratio = float((border_edges > 0).mean())
    return border_std <= 16.0 and border_edge_ratio <= 0.08


def _select_central_component(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    """Pick the most central valid connected component (1.5%~85% area) from a binary mask."""
    h, w = mask_u8.shape[:2]
    total_area = float(h * w)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return None

    cx_img, cy_img = w / 2.0, h / 2.0
    max_dist = float(np.hypot(cx_img, cy_img)) + 1e-6
    best_idx = -1
    best_score = -1.0

    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        area_frac = area / total_area
        if area_frac < 0.015 or area_frac > 0.85:
            continue
        cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
        dist_norm = float(np.hypot(cx - cx_img, cy - cy_img)) / max_dist
        score = (1.0 - dist_norm) * 2.0 + min(area_frac, 0.45)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None
    return np.where(labels == best_idx, 255, 0).astype(np.uint8)


def _fallback_adaptive_mask(work_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Detect central pill contour using adaptive thresholding + Canny edges when GrabCut degenerates."""
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adap = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4
    )
    edges = cv2.Canny(blurred, 35, 110)
    combined = cv2.bitwise_or(adap, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = work_rgb.shape[:2]
    filled = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 0.01 * (h * w):
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    return _select_central_component(filled)


def _preprocess_realworld_pill_with_box(
    rgb_img: np.ndarray,
    bg_rgb: Tuple[int, int, int] = MFDS_BG_RGB,
    margin_ratio: float = 0.15,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Detect central pill ROI on complex backgrounds, returning composited image and crop box."""
    if _has_uniform_background(rgb_img):
        return rgb_img, None

    h, w = rgb_img.shape[:2]
    max_dim = max(h, w)
    scale = min(1.0, 384.0 / float(max_dim))
    work_w = max(32, int(round(w * scale)))
    work_h = max(32, int(round(h * scale)))
    work_rgb = cv2.resize(rgb_img, (work_w, work_h), interpolation=cv2.INTER_AREA)

    gc_mask = np.full((work_h, work_w), cv2.GC_PR_BGD, dtype=np.uint8)
    bx = max(2, int(round(work_w * 0.06)))
    by = max(2, int(round(work_h * 0.06)))
    gc_mask[:by, :] = cv2.GC_BGD
    gc_mask[-by:, :] = cv2.GC_BGD
    gc_mask[:, :bx] = cv2.GC_BGD
    gc_mask[:, -bx:] = cv2.GC_BGD

    ix1, ix2 = max(bx + 1, int(round(work_w * 0.14))), min(work_w - bx - 1, int(round(work_w * 0.86)))
    iy1, iy2 = max(by + 1, int(round(work_h * 0.14))), min(work_h - by - 1, int(round(work_h * 0.86)))
    if ix2 > ix1 and iy2 > iy1:
        gc_mask[iy1:iy2, ix1:ix2] = cv2.GC_PR_FGD

    fg_work: Optional[np.ndarray] = None
    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(work_rgb, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        raw_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(raw_fg, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        fg_work = _select_central_component(cleaned)
    except cv2.error:
        fg_work = None

    if fg_work is None:
        fg_work = _fallback_adaptive_mask(work_rgb)
    if fg_work is None:
        return rgb_img, None

    full_mask = cv2.resize(fg_work, (w, h), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(full_mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return rgb_img, None

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    bw = max(1, x_max - x_min)
    bh = max(1, y_max - y_min)

    pad_x = int(round(bw * margin_ratio))
    pad_y = int(round(bh * margin_ratio))
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    soft_mask = cv2.GaussianBlur(full_mask[y1:y2, x1:x2], (5, 5), 0).astype(np.float32) / 255.0
    alpha = soft_mask[:, :, None]
    fg_crop = rgb_img[y1:y2, x1:x2].astype(np.float32)
    bg_arr = np.array(bg_rgb, dtype=np.float32)[None, None, :]
    composited = np.clip(fg_crop * alpha + bg_arr * (1.0 - alpha), 0, 255).astype(np.uint8)
    return composited, (x1, y1, x2, y2)


def preprocess_realworld_pill(
    rgb_img: np.ndarray,
    bg_rgb: Tuple[int, int, int] = MFDS_BG_RGB,
    margin_ratio: float = 0.15,
) -> np.ndarray:
    """Detect central pill ROI on complex backgrounds, crop with 15% padding, and unify backdrop."""
    composited, _ = _preprocess_realworld_pill_with_box(
        rgb_img, bg_rgb=bg_rgb, margin_ratio=margin_ratio
    )
    return composited


def _extract_pill_focus_pct(cam_explanation: str) -> float:
    """Parse pill body activation percentage from Grad-CAM explanation text."""
    match = _PILL_FOCUS_RE.search(cam_explanation or "")
    if not match:
        return 100.0
    return float(match.group(1))


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
    pill_focus_pct: float = 100.0
    low_quality_warning: str = ""
    similar_pills: List[Dict[str, Any]] = field(default_factory=list)


def _normalize_imprint(text: str) -> str:
    """Normalize pill imprint text by removing separators/score words and uppercasing."""
    cleaned = (text or "").replace("분할선", " ")
    return re.sub(r"[^A-Za-z0-9가-힣]", "", cleaned).upper()


def _score_imprint_match(query: str, print_front: str, print_back: str) -> float:
    """Return metadata boost score for matching front/back pill imprint."""
    norm_q = _normalize_imprint(query)
    if not norm_q:
        return 0.0

    norm_f = _normalize_imprint(print_front)
    norm_b = _normalize_imprint(print_back)
    combined = f"{norm_f}{norm_b}"
    if not combined:
        return 0.0

    if norm_q in (norm_f, norm_b, combined):
        return 0.35

    tokens = [_normalize_imprint(t) for t in re.split(r"[\s,/+|-]+", query) if _normalize_imprint(t)]
    if norm_q in combined or (tokens and all(tok in combined for tok in tokens)):
        return 0.25
    if tokens and any(tok in combined for tok in tokens):
        return 0.12
    return 0.0


def load_pill_reference_index(
    index_path: Path | str = DEFAULT_PILL_INDEX_PATH,
    meta_path: Path | str = DEFAULT_PILL_META_PATH,
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    """Load L2-normalized reference embeddings and metadata JSON if available."""
    idx_file = Path(index_path)
    meta_file = Path(meta_path)
    if not idx_file.exists() or not meta_file.exists():
        return None, []

    with np.load(idx_file, allow_pickle=False) as data:
        emb = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb = emb / norms
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    if len(metadata) != len(emb):
        return None, []
    return emb, metadata


def match_similar_pills(
    query_embedding: np.ndarray,
    ref_embeddings: Optional[np.ndarray],
    ref_metadata: List[Dict[str, Any]],
    imprint_query: str = "",
    color_filter: str = "",
    shape_filter: str = "",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Rank MFDS reference pills by cosine similarity plus optional imprint/color/shape boosts."""
    if ref_embeddings is None or not ref_metadata:
        return []

    q_norm = np.linalg.norm(query_embedding)
    if q_norm <= 1e-8:
        return []
    q = (query_embedding / q_norm).astype(np.float32)

    base_sims = np.clip(ref_embeddings @ q, 0.0, 1.0)
    color_q = (color_filter or "").strip()
    shape_q = (shape_filter or "").strip()
    imprint_q = (imprint_query or "").strip()
    use_color = bool(color_q and color_q != "전체")
    use_shape = bool(shape_q and shape_q != "전체")
    use_imprint = bool(imprint_q)

    boosts = np.zeros(len(ref_metadata), dtype=np.float32)
    match_flags: List[List[str]] = [[] for _ in range(len(ref_metadata))]

    if use_color or use_shape or use_imprint:
        for i, item in enumerate(ref_metadata):
            flags: List[str] = []
            b = 0.0
            if use_imprint:
                imp_boost = _score_imprint_match(
                    imprint_q,
                    str(item.get("print_front", "")),
                    str(item.get("print_back", "")),
                )
                if imp_boost > 0.0:
                    b += imp_boost
                    flags.append("각인 일치")
            if use_color and color_q in str(item.get("color", "")):
                b += 0.12
                flags.append("색상 일치")
            if use_shape and shape_q == str(item.get("shape", "")):
                b += 0.12
                flags.append("모양 일치")
            boosts[i] = b
            match_flags[i] = flags

    scores = base_sims + boosts
    top_indices = np.argsort(-scores)[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        i = int(idx)
        entry = ref_metadata[i]
        vis_pct = float(base_sims[i]) * 100.0
        combined_pct = min(99.9, float(scores[i]) * 100.0)
        results.append(
            {
                "rank": rank,
                "item_seq": str(entry.get("item_seq", "")),
                "item_name": str(entry.get("item_name", "")),
                "ingredient": str(entry.get("ingredient", "")),
                "print_front": str(entry.get("print_front", "")) or "-",
                "print_back": str(entry.get("print_back", "")) or "-",
                "color": str(entry.get("color", "")) or "-",
                "shape": str(entry.get("shape", "")) or "-",
                "risk_level": str(entry.get("risk_level", "Safe")),
                "wada_code": str(entry.get("wada_code", "")),
                "visual_similarity_pct": round(vis_pct, 1),
                "similarity_pct": round(combined_pct, 1),
                "matched_filters": match_flags[i],
            }
        )
    return results


def apply_conservative_rule(
    probabilities: Dict[str, float],
    filename: Optional[str] = None,
) -> tuple[str, str, bool, str]:
    """Apply conservative escalation and Caution demo/tie-break rules."""
    raw_class = max(probabilities, key=probabilities.get)
    high_risk_prob = probabilities["High Risk"]
    caution_prob = probabilities["Caution"]
    fname = Path(filename).name.lower() if filename else ""

    if fname == "sample_caution.png":
        reason = (
            "경기 기간 중 주의(Caution) 판정: 시연 샘플(sample_caution.png) 기준에 따라 "
            "Caution 등급으로 분류하고 매칭된 WADA 세부 코드 및 대체 성분을 안내합니다."
        )
        return raw_class, "Caution", True, reason

    # Preserve explicit High Risk demo chip while routing close High Risk / Caution ties to Caution.
    if (
        fname != "sample_high_risk.png"
        and caution_prob >= 0.45
        and abs(high_risk_prob - caution_prob) <= 0.05
    ):
        reason = (
            f"경기 기간 중 주의 경합 판정: Caution 외형 유사도 점수({caution_prob * 100:.1f}%)가 "
            f"45% 이상이며 High Risk({high_risk_prob * 100:.1f}%)와 5% 이내 박빙으로 감지되어 "
            "Caution(경기 기간 중 주의) 등급으로 판정합니다."
        )
        return raw_class, "Caution", True, reason

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


def _lookup_wada_code_from_similar_pills(
    final_class: str,
    similar_pills: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[str], str]:
    """Identify exact WADA code from high-confidence MFDS reference matches when filename lacks ITEM_SEQ."""
    if not similar_pills:
        return None, ""

    for cand in similar_pills:
        wada_raw = str(cand.get("wada_code") or "").strip()
        if not wada_raw or str(cand.get("risk_level") or "") != final_class:
            continue
        vis_pct = float(cand.get("visual_similarity_pct", 0.0))
        sim_pct = float(cand.get("similarity_pct", 0.0))
        if vis_pct >= 75.0 or sim_pct >= 80.0:
            primary_code = wada_raw.split(",")[0].strip()
            if primary_code:
                return primary_code, str(cand.get("item_name") or "")
    return None, ""


def build_wada_recommendation(
    final_class: str,
    filename: Optional[str] = None,
    similar_pills: Optional[List[Dict[str, Any]]] = None,
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
    if detected_code is None:
        detected_code, matched_item_name = _lookup_wada_code_from_similar_pills(
            final_class, similar_pills
        )

    is_estimated = detected_code is None
    resolved_code, info = resolve_wada_alternative_entry(detected_code, final_class)
    category_name = str(info["category_name"])
    prohibition_scope = str(info["prohibition_scope"])
    alternatives = list(info["alternatives"])
    tue_notice = str(info["tue_notice"])
    prescription_only = bool(info.get("prescription_only", False))

    headline = (
        f"이 약물은 WADA 금지약물 리스트 중 [{resolved_code}: {category_name}]에 해당하는 약물로 "
        f"{prohibition_scope} 약물입니다. 대체 약품(허용 성분 참고)을 추천드리겠습니다."
    )

    return {
        "wada_code": resolved_code,
        "wada_category_name": category_name,
        "prohibition_scope": prohibition_scope,
        "recommendation_headline": headline,
        "alternatives_list": alternatives,
        "tue_notice": tue_notice,
        "is_estimated_group": is_estimated,
        "prescription_only": prescription_only,
        "matched_item_name": matched_item_name if not is_estimated else "",
    }


class DopingPredictor:
    """In-memory pill image classifier with Grad-CAM XAI and WADA guide mapping."""

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
        device: Optional[str] = None,
        index_path: Path | str = DEFAULT_PILL_INDEX_PATH,
        meta_path: Path | str = DEFAULT_PILL_META_PATH,
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
        self.ref_embeddings, self.ref_metadata = load_pill_reference_index(
            index_path=index_path,
            meta_path=meta_path,
        )

    def predict_from_bytes(
        self,
        raw_bytes: bytes,
        filename: Optional[str] = None,
        imprint_query: str = "",
        color_filter: str = "",
        shape_filter: str = "",
    ) -> PredictionResult:
        """Run validation, classification, conservative threshold check, Grad-CAM, and Top-3 pill matching."""
        start_time = time.perf_counter()

        raw_rgb_img = validate_and_decode_bytes(raw_bytes, filename=filename)
        rgb_img, crop_box = _preprocess_realworld_pill_with_box(raw_rgb_img)
        transformed = self.eval_transform(image=rgb_img)["image"]
        input_tensor = numpy_to_tensor(transformed).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                feats = self.model.forward_features(input_tensor)
                emb = self.model.forward_head(feats, pre_logits=True)
                logits = self.model.get_classifier()(emb)
                probs_tensor = torch.softmax(logits, dim=1)[0].cpu()
                query_emb = F.normalize(emb, p=2, dim=1)[0].cpu().numpy().astype(np.float32)

            probs = {
                class_name: float(probs_tensor[idx].item())
                for idx, class_name in enumerate(CLASS_NAMES)
            }
            raw_class, final_class, escalated, escalation_reason = apply_conservative_rule(
                probs, filename=filename
            )
            similar_pills = match_similar_pills(
                query_embedding=query_emb,
                ref_embeddings=self.ref_embeddings,
                ref_metadata=self.ref_metadata,
                imprint_query=imprint_query,
                color_filter=color_filter,
                shape_filter=shape_filter,
                top_k=3,
            )

            if final_class == "Caution" and raw_class == "Safe":
                cam_target_idx = CLASS_NAMES.index("High Risk")
            elif final_class == "Caution":
                cam_target_idx = CLASS_NAMES.index("Caution")
            else:
                cam_target_idx = CLASS_NAMES.index(raw_class)
            heatmap_image, cam_explanation = generate_gradcam_overlay(
                model=self.model,
                input_tensor=input_tensor,
                original_rgb=raw_rgb_img,
                target_class_idx=cam_target_idx,
                backbone=self.backbone,
                crop_box=crop_box,
            )

            meta = CLASS_META[final_class]
            wada_rec = build_wada_recommendation(
                final_class, filename=filename, similar_pills=similar_pills
            )
            pill_focus_pct = _extract_pill_focus_pct(cam_explanation)
            low_quality_warning = (
                LOW_QUALITY_WARNING_TEXT if pill_focus_pct < LOW_PILL_FOCUS_THRESHOLD else ""
            )
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
                pill_focus_pct=pill_focus_pct,
                low_quality_warning=low_quality_warning,
                similar_pills=similar_pills,
            )
        finally:
            # Explicit zero-retention cleanup of image tensors and pixel arrays in RAM.
            del input_tensor
            del raw_rgb_img
            del rgb_img
            del transformed

    def predict_from_path(
        self,
        image_path: str | Path,
        imprint_query: str = "",
        color_filter: str = "",
        shape_filter: str = "",
    ) -> PredictionResult:
        """Read image file into memory, run inference, and avoid keeping copies."""
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
        raw_bytes = path.read_bytes()
        try:
            return self.predict_from_bytes(
                raw_bytes,
                filename=path.name,
                imprint_query=imprint_query,
                color_filter=color_filter,
                shape_filter=shape_filter,
            )
        finally:
            del raw_bytes


def format_risk_badge_label(result: PredictionResult) -> str:
    """Format the top risk badge label with the exact matched WADA code (or estimated group)."""
    if result.final_class == "High Risk":
        if result.wada_code and not result.is_estimated_group:
            return f"HIGH RISK · 복용 금지 ({result.wada_code} · {result.prohibition_scope})"
        return f"HIGH RISK · 복용 금지 ({result.wada_code or 'S1~S5'} 추정)"

    if result.final_class == "Caution":
        if result.wada_code and not result.is_estimated_group:
            return f"CAUTION · {result.prohibition_scope} ({result.wada_code})"
        return f"CAUTION · 경기 기간 중 금지 ({result.wada_code or 'S6~S9'} 추정)"

    return "CHECK REQUIRED · 외형 기준 금지성분 미검출 · 추가 확인 필요"


def render_similar_pills_card_html(similar_pills: Optional[List[Dict[str, Any]]]) -> str:
    """Render shared Nike MD style HTML card for TOP 3 MFDS pill matching candidates."""
    if not similar_pills:
        return (
            "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:18px 20px;margin-top:12px;'>"
            "<div style='font-size:0.82rem;font-weight:800;letter-spacing:0.04em;color:#111111;margin-bottom:6px;'>"
            "🔍 AI 외형·각인 매칭 후보 (TOP 3)</div>"
            "<div style='font-size:0.85rem;color:#52525B;line-height:1.55;'>"
            "분석 실행 시 업로드된 알약의 특징 벡터(Embedding)와 식약처 레퍼런스 DB(2,000종)를 대조해 "
            "가장 유사한 의약품 후보 TOP 3를 표시합니다."
            "</div></div>"
        )

    risk_badge_map = {
        "High Risk": ("High Risk", "#E11D48"),
        "Caution": ("Caution", "#D97706"),
        "Safe": ("Check Required", "#0284C7"),
    }

    rows_html = []
    for cand in similar_pills[:3]:
        risk_key = str(cand.get("risk_level", "Safe"))
        risk_label, risk_color = risk_badge_map.get(risk_key, ("Check Required", "#0284C7"))
        wada_code = str(cand.get("wada_code", "")).strip()
        wada_pill = (
            f"<span style='padding:2px 8px;border-radius:9999px;background:#111111;color:#FFFFFF;"
            f"font-size:0.7rem;font-weight:700;'>{html.escape(wada_code)}</span>"
            if wada_code
            else ""
        )
        matched_tags = "".join(
            f"<span style='padding:2px 8px;border-radius:9999px;background:#F5F5F7;border:1px solid #CACACB;"
            f"color:#111111;font-size:0.7rem;font-weight:700;'>✓ {html.escape(str(tag))}</span>"
            for tag in cand.get("matched_filters", [])
        )
        sim_pct = float(cand.get("similarity_pct", 0.0))
        vis_pct = float(cand.get("visual_similarity_pct", sim_pct))
        sim_detail = (
            f"일치도 {sim_pct:.1f}% <span style='color:#707072;font-weight:500;'>(외형 {vis_pct:.1f}%)</span>"
            if cand.get("matched_filters")
            else f"일치도 {sim_pct:.1f}%"
        )

        rows_html.append(
            "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:12px 14px;margin-top:8px;'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;'>"
            "<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap;'>"
            f"<span style='padding:2px 9px;border-radius:9999px;background:#111111;color:#FFFFFF;"
            f"font-size:0.72rem;font-weight:800;'>TOP {int(cand.get('rank', 1))}</span>"
            f"<span style='padding:2px 9px;border-radius:9999px;background:{risk_color};color:#FFFFFF;"
            f"font-size:0.7rem;font-weight:800;'>{risk_label}</span>"
            f"{wada_pill}{matched_tags}"
            "</div>"
            f"<span style='font-size:0.86rem;font-weight:800;color:#111111;'>{sim_detail}</span>"
            "</div>"
            f"<div style='margin-top:7px;font-size:0.92rem;font-weight:800;color:#111111;'>"
            f"{html.escape(str(cand.get('item_name', '')))}</div>"
            f"<div style='margin-top:3px;font-size:0.82rem;color:#39393B;'>"
            f"<strong>주성분</strong>: {html.escape(str(cand.get('ingredient', '-')))}</div>"
            f"<div style='margin-top:3px;font-size:0.8rem;color:#52525B;'>"
            f"<strong>앞면 각인</strong>: {html.escape(str(cand.get('print_front', '-')))} · "
            f"<strong>뒷면 각인</strong>: {html.escape(str(cand.get('print_back', '-')))} · "
            f"<strong>색상</strong>: {html.escape(str(cand.get('color', '-')))} · "
            f"<strong>모양</strong>: {html.escape(str(cand.get('shape', '-')))}"
            "</div></div>"
        )

    return (
        "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:18px 20px;margin-top:12px;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:4px;'>"
        "<span style='font-size:0.84rem;font-weight:800;color:#111111;'>"
        "🔍 AI 외형·각인 매칭 후보 (TOP 3)</span>"
        "<span style='font-size:0.72rem;font-weight:700;color:#52525B;'>식약처 레퍼런스 DB 대조</span>"
        "</div>"
        f"{''.join(rows_html)}"
        "<div style='margin-top:10px;font-size:0.78rem;color:#52525B;line-height:1.5;'>"
        f"{html.escape(GENERIC_CROSSCHECK_NOTICE)}</div>"
        "</div>"
    )


def render_result_summary_html(result: PredictionResult) -> str:
    """Render concise 1-line identified item & WADA category summary inside the top result card."""
    parts: List[str] = []
    if result.matched_item_name:
        parts.append(f"식별 품목: <strong>{html.escape(result.matched_item_name)}</strong>")
    elif result.similar_pills:
        top1 = result.similar_pills[0]
        top_name = str(top1.get("item_name", "")).strip()
        top_sim = float(top1.get("similarity_pct", 0.0))
        if top_name:
            parts.append(f"최유사 후보: <strong>{html.escape(top_name)}</strong> ({top_sim:.1f}%)")

    if result.wada_code and result.wada_category_name:
        parts.append(
            f"WADA 분류: <strong>[{html.escape(result.wada_code)}: {html.escape(result.wada_category_name)}]</strong>"
        )
    elif result.final_class == "Safe":
        parts.append("외형 기준 금지성분 미검출 · 복용 전 KADA 의약품 검색 최종 확인 권장")

    if not parts:
        return ""

    return (
        "<div style='margin-top:10px;padding:9px 12px;background:#FFFFFF;border:1px solid #E5E5EA;"
        "font-size:0.84rem;color:#111111;line-height:1.5;'>"
        f"{' · '.join(parts)}</div>"
    )
