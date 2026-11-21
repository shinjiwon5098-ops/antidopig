"""PLAY TRUE AI - Nike MD Editorial Style Gradio Web UI for WADA Doping Screening."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
from PIL import Image

from src.config import (
    APP_SUBTITLE,
    DEFAULT_WEIGHTS_PATH,
    DISCLAIMER_TEXT,
    EMPTY_INPUT_WARNING,
    HIGH_RISK_CONSERVATIVE_THRESHOLD,
    OTC_RECOMMENDATION_NOTICE,
    PILL_COLOR_OPTIONS,
    PILL_SHAPE_OPTIONS,
    SAMPLES_DIR,
    SUPPORTED_FORMATS_TEXT,
    get_otc_examples,
)
from src.dataset import (
    InvalidImageFormatError,
    generate_mvp_pill_dataset,
    validate_and_decode_bytes,
)
from src.inference import (
    DopingPredictor,
    PredictionResult,
    format_risk_badge_label,
    render_result_summary_html,
    render_similar_pills_card_html,
)

# Ensure 3 UI sample images exist on startup without regenerating full training splits.
if not all(
    (SAMPLES_DIR / name).exists()
    for name in ("sample_high_risk.png", "sample_caution.png", "sample_safe.png")
):
    generate_mvp_pill_dataset()
PREDICTOR = DopingPredictor(weights_path=DEFAULT_WEIGHTS_PATH)

SAMPLE_MAP = {
    "high_risk": ("sample_high_risk.png", SAMPLES_DIR / "sample_high_risk.png"),
    "caution": ("sample_caution.png", SAMPLES_DIR / "sample_caution.png"),
    "safe": ("sample_safe.png", SAMPLES_DIR / "sample_safe.png"),
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+KR:wght@400;500;600;700;900&display=swap');
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

:root {
    --nike-canvas: #FFFFFF;
    --nike-cloud: #F5F5F7;
    --nike-ink: #111111;
    --nike-charcoal: #39393B;
    --nike-mute: #707072;
    --nike-hairline: #E5E5EA;
    --nike-sky: #0EA5E9;
    --sig-high: #E11D48;
    --sig-caution: #D97706;
    --sig-check: #0284C7;
}

body, .gradio-container {
    background: var(--nike-canvas) !important;
    font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--nike-ink) !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 24px 48px 24px !important;
}

.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .md,
.gradio-container .md p,
.gradio-container .md li,
.gradio-container .md span {
    color: var(--nike-ink) !important;
}

/* Utility Top Strip (Nike Editorial Style) */
.nike-utility-strip {
    background: var(--nike-cloud);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--nike-charcoal);
    border-bottom: 1px solid var(--nike-hairline);
}

/* Editorial Brand Header */
.nike-hero-header {
    background: var(--nike-canvas);
    padding: 28px 0 24px 0;
    border-bottom: 1px solid var(--nike-ink);
    margin-bottom: 28px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}

.nike-eyebrow {
    font-family: 'Bebas Neue', 'Impact', 'Arial Black', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--nike-mute);
    margin-bottom: 4px;
}

.nike-display-title {
    font-family: 'Bebas Neue', 'Impact', 'Arial Black', sans-serif;
    font-size: 3.25rem;
    font-weight: 900;
    line-height: 0.92;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--nike-ink);
    margin: 0;
}

.nike-subtitle {
    font-size: 0.96rem;
    font-weight: 500;
    color: var(--nike-charcoal);
    margin: 10px 0 0 0;
    letter-spacing: -0.01em;
}

.nike-header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--nike-cloud);
    color: var(--nike-ink);
    padding: 8px 16px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Section Step Labels */
.nike-step-label {
    font-family: 'Bebas Neue', 'Impact', 'Arial Black', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--nike-ink);
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--nike-hairline);
}

/* Minimal Flat Panels (No Heavy Borders or Drop Shadows) */
.nike-panel {
    background: var(--nike-canvas) !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 14px !important;
}

.nike-surface-card {
    background: var(--nike-cloud) !important;
    border: 1px solid var(--nike-hairline) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 20px !important;
}

/* Warning Banner */
.warning-banner {
    background: #FFF1F2;
    border-left: 4px solid var(--sig-high);
    color: var(--nike-ink);
    padding: 12px 16px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 16px;
}

/* Upload Dropzone */
.upload-dropzone {
    background: var(--nike-cloud) !important;
    border: 1px dashed #B4B4BA !important;
    border-radius: 0 !important;
    padding: 16px !important;
    min-height: 170px !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, background 0.15s ease;
}
.upload-dropzone:hover {
    background: #ECECEF !important;
    border-color: var(--nike-ink) !important;
}
.upload-dropzone .wrap,
.upload-dropzone .upload-container,
.upload-dropzone [data-testid="file"] {
    background: transparent !important;
    border: none !important;
}
.upload-dropzone,
.upload-dropzone span,
.upload-dropzone p,
.upload-dropzone label,
.upload-dropzone div {
    color: var(--nike-ink) !important;
}

.file-status-text,
.file-status-text * {
    font-size: 0.84rem !important;
    color: var(--nike-charcoal) !important;
    margin: 2px 0 !important;
}

/* Preview & Grad-CAM Image Surfaces */
.preview-box,
.gradcam-frame {
    background: var(--nike-cloud) !important;
    border: 1px solid var(--nike-hairline) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
.preview-box img,
.gradcam-frame img {
    object-fit: contain !important;
    width: auto !important;
    height: auto !important;
    max-width: 100% !important;
    max-height: 280px !important;
    margin: 0 auto !important;
}

/* Sample Chips */
.sample-heading {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--nike-mute);
    margin: 8px 0 6px 0;
}
.sample-chip-btn button {
    border-radius: 9999px !important;
    border: 1px solid var(--nike-hairline) !important;
    background: var(--nike-canvas) !important;
    color: var(--nike-ink) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
    min-height: 36px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
.sample-chip-btn button:hover {
    background: var(--nike-ink) !important;
    border-color: var(--nike-ink) !important;
    color: var(--nike-canvas) !important;
}

/* Nike Pill CTAs */
.analyze-pill-btn button {
    width: 100% !important;
    border-radius: 9999px !important;
    background: var(--nike-ink) !important;
    border: none !important;
    color: var(--nike-canvas) !important;
    font-size: 0.96rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    min-height: 48px !important;
    box-shadow: none !important;
    margin-top: 8px !important;
    transition: background-color 0.18s ease, transform 0.12s ease !important;
}
.analyze-pill-btn button:hover {
    background: var(--nike-sky) !important;
    color: var(--nike-canvas) !important;
}

.clear-pill-btn button {
    width: 100% !important;
    border-radius: 9999px !important;
    background: var(--nike-cloud) !important;
    border: none !important;
    color: var(--nike-ink) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    min-height: 42px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease !important;
}
.clear-pill-btn button:hover {
    background: #E5E5EA !important;
    color: var(--nike-ink) !important;
}

/* Grad-CAM Inner Block */
.gradcam-card {
    background: var(--nike-cloud);
    border: 1px solid var(--nike-hairline);
    padding: 18px;
}
.gradcam-card-title {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--nike-ink);
    margin: 0 0 10px 0;
}

/* Actionable Guide & Disclaimer */
.guide-block {
    background: var(--nike-canvas);
    border-top: 1px solid var(--nike-hairline);
    padding-top: 16px;
    margin-top: 4px;
}
.guide-heading {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--nike-ink);
    margin: 0 0 8px 0;
}
.guide-body,
.guide-body * {
    font-size: 0.88rem;
    color: var(--nike-charcoal) !important;
    line-height: 1.6;
}
.disclaimer-footnote {
    margin-top: 24px;
    padding: 16px 20px;
    background: var(--nike-cloud);
    border-top: 2px solid var(--nike-ink);
    font-size: 0.8rem;
    line-height: 1.55;
    color: var(--nike-charcoal) !important;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 0 12px 28px 12px !important;
    }
    .nike-hero-header {
        padding: 20px 0 16px 0;
        margin-bottom: 18px;
    }
    .nike-display-title {
        font-size: 2.35rem;
    }
}
"""


def _render_warning_html(message: str) -> gr.update:
    if not message:
        return gr.update(value="", visible=False)
    safe_msg = html.escape(message)
    return gr.update(
        value=f"<div class='warning-banner'>⚠️ {safe_msg}</div>",
        visible=True,
    )


def _render_initial_badge_html() -> str:
    return (
        "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:18px 20px;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>"
        "<span style='display:inline-block;padding:5px 14px;border-radius:9999px;"
        "background:#111111;color:#FFFFFF;font-size:0.78rem;font-weight:800;letter-spacing:0.06em;'>"
        "READY TO SCREEN</span>"
        "<span style='font-size:0.95rem;font-weight:700;color:#111111;'>외형 유사도 점수 --%</span>"
        "</div>"
        "<div style='margin-top:10px;font-size:0.86rem;color:#52525B;'>"
        "좌측 패널에서 알약 이미지를 업로드하거나 샘플을 선택한 뒤 분석을 실행하세요."
        "</div>"
        "</div>"
    )


def _render_initial_probs_html() -> str:
    return (
        "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:18px 20px;'>"
        "<div style='font-size:0.8rem;font-weight:800;letter-spacing:0.06em;"
        "text-transform:uppercase;color:#111111;margin-bottom:10px;'>"
        "3단계 외형 유사도 점수 (Visual Similarity Score)</div>"
        "<div style='display:flex;height:8px;border-radius:9999px;background:#F5F5F7;overflow:hidden;'>"
        "<div style='width:33.3%;background:#D4D4D8;'></div>"
        "<div style='width:33.3%;background:#E4E4E7;'></div>"
        "<div style='width:33.4%;background:#F4F4F5;'></div>"
        "</div>"
        "<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;"
        "margin-top:10px;font-size:0.82rem;font-weight:600;color:#52525B;'>"
        "<span>High Risk 0.0%</span>"
        "<span>Caution 0.0%</span>"
        "<span>Check Required 0.0%</span>"
        "</div></div>"
    )


def _render_initial_alt_card_html() -> str:
    return (
        "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:18px 20px;'>"
        "<div style='font-size:0.8rem;font-weight:800;letter-spacing:0.06em;"
        "text-transform:uppercase;color:#111111;margin-bottom:6px;'>"
        "WADA 대체 약품 · 허용 성분 참고</div>"
        "<div style='font-size:0.86rem;color:#52525B;line-height:1.55;'>"
        "분석 결과 <strong>High Risk</strong> 또는 <strong>Caution</strong> 판정 시 "
        "해당 WADA 세부 코드(S0~S9, P1 또는 추정군), 금지 범위, TUE 안내 및 허용 성분 참고 카드가 표시됩니다."
        "</div></div>"
    )


def _render_result_badge_html(result: PredictionResult) -> str:
    signal_colors = {
        "High Risk": "#E11D48",
        "Caution": "#D97706",
        "Safe": "#0284C7",
    }
    badge_label = format_risk_badge_label(result)
    signal_color = signal_colors[result.final_class]
    score_pct = result.probabilities[result.raw_class] * 100.0

    low_quality_html = ""
    if result.low_quality_warning:
        low_quality_html = (
            "<div style='margin-bottom:12px;padding:11px 14px;background:#FFFBEB;"
            "border:1px solid #FDE68A;border-left:4px solid #D97706;"
            "color:#111111;font-size:0.86rem;font-weight:700;line-height:1.5;'>"
            f"⚠️ {html.escape(result.low_quality_warning)}</div>"
        )

    escalation_html = ""
    if result.escalated:
        escalation_html = (
            "<div style='margin-top:12px;padding:10px 14px;background:#FFFFFF;"
            "border-left:3px solid #D97706;color:#111111;font-size:0.83rem;line-height:1.5;'>"
            f"🛡️ {html.escape(result.escalation_reason)}</div>"
        )

    summary_html = render_result_summary_html(result)

    return (
        "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:18px 20px;'>"
        f"{low_quality_html}"
        "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>"
        f"<span style='display:inline-block;padding:6px 14px;border-radius:9999px;"
        f"background:{signal_color};color:#FFFFFF;font-size:0.78rem;font-weight:800;"
        f"letter-spacing:0.04em;'>{badge_label}</span>"
        "<div style='text-align:right;'>"
        f"<span style='font-size:1.05rem;font-weight:800;color:#111111;'>외형 유사도 점수 {score_pct:.1f}%</span>"
        f"<span style='margin-left:6px;font-size:0.78rem;font-weight:500;color:#707072;'>({result.elapsed_ms:.0f}ms)</span>"
        "</div></div>"
        f"{summary_html}"
        f"{escalation_html}"
        "</div>"
    )


def _render_probs_html(result: PredictionResult) -> str:
    p_high = result.probabilities["High Risk"] * 100.0
    p_caution = result.probabilities["Caution"] * 100.0
    p_check = result.probabilities["Safe"] * 100.0

    return (
        "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:18px 20px;'>"
        "<div style='font-size:0.8rem;font-weight:800;letter-spacing:0.06em;"
        "text-transform:uppercase;color:#111111;margin-bottom:10px;'>"
        "3단계 외형 유사도 점수</div>"
        "<div style='display:flex;height:8px;border-radius:9999px;background:#F5F5F7;overflow:hidden;'>"
        f"<div style='width:{p_high:.2f}%;background:#E11D48;transition:width 0.3s ease;' title='High Risk {p_high:.1f}%'></div>"
        f"<div style='width:{p_caution:.2f}%;background:#D97706;transition:width 0.3s ease;' title='Caution {p_caution:.1f}%'></div>"
        f"<div style='width:{p_check:.2f}%;background:#0284C7;transition:width 0.3s ease;' title='Check Required {p_check:.1f}%'></div>"
        "</div>"
        "<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;"
        "margin-top:10px;font-size:0.83rem;font-weight:700;color:#111111;'>"
        f"<span><span style='color:#E11D48;'>●</span> High Risk {p_high:.1f}%</span>"
        f"<span><span style='color:#D97706;'>●</span> Caution {p_caution:.1f}%</span>"
        f"<span><span style='color:#0284C7;'>●</span> Check Required {p_check:.1f}%</span>"
        "</div>"
        "<div style='margin-top:8px;font-size:0.77rem;color:#707072;'>"
        f"※ 보수적 안전 규칙: High Risk 외형 유사도 점수 ≥ {HIGH_RISK_CONSERVATIVE_THRESHOLD * 100:.0f}% 시 Caution 이상 자동 경고"
        "</div></div>"
    )


def _render_alt_card_html(result: PredictionResult) -> str:
    if result.final_class not in ("High Risk", "Caution"):
        return (
            "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:18px 20px;'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px;'>"
            "<span style='font-size:0.8rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#111111;'>"
            "WADA 교차 확인 안내</span>"
            "<span style='display:inline-block;padding:3px 10px;border-radius:9999px;"
            "background:#0284C7;color:#FFFFFF;font-size:0.72rem;font-weight:700;'>CHECK REQUIRED</span>"
            "</div>"
            "<div style='font-size:0.88rem;color:#111111;line-height:1.6;'>"
            "AI 외형 분석상 금지약물 고위험 신호가 검출되지 않았으나, <strong>동일 외형 복제약(제네릭)</strong>이나 "
            "복합 감기약·진통제(에페드린·코데인 등 첨가)일 수 있습니다. "
            "복용 전 <strong>KADA 의약품 검색 서비스</strong>에서 실제 성분명을 반드시 확인하십시오."
            "</div></div>"
        )

    signal_color = "#E11D48" if result.final_class == "High Risk" else "#D97706"
    tag_text = (
        "추정군 매핑 (미특정)"
        if result.is_estimated_group
        else f"식약처 레퍼런스 매칭 · {html.escape(result.wada_code)}"
    )
    item_line = (
        f"<div style='font-size:0.8rem;color:#52525B;margin-bottom:8px;'>식별 품목: <strong>{html.escape(result.matched_item_name)}</strong></div>"
        if result.matched_item_name
        else ""
    )

    items_html = "".join(
        f"<li style='margin-bottom:6px;color:#111111;'>{html.escape(item)}</li>"
        for item in result.alternatives_list
    )

    otc_examples = get_otc_examples(result.wada_code)
    otc_section_html = ""
    if otc_examples:
        otc_rows = []
        for entry in otc_examples:
            group_label = html.escape(str(entry["label"]))
            ingredients = ", ".join(html.escape(str(x)) for x in entry["ingredients"])
            products = ", ".join(html.escape(str(p)) for p in entry["products"])
            otc_rows.append(
                "<div style='padding:8px 10px;background:#F5F5F7;border-left:3px solid #111111;margin-top:8px;'>"
                f"<div style='font-size:0.82rem;font-weight:700;color:#111111;'>{group_label}</div>"
                f"<div style='font-size:0.82rem;color:#39393B;margin-top:3px;'>허용 성분: [{ingredients}]</div>"
                f"<div style='font-size:0.84rem;font-weight:600;color:#111111;margin-top:3px;'>"
                f"→ 추천 일반의약품 예시: {products}</div>"
                "</div>"
            )
        otc_section_html = (
            "<div style='margin-top:10px;padding:12px 14px;background:#FFFFFF;border:1px solid #E5E5EA;'>"
            "<div style='font-size:0.78rem;font-weight:800;color:#111111;'>"
            "🏪 약국 구매 가능 대표 일반의약품(OTC) 추천 예시</div>"
            f"{''.join(otc_rows)}"
            "</div>"
            "<div style='margin-top:10px;font-size:0.77rem;color:#52525B;line-height:1.5;'>"
            f"{html.escape(OTC_RECOMMENDATION_NOTICE)}"
            "</div>"
        )

    return (
        "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:20px;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px;'>"
        "<span style='font-size:0.82rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#111111;'>"
        "대체 약품 / 허용 성분 참고 카드</span>"
        "<div style='display:flex;gap:6px;flex-wrap:wrap;'>"
        f"<span style='padding:4px 11px;border-radius:9999px;background:{signal_color};color:#FFFFFF;"
        f"font-size:0.72rem;font-weight:800;'>{html.escape(result.wada_code)}</span>"
        f"<span style='padding:4px 11px;border-radius:9999px;background:#111111;color:#FFFFFF;"
        f"font-size:0.72rem;font-weight:700;'>{html.escape(result.prohibition_scope)}</span>"
        f"<span style='padding:4px 11px;border-radius:9999px;background:#FFFFFF;border:1px solid #CACACB;"
        f"color:#39393B;font-size:0.72rem;font-weight:600;'>{tag_text}</span>"
        "</div></div>"
        f"{item_line}"
        "<div style='background:#FFFFFF;border-left:4px solid "
        f"{signal_color};padding:12px 14px;font-size:0.9rem;font-weight:600;color:#111111;line-height:1.55;'>"
        f"이 약물은 WADA 금지약물 리스트 중 <strong>[{html.escape(result.wada_code)}: {html.escape(result.wada_category_name)}]</strong>에 "
        f"해당하는 약물로 <strong>{html.escape(result.prohibition_scope)}</strong> 약물입니다. "
        "대체 약품(허용 성분 참고)을 추천드리겠습니다."
        "</div>"
        "<div style='margin-top:12px;padding:12px 14px;background:#FFFFFF;border:1px solid #E5E5EA;'>"
        "<div style='font-size:0.78rem;font-weight:800;color:#111111;margin-bottom:6px;'>"
        "📌 전문의 상담 및 TUE(치료목적사용면책) 안내</div>"
        f"<div style='font-size:0.84rem;color:#39393B;line-height:1.55;'>{html.escape(result.tue_notice)}</div>"
        "</div>"
        "<div style='margin-top:10px;padding:12px 14px;background:#FFFFFF;border:1px solid #E5E5EA;'>"
        "<div style='font-size:0.78rem;font-weight:800;color:#111111;margin-bottom:6px;'>"
        "💊 허용 성분 참고 · 권장 조치</div>"
        f"<ul style='margin:0;padding-left:18px;font-size:0.84rem;line-height:1.55;'>{items_html}</ul>"
        "</div>"
        f"{otc_section_html}"
        "</div>"
    )


def _safe_cleanup_temp_file(filepath: str) -> None:
    """Zero-retention cleanup for temporary Gradio upload files."""
    try:
        p = Path(filepath)
        if p.exists() and p.is_file() and SAMPLES_DIR not in p.parents:
            os.remove(p)
    except OSError:
        pass


def handle_file_upload(
    file_path: Optional[str],
) -> Tuple[Optional[dict], Any, str, Any]:
    """Read uploaded file into RAM, validate format (JPG/PNG/WEBP), and purge temp file."""
    if not file_path:
        return None, gr.update(value=None, visible=False), "선택된 파일 없음", _render_warning_html("")

    path = Path(file_path)
    filename = path.name
    try:
        raw_bytes = path.read_bytes()
    finally:
        _safe_cleanup_temp_file(file_path)

    try:
        rgb_np = validate_and_decode_bytes(raw_bytes, filename=filename)
        preview_pil = Image.fromarray(rgb_np)
        payload = {"bytes": raw_bytes, "filename": filename}
        return (
            payload,
            gr.update(value=preview_pil, visible=True),
            f"📄 활성 파일: `{filename}` (메모리 적재 · 디스크 미저장)",
            _render_warning_html(""),
        )
    except InvalidImageFormatError as exc:
        return (
            None,
            gr.update(value=None, visible=False),
            "선택된 파일 없음",
            _render_warning_html(str(exc)),
        )


def handle_sample_select(
    sample_key: str,
) -> Tuple[Optional[dict], Any, str, Any, Optional[str]]:
    """Load one of the 3 sample pill chips into RAM preview."""
    filename, sample_path = SAMPLE_MAP[sample_key]
    raw_bytes = sample_path.read_bytes()
    rgb_np = validate_and_decode_bytes(raw_bytes, filename=filename)
    preview_pil = Image.fromarray(rgb_np)
    payload = {"bytes": raw_bytes, "filename": filename}
    return (
        payload,
        gr.update(value=preview_pil, visible=True),
        f"🧪 샘플 선택됨: `{filename}` (메모리 적재 완료)",
        _render_warning_html(""),
        None,
    )


def handle_analyze(
    image_payload: Optional[dict],
    imprint_query: str = "",
    color_filter: str = "전체",
    shape_filter: str = "전체",
) -> Tuple[Any, str, str, Optional[Image.Image], str, str, str, str]:
    """Execute zero-retention inference, Grad-CAM generation, Top-3 matching, and WADA guidance."""
    if not image_payload or not image_payload.get("bytes"):
        return (
            _render_warning_html(EMPTY_INPUT_WARNING),
            _render_initial_badge_html(),
            _render_initial_probs_html(),
            None,
            "분석할 이미지가 업로드되지 않았습니다.",
            render_similar_pills_card_html(None),
            _render_initial_alt_card_html(),
            "- 이미지를 업로드하거나 샘플 약품 칩을 선택한 뒤 **도핑 위험도 분석 실행** 버튼을 눌러 주세요.",
        )

    try:
        result = PREDICTOR.predict_from_bytes(
            raw_bytes=image_payload["bytes"],
            filename=image_payload.get("filename"),
            imprint_query=imprint_query or "",
            color_filter=color_filter or "전체",
            shape_filter=shape_filter or "전체",
        )
    except InvalidImageFormatError as exc:
        return (
            _render_warning_html(str(exc)),
            _render_initial_badge_html(),
            _render_initial_probs_html(),
            None,
            "지원 이미지 형식을 확인해 주세요.",
            render_similar_pills_card_html(None),
            _render_initial_alt_card_html(),
            f"- 지원 형식: {SUPPORTED_FORMATS_TEXT}",
        )

    heatmap_pil = Image.fromarray(result.heatmap_image)
    return (
        _render_warning_html(""),
        _render_result_badge_html(result),
        _render_probs_html(result),
        heatmap_pil,
        result.cam_explanation,
        render_similar_pills_card_html(result.similar_pills),
        _render_alt_card_html(result),
        result.guide_text,
    )


def handle_sample_and_analyze(
    sample_key: str,
    imprint_query: str = "",
    color_filter: str = "전체",
    shape_filter: str = "전체",
) -> Tuple[Optional[dict], Any, str, Optional[str], Any, str, str, Optional[Image.Image], str, str, str, str]:
    """Load sample pill into RAM preview and immediately run analysis."""
    payload, preview_pil, status_text, _, file_reset = handle_sample_select(sample_key)
    (
        warn_html,
        badge_html,
        probs_html,
        heatmap_pil,
        cam_desc,
        similar_pills_html,
        alt_card_html,
        guide_md,
    ) = handle_analyze(
        payload,
        imprint_query=imprint_query,
        color_filter=color_filter,
        shape_filter=shape_filter,
    )
    return (
        payload,
        preview_pil,
        status_text,
        file_reset,
        warn_html,
        badge_html,
        probs_html,
        heatmap_pil,
        cam_desc,
        similar_pills_html,
        alt_card_html,
        guide_md,
    )


def handle_clear() -> Tuple[
    None, None, Any, str, Any, str, str, None, str, str, str, str, str, str, str
]:
    """Reset all inputs, previews, warnings, filters, and prediction outputs cleanly."""
    return (
        None,  # file_input
        None,  # memory_state
        gr.update(value=None, visible=False),  # preview_image
        "선택된 파일 없음",  # file_status
        _render_warning_html(""),  # warning_banner
        _render_initial_badge_html(),  # risk_badge
        _render_initial_probs_html(),  # prob_bars
        None,  # gradcam_image
        "분석 실행 시 모델이 주목한 각인·분할선·제형 윤곽 히트맵이 표시됩니다.",  # cam_desc
        render_similar_pills_card_html(None),  # similar_pills_card
        _render_initial_alt_card_html(),  # alt_rec_card
        "- 분석 실행 후 등급별 맞춤형 복용 주의 가이드가 표시됩니다.",  # guide_md
        "",  # imprint_input
        "전체",  # color_input
        "전체",  # shape_input
    )


def build_app() -> gr.Blocks:
    """Construct the Nike MD Editorial PLAY TRUE AI Gradio workspace."""
    with gr.Blocks(title="PLAY TRUE AI · CLEAN SPORT SCREENING") as demo:
        memory_state = gr.State(value=None)

        gr.HTML(
            f"""
            <div class="nike-utility-strip">
                <span>WADA PROHIBITED LIST SCREENING ASSISTANT</span>
                <span>ZERO-RETENTION · IN-MEMORY ANALYSIS</span>
            </div>
            <header class="nike-hero-header">
                <div>
                    <div class="nike-eyebrow">CLEAN SPORT SCREENING</div>
                    <h1 class="nike-display-title">PLAY TRUE AI</h1>
                    <p class="nike-subtitle">{html.escape(APP_SUBTITLE)}</p>
                </div>
                <div class="nike-header-badge">🔒 메모리 분석 후 즉시 파기</div>
            </header>
            """
        )

        warning_banner = gr.HTML(value="", visible=False)

        with gr.Row(equal_height=False):
            # Left Column: 01. Image Input & Controls
            with gr.Column(scale=5, elem_classes=["nike-panel"]):
                gr.HTML("<div class='nike-step-label'>01. IMAGE INPUT</div>")

                file_input = gr.File(
                    label="Drag and Drop · Browse (JPG, PNG, WEBP)",
                    file_count="single",
                    type="filepath",
                    elem_classes=["upload-dropzone"],
                )

                file_status = gr.Markdown(
                    "선택된 파일 없음",
                    elem_classes=["file-status-text"],
                )

                preview_image = gr.Image(
                    label="업로드 약물 미리보기",
                    type="pil",
                    interactive=False,
                    visible=False,
                    height=220,
                    elem_classes=["preview-box"],
                )

                gr.HTML("<div class='sample-heading'>SAMPLE PILLS</div>")
                with gr.Row():
                    btn_sample_high = gr.Button(
                        "High Risk · 샘플",
                        size="sm",
                        elem_classes=["sample-chip-btn"],
                    )
                    btn_sample_caution = gr.Button(
                        "Caution · 샘플",
                        size="sm",
                        elem_classes=["sample-chip-btn"],
                    )
                    btn_sample_safe = gr.Button(
                        "Check Required · 샘플",
                        size="sm",
                        elem_classes=["sample-chip-btn"],
                    )

                analyze_btn = gr.Button(
                    "도핑 위험도 분석 실행",
                    variant="primary",
                    elem_classes=["analyze-pill-btn"],
                )
                clear_btn = gr.Button(
                    "초기화",
                    variant="secondary",
                    elem_classes=["clear-pill-btn"],
                )

                with gr.Accordion("선택 입력: 알약 각인 문자 / 색상 / 모양 교차 확인", open=False):
                    imprint_input = gr.Textbox(
                        label="표면 각인 문자 (예: TY 500, PSE, YH)",
                        placeholder="예: TY 500",
                        value="",
                    )
                    with gr.Row():
                        color_input = gr.Dropdown(
                            label="색상",
                            choices=PILL_COLOR_OPTIONS,
                            value="전체",
                        )
                        shape_input = gr.Dropdown(
                            label="모양",
                            choices=PILL_SHAPE_OPTIONS,
                            value="전체",
                        )

            # Right Column: 02. Screening Result & Alternatives
            with gr.Column(scale=7, elem_classes=["nike-panel"]):
                gr.HTML("<div class='nike-step-label'>02. SCREENING RESULT</div>")

                risk_badge = gr.HTML(value=_render_initial_badge_html())

                with gr.Accordion("🔍 AI 외형·각인 매칭 후보 (TOP 3)", open=False):
                    similar_pills_card = gr.HTML(value=render_similar_pills_card_html(None))

                with gr.Accordion("💊 WADA 세부 코드 · 대체 성분 및 일반의약품(OTC) 안내", open=False):
                    alt_rec_card = gr.HTML(value=_render_initial_alt_card_html())

                with gr.Accordion("🎯 Grad-CAM 판단 근거 시각화 및 3단계 예측 확률", open=False):
                    prob_bars = gr.HTML(value=_render_initial_probs_html())
                    with gr.Column(elem_classes=["gradcam-card"]):
                        gr.HTML("<div class='gradcam-card-title'>GRAD-CAM VISUAL EXPLANATION</div>")
                        gradcam_image = gr.Image(
                            show_label=False,
                            type="pil",
                            interactive=False,
                            elem_classes=["gradcam-frame"],
                        )
                        cam_desc = gr.Markdown(
                            "분석 실행 시 모델이 주목한 각인·분할선·제형 윤곽 히트맵이 표시됩니다.",
                            elem_classes=["file-status-text"],
                        )

                with gr.Accordion("📋 등급별 맞춤형 복용 주의 가이드", open=False):
                    guide_md = gr.Markdown(
                        "- 분석 실행 후 등급별 맞춤형 복용 주의 가이드가 표시됩니다.",
                        elem_classes=["guide-body"],
                    )

        gr.HTML(
            f"<footer class='disclaimer-footnote'><strong>[면책 및 교차 확인 필수]</strong> {html.escape(DISCLAIMER_TEXT)}</footer>"
        )

        # Event bindings
        file_input.change(
            fn=handle_file_upload,
            inputs=[file_input],
            outputs=[memory_state, preview_image, file_status, warning_banner],
        )

        for btn_sample, sample_key in (
            (btn_sample_high, "high_risk"),
            (btn_sample_caution, "caution"),
            (btn_sample_safe, "safe"),
        ):
            btn_sample.click(
                fn=lambda imp, col, shp, key=sample_key: handle_sample_and_analyze(
                    key, imprint_query=imp, color_filter=col, shape_filter=shp
                ),
                inputs=[imprint_input, color_input, shape_input],
                outputs=[
                    memory_state,
                    preview_image,
                    file_status,
                    file_input,
                    warning_banner,
                    risk_badge,
                    prob_bars,
                    gradcam_image,
                    cam_desc,
                    similar_pills_card,
                    alt_rec_card,
                    guide_md,
                ],
            )

        analyze_btn.click(
            fn=handle_analyze,
            inputs=[memory_state, imprint_input, color_input, shape_input],
            outputs=[
                warning_banner,
                risk_badge,
                prob_bars,
                gradcam_image,
                cam_desc,
                similar_pills_card,
                alt_rec_card,
                guide_md,
            ],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[
                file_input,
                memory_state,
                preview_image,
                file_status,
                warning_banner,
                risk_badge,
                prob_bars,
                gradcam_image,
                cam_desc,
                similar_pills_card,
                alt_rec_card,
                guide_md,
                imprint_input,
                color_input,
                shape_input,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    default_host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    server_name = os.getenv("SERVER_NAME", default_host)
    app.launch(
        server_name=server_name,
        server_port=port,
        inbrowser=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(primary_hue="neutral", neutral_hue="neutral"),
    )
