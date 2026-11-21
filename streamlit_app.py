"""PLAY TRUE AI - Nike MD Editorial Style Streamlit Web UI for WADA Doping Screening."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path
from typing import Optional

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "playtrue_mpl"))

import streamlit as st
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

SAMPLE_MAP = {
    "high_risk": ("sample_high_risk.png", SAMPLES_DIR / "sample_high_risk.png"),
    "caution": ("sample_caution.png", SAMPLES_DIR / "sample_caution.png"),
    "safe": ("sample_safe.png", SAMPLES_DIR / "sample_safe.png"),
}

NIKE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+KR:wght@400;500;600;700;900&display=swap');
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"], .stApp {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.block-container {
    max-width: 1240px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
}

.nike-utility-strip {
    background: #F5F5F7;
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #39393B;
    border-bottom: 1px solid #E5E5EA;
}

.nike-hero-header {
    background: #FFFFFF;
    padding: 22px 0 18px 0;
    border-bottom: 2px solid #111111;
    margin-bottom: 22px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 14px;
}

.nike-eyebrow {
    font-family: 'Bebas Neue', 'Impact', 'Arial Black', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #707072;
    margin-bottom: 4px;
}

.nike-display-title {
    font-family: 'Bebas Neue', 'Impact', 'Arial Black', sans-serif;
    font-size: 3.1rem;
    font-weight: 900;
    line-height: 0.92;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: #111111;
    margin: 0;
}

.nike-subtitle {
    font-size: 0.96rem;
    font-weight: 500;
    color: #39393B;
    margin: 10px 0 0 0;
}

.nike-header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F5F5F7;
    color: #111111;
    padding: 8px 16px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
}

.nike-step-label {
    font-family: 'Bebas Neue', 'Impact', 'Arial Black', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #111111;
    margin: 0 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #E5E5EA;
}

.stButton > button[kind="primary"] {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9999px !important;
    font-weight: 700 !important;
    min-height: 46px !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #0EA5E9 !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="secondary"] {
    background-color: #F5F5F7 !important;
    color: #111111 !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 9999px !important;
    font-weight: 600 !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    border-color: #111111 !important;
}

.disclaimer-footnote {
    margin-top: 24px;
    padding: 16px 20px;
    background: #F5F5F7;
    border-top: 2px solid #111111;
    font-size: 0.8rem;
    line-height: 1.55;
    color: #39393B;
}
</style>
"""


@st.cache_resource(show_spinner="AI 모델을 불러오는 중입니다...")
def get_predictor() -> DopingPredictor:
    """Load the trained EfficientNet-B0 doping classifier once per server process."""
    if not all(
        (SAMPLES_DIR / name).exists()
        for name in ("sample_high_risk.png", "sample_caution.png", "sample_safe.png")
    ):
        generate_mvp_pill_dataset()
    return DopingPredictor(weights_path=DEFAULT_WEIGHTS_PATH)


def render_badge_html(result: Optional[PredictionResult]) -> str:
    """Render Nike MD risk badge card or ready placeholder."""
    if result is None:
        return (
            "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:18px 20px;margin-bottom:12px;'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>"
            "<span style='display:inline-block;padding:5px 14px;border-radius:9999px;"
            "background:#111111;color:#FFFFFF;font-size:0.78rem;font-weight:800;letter-spacing:0.06em;'>"
            "READY TO SCREEN</span>"
            "<span style='font-size:0.95rem;font-weight:700;color:#111111;'>외형 유사도 점수 --%</span>"
            "</div>"
            "<div style='margin-top:10px;font-size:0.86rem;color:#52525B;'>"
            "좌측 패널에서 알약 이미지를 업로드하거나 샘플을 선택하면 즉시 분석이 실행됩니다."
            "</div></div>"
        )

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
        "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:18px 20px;margin-bottom:12px;'>"
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


def render_probs_html(result: Optional[PredictionResult]) -> str:
    """Render 3-tier visual similarity segmented bar."""
    if result is None:
        return (
            "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:18px 20px;margin-bottom:12px;'>"
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

    p_high = result.probabilities["High Risk"] * 100.0
    p_caution = result.probabilities["Caution"] * 100.0
    p_check = result.probabilities["Safe"] * 100.0

    return (
        "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:18px 20px;margin-bottom:12px;'>"
        "<div style='font-size:0.8rem;font-weight:800;letter-spacing:0.06em;"
        "text-transform:uppercase;color:#111111;margin-bottom:10px;'>"
        "3단계 외형 유사도 점수</div>"
        "<div style='display:flex;height:8px;border-radius:9999px;background:#F5F5F7;overflow:hidden;'>"
        f"<div style='width:{p_high:.2f}%;background:#E11D48;'></div>"
        f"<div style='width:{p_caution:.2f}%;background:#D97706;'></div>"
        f"<div style='width:{p_check:.2f}%;background:#0284C7;'></div>"
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


def render_alt_card_html(result: Optional[PredictionResult]) -> str:
    """Render WADA alternative / OTC recommendation card."""
    if result is None:
        return (
            "<div style='background:#FFFFFF;border:1px solid #E5E5EA;padding:18px 20px;margin-top:12px;'>"
            "<div style='font-size:0.8rem;font-weight:800;letter-spacing:0.06em;"
            "text-transform:uppercase;color:#111111;margin-bottom:6px;'>"
            "WADA 대체 약품 · 허용 성분 참고</div>"
            "<div style='font-size:0.86rem;color:#52525B;line-height:1.55;'>"
            "분석 결과 <strong>High Risk</strong> 또는 <strong>Caution</strong> 판정 시 "
            "해당 WADA 세부 코드(S0~S9, P1 또는 추정군), 금지 범위, TUE 안내 및 허용 성분/일반의약품 참고 카드가 표시됩니다."
            "</div></div>"
        )

    if result.final_class not in ("High Risk", "Caution"):
        return (
            "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:18px 20px;margin-top:12px;'>"
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
        "<div style='background:#F5F5F7;border:1px solid #E5E5EA;padding:20px;margin-top:12px;'>"
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


def _run_inference(raw_bytes: bytes, filename: str) -> Optional[PredictionResult]:
    try:
        predictor = get_predictor()
        return predictor.predict_from_bytes(
            raw_bytes=raw_bytes,
            filename=filename,
            imprint_query=str(st.session_state.get("imprint_filter", "") or ""),
            color_filter=str(st.session_state.get("color_filter", "전체") or "전체"),
            shape_filter=str(st.session_state.get("shape_filter", "전체") or "전체"),
        )
    except InvalidImageFormatError as exc:
        st.session_state["warning_msg"] = str(exc)
        return None


def main() -> None:
    st.set_page_config(
        page_title="PLAY TRUE AI · CLEAN SPORT SCREENING",
        page_icon="🛡️",
        layout="wide",
    )
    st.markdown(NIKE_CSS, unsafe_allow_html=True)

    for key, default in (
        ("payload", None),
        ("preview_img", None),
        ("file_status", "선택된 파일 없음"),
        ("warning_msg", ""),
        ("result", None),
        ("uploader_nonce", 0),
    ):
        st.session_state.setdefault(key, default)

    st.markdown(
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
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["warning_msg"]:
        st.warning(st.session_state["warning_msg"], icon="⚠️")

    left_col, right_col = st.columns([5, 7], gap="large")

    with left_col:
        st.markdown("<div class='nike-step-label'>01. IMAGE INPUT</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            f"의약품 낱알 이미지 업로드 ({SUPPORTED_FORMATS_TEXT})",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"pill_uploader_{st.session_state['uploader_nonce']}",
        )
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            filename = uploaded.name
            if (
                st.session_state["payload"] is None
                or st.session_state["payload"].get("filename") != filename
                or st.session_state["payload"].get("bytes") != raw_bytes
            ):
                try:
                    rgb_np = validate_and_decode_bytes(raw_bytes, filename=filename)
                    st.session_state["payload"] = {"bytes": raw_bytes, "filename": filename}
                    st.session_state["preview_img"] = Image.fromarray(rgb_np)
                    st.session_state["file_status"] = f"📄 활성 파일: `{filename}` (메모리 적재 · 디스크 미저장)"
                    st.session_state["warning_msg"] = ""
                    st.session_state["result"] = None
                except InvalidImageFormatError as exc:
                    st.session_state["payload"] = None
                    st.session_state["preview_img"] = None
                    st.session_state["file_status"] = "선택된 파일 없음"
                    st.session_state["warning_msg"] = str(exc)
                    st.session_state["result"] = None

        st.caption(st.session_state["file_status"])

        if st.session_state["preview_img"] is not None:
            st.image(st.session_state["preview_img"], caption="업로드 약물 미리보기", use_container_width=True)

        st.markdown(
            "<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.08em;color:#707072;margin:8px 0 6px 0;'>SAMPLE PILLS</div>",
            unsafe_allow_html=True,
        )
        s_cols = st.columns(3)
        sample_defs = (
            ("high_risk", "High Risk · 샘플"),
            ("caution", "Caution · 샘플"),
            ("safe", "Check Required · 샘플"),
        )
        for col, (s_key, s_label) in zip(s_cols, sample_defs):
            with col:
                if st.button(s_label, key=f"btn_{s_key}", type="secondary", use_container_width=True):
                    fname, spath = SAMPLE_MAP[s_key]
                    s_bytes = spath.read_bytes()
                    rgb_np = validate_and_decode_bytes(s_bytes, filename=fname)
                    st.session_state["payload"] = {"bytes": s_bytes, "filename": fname}
                    st.session_state["preview_img"] = Image.fromarray(rgb_np)
                    st.session_state["file_status"] = f"🧪 샘플 선택됨: `{fname}` (메모리 적재 완료)"
                    st.session_state["warning_msg"] = ""
                    st.session_state["result"] = _run_inference(s_bytes, fname)
                    st.rerun()

        if st.button("도핑 위험도 분석 실행", type="primary", use_container_width=True):
            payload = st.session_state["payload"]
            if not payload or not payload.get("bytes"):
                st.session_state["warning_msg"] = EMPTY_INPUT_WARNING
                st.rerun()
            else:
                st.session_state["warning_msg"] = ""
                st.session_state["result"] = _run_inference(
                    payload["bytes"], payload.get("filename", "upload.png")
                )
                st.rerun()

        if st.button("초기화", type="secondary", use_container_width=True):
            st.session_state["payload"] = None
            st.session_state["preview_img"] = None
            st.session_state["file_status"] = "선택된 파일 없음"
            st.session_state["warning_msg"] = ""
            st.session_state["result"] = None
            st.session_state["imprint_filter"] = ""
            st.session_state["color_filter"] = "전체"
            st.session_state["shape_filter"] = "전체"
            st.session_state["uploader_nonce"] += 1
            st.rerun()

        with st.expander("선택 입력: 알약 각인 문자 / 색상 / 모양 교차 확인", expanded=False):
            st.text_input(
                "표면 각인 문자 (예: TY 500, PSE, YH)",
                key="imprint_filter",
                placeholder="예: TY 500",
            )
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.selectbox("색상", PILL_COLOR_OPTIONS, key="color_filter")
            with f_col2:
                st.selectbox("모양", PILL_SHAPE_OPTIONS, key="shape_filter")

    with right_col:
        st.markdown("<div class='nike-step-label'>02. SCREENING RESULT</div>", unsafe_allow_html=True)
        result: Optional[PredictionResult] = st.session_state["result"]

        st.markdown(render_badge_html(result), unsafe_allow_html=True)

        top3_title = "🔍 AI 외형·각인 매칭 후보 (TOP 3)"
        if result is not None and result.similar_pills:
            top1_name = str(result.similar_pills[0].get("item_name", "")).strip()
            top1_sim = float(result.similar_pills[0].get("similarity_pct", 0.0))
            if top1_name:
                top3_title = f"🔍 AI 외형·각인 매칭 후보 (TOP 3) · 1위 {top1_name} ({top1_sim:.1f}%)"

        with st.expander(top3_title, expanded=False):
            st.markdown(
                render_similar_pills_card_html(result.similar_pills if result else None),
                unsafe_allow_html=True,
            )

        with st.expander("💊 WADA 세부 코드 · 대체 성분 및 일반의약품(OTC) 안내", expanded=False):
            st.markdown(render_alt_card_html(result), unsafe_allow_html=True)

        with st.expander("🎯 Grad-CAM 판단 근거 시각화 및 3단계 예측 확률", expanded=False):
            st.markdown(render_probs_html(result), unsafe_allow_html=True)
            if result is not None:
                st.image(
                    result.heatmap_image,
                    caption="Grad-CAM 판단 근거 히트맵 (원본 비율 유지)",
                    use_container_width=True,
                )
                st.caption(result.cam_explanation)
            else:
                st.caption("분석 실행 시 모델이 주목한 각인·분할선·제형 윤곽 히트맵이 표시됩니다.")

        with st.expander("📋 등급별 맞춤형 복용 주의 가이드", expanded=False):
            if result is not None:
                st.markdown(result.guide_text)
            else:
                st.markdown("- 분석 실행 후 등급별 맞춤형 복용 주의 가이드가 표시됩니다.")

    st.markdown(
        f"<footer class='disclaimer-footnote'><strong>[면책 및 교차 확인 필수]</strong> {html.escape(DISCLAIMER_TEXT)}</footer>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
