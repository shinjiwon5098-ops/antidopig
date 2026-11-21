"""Shared constants, WADA risk class metadata, WADA code alternatives, and business rules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
INTRO_VIDEO_PATH = BASE_DIR.parent / "PLAT TRUE(인트로).mp4"
PROJECT_ROOT_DIR = BASE_DIR.parent
MAPPED_PILL_CSV_PATH = PROJECT_ROOT_DIR / "pill_doping_mapped.csv"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "best_model.pt"

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Class order must stay consistent across dataset, model, and inference.
CLASS_NAMES = ["High Risk", "Caution", "Safe"]
CLASS_DIR_NAMES = {
    "High Risk": "High_Risk",
    "Caution": "Caution",
    "Safe": "Safe",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_FORMATS_TEXT = "JPG, PNG, WEBP"

# PRD Section 2.5: If P(High Risk) >= 35%, escalate warning to at least Caution.
HIGH_RISK_CONSERVATIVE_THRESHOLD = 0.35

APP_SUBTITLE = "AI 기반 의약품 외형 식별 보조 및 WADA 금지성분 확인 지원 시스템"

EMPTY_INPUT_WARNING = (
    "분석할 의약품 또는 보충제 이미지를 먼저 업로드해 주세요."
)

INVALID_FORMAT_WARNING = (
    f"지원하지 않는 파일 형식입니다. 지원 이미지 형식({SUPPORTED_FORMATS_TEXT}) 파일만 업로드해 주세요."
)

DISCLAIMER_TEXT = (
    "본 결과는 AI 예측 모델의 외형 기반 보조 분석 결과로, 실제 WADA/KADA 도핑 규정 준수 여부를 "
    "최종 보장하지 않습니다. 동일 외형 복제약(제네릭) 존재 가능성 및 복용 전 KADA·전문의 교차 확인 필수입니다. "
    "실제 복용 전 반드시 KADA(한국도핑방지위원회) 의약품 검색 서비스 또는 스포츠 의학 전문의와 상의하십시오."
)

CLASS_META = {
    "High Risk": {
        "title": "High Risk (복용 금지)",
        "short_label": "복용 금지",
        "badge_color": "#E11D48",
        "bg_color": "#FFF1F2",
        "border_color": "#FECDD3",
        "wada_ref": (
            "WADA 상시 금지약물 추정군 (예: S1. 동화작용제, S2. 펩타이드 호르몬·성장인자, "
            "S4. 호르몬 및 대사 변조제, S5. 이뇨제·은폐제)"
        ),
        "guide": (
            "### 🚫 복용 금지 (High Risk)\n"
            "- **즉각 조치**: 해당 의약품의 임의 복용을 **즉시 중단**하십시오. "
            "경기 기간 여부와 무관하게 **상시 금지약물**군에 해당할 가능성이 높습니다.\n"
            "- **선수/지도자 안내**: 치료 목적이 불가피한 경우 복용 전 반드시 "
            "전문의 처방 변경 상담 및 **치료목적사용면책(TUE)** 신청·승인 여부를 확인하십시오.\n"
            "- **최종 검증**: KADA 의약품 검색 서비스에서 실제 제품명·낱알 각인·성분명을 재확인하시기 바랍니다."
        ),
    },
    "Caution": {
        "title": "Caution (주의 · 전문가 확인 필요)",
        "short_label": "주의 · 전문가 확인",
        "badge_color": "#D97706",
        "bg_color": "#FFFBEB",
        "border_color": "#FDE68A",
        "wada_ref": (
            "경기기간 중 금지 또는 투여 경로·용량·특정 종목 제한 약물 "
            "(예: S6. 흥분제, S7. 마약성 진통제, S9. 당질코르티코이드, P1. 베타차단제)"
        ),
        "guide": (
            "### ⚠️ 주의 · 전문가 확인 필요 (Caution)\n"
            "- **즉각 조치**: 임의 복용을 보류하고 **스포츠 의학 전문의 또는 KADA 상담**을 먼저 진행하십시오.\n"
            "- **주의 포인트**: 감기약·비염약·진통제·항염증제 등은 **경기기간 중 금지** 또는 "
            "**투여 경로·허용 용량 제한** 규정이 적용될 수 있습니다.\n"
            "- **최종 검증**: 대회 일정(경기기간 개시 시점)과 체내 배설 반감기를 반드시 고려하고 대체 허용 성분을 확인하십시오."
        ),
    },
    "Safe": {
        "title": "Check Required (외형 기준 금지성분 미검출 · 추가 확인 필요)",
        "short_label": "외형 기준 금지성분 미검출 · 추가 확인 필요",
        "badge_color": "#0284C7",
        "bg_color": "#F0F9FF",
        "border_color": "#BAE6FD",
        "wada_ref": (
            "AI 외형 분석 기준 WADA 금지성분 위험 신호 미검출 "
            "(동일 외형 복제약·복합제 가능성으로 KADA 성분 대조 필수)"
        ),
        "guide": (
            "### 🔎 Check Required (외형 기준 금지성분 미검출 · 추가 확인 필요)\n"
            "- **판정 요약**: AI 외형 유사도 분석 기준 WADA 금지성분 고위험 신호는 검출되지 않았습니다.\n"
            "- **교차 확인 필수**: 동일 외형의 복제약(제네릭)이나 복합제(예: 카페인·에페드린·코데인 첨가)일 가능성이 있으므로 "
            "복용 전 처방전·포장지 성분표와 **KADA 의약품 검색 서비스**를 반드시 대조하십시오.\n"
            "- **복용 유의**: 타인의 약물과 혼입되지 않도록 원포장 상태를 유지하고 전문의·약사와 최종 확인 후 복용하십시오."
        ),
    },
}

OTC_RECOMMENDATION_NOTICE = (
    "※ 제시된 일반의약품은 동일 효능군 내 WADA 허용 단일성분 참고 예시이며, "
    "구매 시 약사 및 KADA 의약품 검색으로 최종 확인하십시오."
)

# WADA Prohibited List detail mapping (S0~S9, P1 + estimated fallback groups)
WADA_CODE_ALTERNATIVES: Dict[str, Dict[str, object]] = {
    "S0": {
        "category_name": "승인되지 않은 물질 (Non-approved substances)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "인체 치료 승인을 받지 않은 물질은 TUE 승인 대상에서 제외되는 경우가 많으므로 즉시 복용 중단 및 KADA·전문의 확인이 필수입니다.",
        "alternatives": [
            "식약처(MFDS) 정식 품목허가를 받은 동일 질환 치료 의약품으로 전문의 처방 변경 상담 필수",
            "출처 불명의 보충제·연구용 펩타이드·미허가 제제 즉시 중단",
        ],
    },
    "S1": {
        "category_name": "동화작용제 (Anabolic Agents)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 내분비내과/스포츠의학 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
            "비호르몬성 보존 치료 또는 허용된 영양·재활 요법 여부 주치의 상의",
        ],
    },
    "S1.1": {
        "category_name": "동화작용 남성호르몬 스테로이드 (AAS)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 사전 승인 필수",
            "성선기능저하증 등 명확한 의학적 적응증 외 복용 절대 금지",
        ],
    },
    "S1.2": {
        "category_name": "기타 동화작용제 (SARMs·클렌부테롤 등)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품 또는 미승인 동화작용제로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE 신청 가능 여부 확인 필수.",
        "alternatives": [
            "임의 대체 불가 — 즉시 복용 중단 후 전문의 처방 변경 상담 및 TUE 신청 여부 확인 필수",
            "기관지 확장 목적인 경우 허용 한도 내 흡입제 등 전문의와 대체 약제 상의",
        ],
    },
    "S2": {
        "category_name": "펩타이드 호르몬·성장인자 및 관련 물질",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 내분비/혈액내과 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
            "빈혈 등 치료 목적인 경우 철분제 등 WADA 허용 보충·치료법 가능 여부 주치의 상의",
        ],
    },
    "S2.1": {
        "category_name": "에리스로포이에틴(EPO) 및 조혈 영향 물질",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
            "경증 빈혈 관리 시 경구·주사 철분제(금지 제외) 적용 가능 여부 주치의 상담",
        ],
    },
    "S2.2": {
        "category_name": "펩타이드 호르몬 및 방출 인자 (성장호르몬 등)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
        ],
    },
    "S2.3": {
        "category_name": "성장 인자 및 조절제",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
        ],
    },
    "S3": {
        "category_name": "베타-2 작용제 (기관지 확장제 등)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "경구제·주사제는 상시 금지이며, 흡입 살부타몰·포르모테롤·살메테롤·빌란테롤 등은 WADA 허용 용량 한도 내에서만 예외 적용됩니다. 한도 초과 또는 경구제는 전문의 상담 및 TUE 신청 필수.",
        "alternatives": [
            "호흡기 전문의 상담 하에 WADA 허용 용량 한도 내 흡입 베타-2 작용제(예: 흡입 살부타몰·포르모테롤) 또는 흡입 스테로이드(ICS) 전환 검토",
            "경구/전신 베타-2 작용제(예: 툴로부테롤 패치/경구 등)는 복용 중단 후 전문의 처방 변경 및 TUE 신청 필수",
        ],
    },
    "S4": {
        "category_name": "호르몬 및 대사 변조제",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
        ],
    },
    "S4.1": {
        "category_name": "아로마타제 억제제",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 종양/내분비 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
        ],
    },
    "S4.2": {
        "category_name": "항에스트로겐 물질 (SERMs·SERDs)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
        ],
    },
    "S4.3": {
        "category_name": "액티빈 수용체 IIB 활성 차단제",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
        ],
    },
    "S4.4": {
        "category_name": "대사 변조제 (인슐린·트리메타지딘 등)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 당뇨(인슐린) 또는 심혈관 질환 치료 시 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
        ],
    },
    "S5": {
        "category_name": "이뇨제 및 은폐제 (Diuretics and Masking Agents)",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "전문의약품으로 임의 대체가 불가하므로 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수.",
        "alternatives": [
            "임의 대체 불가 — 고혈압·부종 목적 복용 시 전문의와 비이뇨제 계열(예: CCB·ARB 등 WADA 비금지 혈압약) 전환 상담 또는 TUE 신청 필수",
            "안과용 국소 탄산탈수효소억제제 등 예외 규정 여부 KADA 재확인",
        ],
    },
    "S6": {
        "category_name": "흥분제 (Stimulants · 감기약/비염약 성분 포함)",
        "prohibition_scope": "경기 기간 중 금지",
        "prescription_only": False,
        "tue_notice": "경기 기간(In-Competition) 중 금지됩니다. 경기 직전·당일 복용 시 반감기를 고려한 중단 또는 전문의 상담·TUE 신청이 필요합니다.",
        "alternatives": [
            "발열·몸살·두통: 아세트아미노펜 단일제(예: 타이레놀 단일 성분), 이부프로펜·나프록센 등 일반 NSAIDs (복합감기약 내 슈도에페드린·메틸에페드린 혼입 주의)",
            "콧물·재채기·비염: 2세대 항히스타민제(세티리진, 로라타딘, 펙소페나딘 등) 및 생리식염수 비강 세척",
            "기침·가래: 덱스트로메토르판·아세틸시스테인·암브록솔 등 비금지 단일 성분 (KADA 의약품 검색 재확인 권장)",
        ],
        "otc_examples": [
            {
                "label": "S6(감기·코막힘)",
                "ingredients": ["아세트아미노펜", "세티리진", "로라타딘"],
                "products": [
                    "타이레놀정 500mg(한국얀센)",
                    "지르텍정(한국유씨비)",
                    "클라리틴정(바이엘)",
                ],
            },
        ],
    },
    "S6.A": {
        "category_name": "비지정 흥분제 (Stimulants)",
        "prohibition_scope": "경기 기간 중 금지",
        "prescription_only": True,
        "tue_notice": "경기 기간 중 엄격히 금지되며, ADHD/기면증 등 치료 목적 시 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 사전 승인이 필수입니다.",
        "alternatives": [
            "ADHD 등 전문치료 목적: 임의 중단/대체 불가 — 전문의 처방 변경 상담(비흥분제 계열 아토목세틴 등 검토) 및 TUE 신청 필수",
        ],
    },
    "S6.B": {
        "category_name": "지정 흥분제 (감기약·비염약 함유 슈도에페드린·에페드린 등)",
        "prohibition_scope": "경기 기간 중 금지",
        "prescription_only": False,
        "tue_notice": "종합감기약·코감기약·한방제제에 빈번히 포함되어 비의도적 도핑이 잦은 군입니다. 경기 기간 중 금지이므로 대체 허용 성분 사용 또는 전문의·TUE 상담이 필요합니다.",
        "alternatives": [
            "발열·통증 완화: 아세트아미노펜 단일제, 이부프로펜·덱시부프로펜 등 NSAIDs 단일제",
            "코막힘·비염 완화: 2세대 항히스타민제(로라타딘, 세티리진, 펙소페나딘), 옥시메타졸린 등 허용 국소 점비제 여부 KADA 확인",
            "생약·한방 감기약(마황=에페드린 함유) 및 복합 감기약 복용 전 성분표 교차 확인 필수",
        ],
        "otc_examples": [
            {
                "label": "S6(감기·코막힘)",
                "ingredients": ["아세트아미노펜", "세티리진", "로라타딘"],
                "products": [
                    "타이레놀정 500mg(한국얀센)",
                    "지르텍정(한국유씨비)",
                    "클라리틴정(바이엘)",
                ],
            },
        ],
    },
    "S7": {
        "category_name": "마약류 진통제 (Narcotics · 트라마돌 포함)",
        "prohibition_scope": "경기 기간 중 금지",
        "prescription_only": True,
        "tue_notice": "경기 기간(In-Competition) 중 금지입니다. 특히 트라마돌(2024년 이후 경기기간 중 금지) 등 처방 진통제는 전문의 처방 변경 상담 및 TUE 신청이 필요합니다.",
        "alternatives": [
            "경증·중등도 통증 대체 참고: 아세트아미노펜 단일제, 이부프로펜·나프록센·셀레콕시브 등 일반 비마약성 소염진통제(NSAIDs)",
            "수술·중증 통증으로 처방된 경우 임의 변경 불가 — 주치의와 허용 진통제 전환 상담 및 필요 시 TUE 신청 필수",
        ],
        "otc_examples": [
            {
                "label": "S7(통증·소염)",
                "ingredients": ["이부프로펜", "덱시부프로펜", "나프록센"],
                "products": [
                    "애드빌정(화이자)",
                    "탁센연질캡슐(GC녹십자)",
                ],
            },
        ],
    },
    "S8": {
        "category_name": "카나비노이드 (Cannabinoids)",
        "prohibition_scope": "경기 기간 중 금지",
        "prescription_only": True,
        "tue_notice": "경기 기간 중 금지(칸나비디올 CBD는 제외되나 미량 THC 오염 위험 존재). 전문의 상담이 필수입니다.",
        "alternatives": [
            "통증·불안 관리 목적 시 전문의와 WADA 허용 표준 치료제(비마약성 진통제 등) 전환 상담 필수",
            "CBD 표기 보충제라도 THC 잔류 검출 위험이 있으므로 인증되지 않은 제품 섭취 지양",
        ],
    },
    "S9": {
        "category_name": "당질코르티코이드 (Glucocorticoids · 염증/알레르기약)",
        "prohibition_scope": "경기 기간 중 금지",
        "prescription_only": True,
        "tue_notice": "경기 기간 중 모든 주사·경구(구강점막 포함)·직장 투여가 금지됩니다. 경기 전 체내 배설(Washout) 기간 준수 또는 전문의 처방 변경 및 TUE 신청이 필요합니다.",
        "alternatives": [
            "알레르기·두드러기 참고: 2세대 항히스타민제(세티리진, 로라타딘, 펙소페나딘 등 경구제)",
            "근골격계 통증·염증 참고: 아세트아미노펜 단일제, 경구/외용 NSAIDs(이부프로펜, 디클로페낙 겔 등)",
            "피부·비강 등 국소 외용제(연고·점안액·비강분무제)는 허용 경로이나, 전신 스테로이드 처방 시 반드시 전문의 상담 및 TUE 신청 필수",
        ],
        "otc_examples": [
            {
                "label": "S9(통증·소염)",
                "ingredients": ["이부프로펜", "덱시부프로펜", "나프록센"],
                "products": [
                    "애드빌정(화이자)",
                    "탁센연질캡슐(GC녹십자)",
                ],
            },
        ],
    },
    "P1": {
        "category_name": "베타차단제 (Beta-blockers · 특정 종목 금지)",
        "prohibition_scope": "특정 종목 금지 (경기 기간 중 금지, 일부 종목 상시 금지)",
        "prescription_only": True,
        "tue_notice": "양궁·사격·골프·자동차경주·당구·다트·수중스포츠 등 특정 종목에서 금지됩니다. 전문의약품이므로 임의 중단 금지 및 처방 변경·TUE 상담이 필수입니다.",
        "alternatives": [
            "임의 복용 중단 금지 — 심혈관/고혈압/편두통 치료 시 전문의와 비금지 계열(CCB·ARB 등) 처방 전환 상담 및 TUE 신청 필수",
            "본인 종목이 WADA P1 금지 대상 종목에 해당하는지 KADA 종목별 규정 교차 확인",
        ],
    },
    "S1~S5": {
        "category_name": "S1~S5 상시 금지군 추정",
        "prohibition_scope": "항시 금지",
        "prescription_only": True,
        "tue_notice": "이미지 단독 추론으로 세부 성분이 특정되지 않아 상시 금지 추정군(S1~S5)으로 분류되었습니다. 전문의약품 가능성이 높으므로 임의 대체 불가하며 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수 여부를 확인하십시오.",
        "alternatives": [
            "임의 대체 불가 — 즉시 복용을 보류하고 전문의 처방 변경 상담 및 TUE(치료목적사용면책) 신청 필수",
            "KADA 의약품 검색 서비스에 실제 제품명 또는 낱알 각인을 입력해 정확한 WADA 세부 코드(S1~S5)를 확인하십시오.",
        ],
    },
    "S6~S9": {
        "category_name": "S6~S9 경기기간 중 주의군 추정",
        "prohibition_scope": "경기 기간 중 금지",
        "prescription_only": False,
        "tue_notice": "이미지 단독 추론으로 세부 성분이 특정되지 않아 경기기간 중 주의 추정군(S6~S9)으로 분류되었습니다. 감기약·진통제·항염증 스테로이드 등일 가능성이 있으므로 경기 일정 확인 및 전문의·KADA 교차 확인이 필수입니다.",
        "alternatives": [
            "감기·해열·통증 참고(허용 성분): 아세트아미노펜 단일제, 이부프로펜·나프록센 등 일반 NSAIDs 단일제",
            "알레르기·콧물 참고(허용 성분): 2세대 항히스타민제(세티리진, 로라타딘, 펙소페나딘)",
            "전문의 처방약(트라마돌·스테로이드 등)일 경우 임의 대체 불가 — 전문의 처방 변경 상담 및 필요 시 TUE 신청 필수",
        ],
        "otc_examples": [
            {
                "label": "S6(감기·코막힘)",
                "ingredients": ["아세트아미노펜", "세티리진", "로라타딘"],
                "products": [
                    "타이레놀정 500mg(한국얀센)",
                    "지르텍정(한국유씨비)",
                    "클라리틴정(바이엘)",
                ],
            },
            {
                "label": "S7/S9(통증·소염)",
                "ingredients": ["이부프로펜", "덱시부프로펜", "나프록센"],
                "products": [
                    "애드빌정(화이자)",
                    "탁센연질캡슐(GC녹십자)",
                ],
            },
        ],
    },
}


def get_otc_examples(wada_code: str | None) -> list[dict[str, object]]:
    """Return OTC product examples only for eligible In-Competition codes (S6/S7/S9), disabled for S1~S5."""
    if not wada_code:
        return []
    normalized = wada_code.strip().upper()
    entry = WADA_CODE_ALTERNATIVES.get(normalized)
    if entry is None:
        base = normalized.split(".", 1)[0]
        entry = WADA_CODE_ALTERNATIVES.get(base)
    if not entry:
        return []
    return list(entry.get("otc_examples", []))


def resolve_wada_alternative_entry(wada_code: str | None, final_class: str) -> tuple[str, Dict[str, object]]:
    """Resolve exact WADA code metadata or fall back safely to estimated S1~S5 / S6~S9 group."""
    if wada_code:
        normalized = wada_code.strip().upper()
        if normalized in WADA_CODE_ALTERNATIVES:
            return normalized, WADA_CODE_ALTERNATIVES[normalized]
        # Handle subcode fallback (e.g., S6.1 -> S6, S4.x -> S4)
        base_code = normalized.split(".", 1)[0]
        if base_code in WADA_CODE_ALTERNATIVES:
            return base_code, WADA_CODE_ALTERNATIVES[base_code]

    fallback_code = "S1~S5" if final_class == "High Risk" else "S6~S9"
    return fallback_code, WADA_CODE_ALTERNATIVES[fallback_code]
