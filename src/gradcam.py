"""Grad-CAM heatmap generation and imprint/contour activation explanation module."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import IMAGE_SIZE
from src.model import get_gradcam_target_layer


def _describe_activation(grayscale_cam: np.ndarray) -> str:
    """Summarize whether Grad-CAM concentrates on pill center (imprint/score) vs edges."""
    h, w = grayscale_cam.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)

    center_mask = dist <= 0.45
    contour_mask = (dist > 0.45) & (dist <= 0.85)

    total = float(grayscale_cam.sum()) + 1e-8
    center_ratio = float(grayscale_cam[center_mask].sum()) / total
    contour_ratio = float(grayscale_cam[contour_mask].sum()) / total
    pill_focus = min(99.0, (center_ratio + contour_ratio) * 100.0)

    if center_ratio >= 0.42:
        focus_detail = "알약 중앙 각인 문자·분할선 영역에 핵심 가중치가 집중되었습니다."
    elif contour_ratio >= 0.42:
        focus_detail = "알약 고유 제형 윤곽·색상 경계 영역을 중심으로 특징을 포착했습니다."
    else:
        focus_detail = "알약 본체의 각인·분할선과 외형 윤곽을 균형 있게 참조했습니다."

    return (
        f"{focus_detail} "
        f"(알약 본체 영역 활성 비율: {pill_focus:.1f}% · "
        f"중앙 각인/분할선 {center_ratio * 100:.1f}% / 제형 윤곽 {contour_ratio * 100:.1f}%)"
    )


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    target_class_idx: int,
    backbone: str = "efficientnet_b0",
    image_size: int = IMAGE_SIZE,
) -> Tuple[np.ndarray, str]:
    """Compute Grad-CAM heatmap on the final conv layer and overlay on 224x224 RGB image."""
    target_layer = get_gradcam_target_layer(model, backbone=backbone)
    resized_rgb = cv2.resize(original_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb_float = np.float32(resized_rgb) / 255.0

    # Ensure gradients are enabled for Grad-CAM even inside inference helpers.
    with torch.enable_grad():
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            targets = [ClassifierOutputTarget(target_class_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=0.55)
    explanation = _describe_activation(grayscale_cam)
    return overlay, explanation
