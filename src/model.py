"""timm-based 3-class doping risk classification model architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn

from src.config import CLASS_NAMES, DEFAULT_WEIGHTS_PATH

SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet50")


def create_doping_classifier(
    backbone: str = "efficientnet_b0",
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> nn.Module:
    """Build a timm image classification backbone with a 3-class head."""
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from {SUPPORTED_BACKBONES}."
        )
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


def get_gradcam_target_layer(model: nn.Module, backbone: str = "efficientnet_b0") -> nn.Module:
    """Return the final convolutional block for Grad-CAM spatial localization."""
    if backbone == "efficientnet_b0":
        return model.conv_head
    if backbone == "resnet50":
        return model.layer4[-1]
    raise ValueError(f"Unsupported backbone '{backbone}' for Grad-CAM.")


def load_trained_model(
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
    device: str | torch.device = "cpu",
    fallback_backbone: str = "efficientnet_b0",
) -> Tuple[nn.Module, Dict]:
    """Load trained checkpoint weights, or initialize a pretrained backbone if missing."""
    path = Path(weights_path)
    device_obj = torch.device(device)

    if path.exists():
        checkpoint = torch.load(path, map_location=device_obj, weights_only=False)
        backbone = checkpoint.get("backbone", fallback_backbone)
        model = create_doping_classifier(
            backbone=backbone,
            num_classes=len(CLASS_NAMES),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device_obj)
        model.eval()
        meta = {
            "backbone": backbone,
            "loaded_weights": str(path),
            "is_trained": True,
            "metrics": checkpoint.get("metrics", {}),
        }
        return model, meta

    model = create_doping_classifier(
        backbone=fallback_backbone,
        num_classes=len(CLASS_NAMES),
        pretrained=True,
    )
    model.to(device_obj)
    model.eval()
    meta = {
        "backbone": fallback_backbone,
        "loaded_weights": None,
        "is_trained": False,
        "metrics": {},
    }
    return model, meta
