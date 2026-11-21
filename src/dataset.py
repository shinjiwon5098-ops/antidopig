"""Image validation, Albumentations preprocessing, PyTorch Dataset, and MVP sample generator."""

from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError
from torch.utils.data import Dataset

from src.config import (
    CLASS_DIR_NAMES,
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SAMPLES_DIR,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS_TEXT,
    SUPPORTED_PIL_FORMATS,
)


class InvalidImageFormatError(ValueError):
    """Raised when an uploaded file is not a supported or readable image."""


def validate_and_decode_bytes(
    raw_bytes: bytes, filename: Optional[str] = None
) -> np.ndarray:
    """Validate format and decode raw image bytes strictly in memory (RGB uint8)."""
    if not raw_bytes:
        raise InvalidImageFormatError(
            f"파일이 비어 있습니다. 지원 이미지 형식({SUPPORTED_FORMATS_TEXT})을 업로드해 주세요."
        )

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise InvalidImageFormatError(
                f"지원하지 않는 파일 형식({ext})입니다. "
                f"지원 이미지 형식({SUPPORTED_FORMATS_TEXT})을 업로드해 주세요."
            )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            fmt = (img.format or "").upper()
            if fmt not in SUPPORTED_PIL_FORMATS:
                raise InvalidImageFormatError(
                    f"지원하지 않는 이미지 포맷({fmt or 'UNKNOWN'})입니다. "
                    f"지원 이미지 형식({SUPPORTED_FORMATS_TEXT})을 업로드해 주세요."
                )
            img.verify()

        # Re-open from RAM after verify() because Pillow closes/invalidates the stream pointer.
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb_img = img.convert("RGB")
            return np.array(rgb_img, dtype=np.uint8)
    except InvalidImageFormatError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageFormatError(
            f"이미지 파일을 읽을 수 없습니다. 손상되지 않은 지원 이미지 형식({SUPPORTED_FORMATS_TEXT})을 사용해 주세요."
        ) from exc


def load_image_from_path(path: str | Path) -> np.ndarray:
    """Read an image file into RAM and validate without retaining extra copies."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    raw_bytes = file_path.read_bytes()
    return validate_and_decode_bytes(raw_bytes, filename=file_path.name)


def load_rgb_fast(path: str | Path) -> np.ndarray:
    """Decode an already-verified dataset image directly into an RGB uint8 array."""
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return load_image_from_path(path)


def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Albumentations training pipeline resilient to lighting, rotation, and angle changes."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.88, 1.12),
                translate_percent=(-0.06, 0.06),
                rotate=(-30, 30),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.18, contrast_limit=0.18, p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=6, sat_shift_limit=15, val_shift_limit=15, p=0.4
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Albumentations deterministic preprocessing pipeline for validation/inference."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def numpy_to_tensor(normalized_img: np.ndarray) -> torch.Tensor:
    """Convert HWC float32 normalized image to CHW torch.Tensor."""
    chw = np.transpose(normalized_img, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(chw)


class PillImageDataset(Dataset):
    """Dataset reading `split/{High_Risk,Caution,Safe}` pill images."""

    def __init__(
        self,
        root_dir: Path | str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir) / split
        self.transform = transform or get_eval_transforms()
        self.samples: List[Tuple[Path, int]] = []

        for label_idx, class_name in enumerate(CLASS_NAMES):
            dir_name = CLASS_DIR_NAMES[class_name]
            class_dir = self.root_dir / dir_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label_idx = self.samples[index]
        rgb = load_rgb_fast(img_path)
        augmented = self.transform(image=rgb)["image"]
        return numpy_to_tensor(augmented), label_idx


# ---------------------------------------------------------------------------
# Synthetic MFDS-style pill dataset generator for MVP verification & samples
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _create_background(size: int, rng: random.Random) -> np.ndarray:
    """Create a neutral studio/table background with subtle gradient and grain."""
    palettes = [
        (232, 235, 240),
        (220, 224, 228),
        (240, 236, 228),
        (210, 218, 225),
        (196, 202, 210),
    ]
    base = np.array(rng.choice(palettes), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    grad = ((xx - 0.5) * rng.uniform(-18, 18) + (yy - 0.5) * rng.uniform(-18, 18))
    bg = np.clip(base[None, None, :] + grad[:, :, None], 160, 250)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 2.5, bg.shape)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def _draw_pill_image(
    class_name: str,
    rng: random.Random,
    size: int = 256,
    deterministic_sample: bool = False,
) -> Image.Image:
    """Render a realistic pill with shape, color, score line, and embossed imprint."""
    bg = Image.fromarray(_create_background(size, rng)).convert("RGBA")
    cx = size // 2 + (0 if deterministic_sample else rng.randint(-12, 12))
    cy = size // 2 + (0 if deterministic_sample else rng.randint(-12, 12))
    angle = 0.0 if deterministic_sample else rng.uniform(-28, 28)

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    sdraw = ImageDraw.Draw(shadow)

    if class_name == "High Risk":
        # Hexagonal / faceted tablet or crimson-amber steroid tablet with sharp imprint
        colors = [
            ((214, 152, 42), (140, 88, 14)),
            ((196, 48, 56), (118, 22, 28)),
            ((52, 82, 148), (28, 46, 92)),
        ]
        fill_rgb, edge_rgb = colors[0] if deterministic_sample else rng.choice(colors)
        imprint = "ST10" if deterministic_sample else rng.choice(["ST10", "OX50", "S1", "TRN", "S4"])
        radius = 70 if deterministic_sample else rng.randint(64, 76)
        pts = [
            (
                cx + radius * math.cos(math.radians(60 * k + 30)),
                cy + radius * math.sin(math.radians(60 * k + 30)),
            )
            for k in range(6)
        ]
        s_pts = [(x + 6, y + 8) for x, y in pts]
        sdraw.polygon(s_pts, fill=(30, 35, 45, 75))
        draw.polygon(pts, fill=(*fill_rgb, 255), outline=(*edge_rgb, 255), width=4)
        inner_pts = [
            (
                cx + (radius - 10) * math.cos(math.radians(60 * k + 30)),
                cy + (radius - 10) * math.sin(math.radians(60 * k + 30)),
            )
            for k in range(6)
        ]
        bevel = tuple(min(255, c + 28) for c in fill_rgb)
        draw.polygon(inner_pts, outline=(*bevel, 200), width=2)
        text_color = tuple(max(20, c - 75) for c in fill_rgb) + (235,)
        font = _load_font(28)
        draw.text((cx, cy), imprint, fill=text_color, font=font, anchor="mm")

    elif class_name == "Caution":
        # Elongated oval / oblong tablet with prominent center score line + S6/PSE imprint
        colors = [
            ((78, 162, 214), (42, 108, 156)),
            ((64, 178, 154), (34, 118, 102)),
            ((232, 128, 62), (168, 82, 28)),
        ]
        fill_rgb, edge_rgb = colors[0] if deterministic_sample else rng.choice(colors)
        imprint = "PSE" if deterministic_sample else rng.choice(["PSE", "S6", "CF", "S3", "120"])
        rx = 88 if deterministic_sample else rng.randint(82, 94)
        ry = 44 if deterministic_sample else rng.randint(40, 48)
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        sbbox = (cx - rx + 6, cy - ry + 8, cx + rx + 6, cy + ry + 8)
        sdraw.rounded_rectangle(sbbox, radius=ry, fill=(30, 35, 45, 70))
        draw.rounded_rectangle(bbox, radius=ry, fill=(*fill_rgb, 255), outline=(*edge_rgb, 255), width=4)
        # Prominent vertical score line dividing the oblong tablet
        draw.line([(cx, cy - ry + 6), (cx, cy + ry - 6)], fill=(*edge_rgb, 220), width=4)
        text_color = (245, 250, 255, 235)
        font = _load_font(26)
        draw.text((cx - rx * 0.45, cy), imprint, fill=text_color, font=font, anchor="mm")
        draw.text((cx + rx * 0.45, cy), "60", fill=text_color, font=font, anchor="mm")

    else:  # Safe
        # Classic round white/ivory pharmaceutical tablet with score line + AAP/500 imprint
        colors = [
            ((246, 247, 249), (190, 196, 204)),
            ((248, 245, 232), (195, 188, 170)),
            ((244, 248, 236), (186, 196, 176)),
        ]
        fill_rgb, edge_rgb = colors[0] if deterministic_sample else rng.choice(colors)
        imprint = "AAP" if deterministic_sample else rng.choice(["AAP", "500", "TY", "VIT", "OK"])
        r = 68 if deterministic_sample else rng.randint(62, 72)
        bbox = (cx - r, cy - r, cx + r, cy + r)
        sbbox = (cx - r + 6, cy - r + 8, cx + r + 6, cy + r + 8)
        sdraw.ellipse(sbbox, fill=(35, 40, 50, 65))
        draw.ellipse(bbox, fill=(*fill_rgb, 255), outline=(*edge_rgb, 255), width=4)
        draw.ellipse(
            (cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8),
            outline=(255, 255, 255, 200),
            width=2,
        )
        # Horizontal score line + imprint above/below
        draw.line([(cx - r + 10, cy), (cx + r - 10, cy)], fill=(*edge_rgb, 210), width=3)
        text_color = (125, 132, 144, 235)
        font = _load_font(25)
        draw.text((cx, cy - r * 0.36), imprint, fill=text_color, font=font, anchor="mm")
        draw.text((cx, cy + r * 0.36), "500", fill=text_color, font=font, anchor="mm")

    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
    if abs(angle) > 0.5:
        layer = layer.rotate(angle, center=(cx, cy), resample=Image.Resampling.BICUBIC)
        shadow = shadow.rotate(angle, center=(cx, cy), resample=Image.Resampling.BICUBIC)

    composed = Image.alpha_composite(bg, shadow)
    composed = Image.alpha_composite(composed, layer).convert("RGB")
    return composed


def generate_mvp_pill_dataset(
    data_dir: Path | str = DATA_DIR,
    samples_dir: Path | str = SAMPLES_DIR,
    train_per_class: int = 60,
    val_per_class: int = 15,
    test_per_class: int = 20,
    seed: int = 42,
) -> None:
    """Create train/val/test datasets and 3 UI sample chips if not already populated."""
    data_root = Path(data_dir)
    samples_root = Path(samples_dir)
    samples_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    splits = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }

    for split_name, count in splits.items():
        for class_name in CLASS_NAMES:
            cls_dir = data_root / split_name / CLASS_DIR_NAMES[class_name]
            cls_dir.mkdir(parents=True, exist_ok=True)
            existing = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
            if len(existing) >= count:
                continue
            for idx in range(count):
                out_path = cls_dir / f"{CLASS_DIR_NAMES[class_name].lower()}_{idx:03d}.png"
                if not out_path.exists():
                    img = _draw_pill_image(class_name, rng=rng, size=256)
                    img.save(out_path, format="PNG")

    sample_files = {
        "High Risk": samples_root / "sample_high_risk.png",
        "Caution": samples_root / "sample_caution.png",
        "Safe": samples_root / "sample_safe.png",
    }
    for class_name, sample_path in sample_files.items():
        if not sample_path.exists():
            img = _draw_pill_image(
                class_name, rng=random.Random(seed), size=320, deterministic_sample=True
            )
            img.save(sample_path, format="PNG")
