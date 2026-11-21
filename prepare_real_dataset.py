"""Replace synthetic MVP pill images with real MFDS pill images mapped by WADA risk."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError

from src.config import (
    CLASS_DIR_NAMES,
    CLASS_NAMES,
    DATA_DIR,
    SAMPLES_DIR,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_PIL_FORMATS,
)

ROOT_PROJECT_DIR = DATA_DIR.parent.parent
DEFAULT_MAPPED_CSV = ROOT_PROJECT_DIR / "pill_doping_mapped.csv"
DEFAULT_IMAGE_DIR = ROOT_PROJECT_DIR / "pill_images"


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    """Map ITEM_SEQ prefix to downloaded image path inside pill_images."""
    index: Dict[str, Path] = {}
    if not image_dir.exists():
        return index

    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        item_seq = path.name.split("_", 1)[0]
        if item_seq and item_seq not in index:
            index[item_seq] = path
    return index


def validate_or_convert_image(src_path: Path, dst_dir: Path, stem: str) -> Path | None:
    """Verify image integrity and write/copy it in a supported format (JPG/PNG/WEBP)."""
    try:
        with Image.open(src_path) as img:
            fmt = (img.format or "").upper()
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    ext = src_path.suffix.lower()
    if fmt in SUPPORTED_PIL_FORMATS and ext in SUPPORTED_EXTENSIONS:
        dst_path = dst_dir / f"{stem}{ext}"
        shutil.copy2(src_path, dst_path)
        return dst_path

    # Convert unsupported formats (e.g., GIF/BMP) into RGB JPEG for dataset compatibility.
    try:
        with Image.open(src_path) as img:
            rgb = img.convert("RGB")
            dst_path = dst_dir / f"{stem}.jpg"
            rgb.save(dst_path, format="JPEG", quality=95)
            return dst_path
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def clear_split_directories(data_dir: Path) -> None:
    """Remove existing images inside train/val/test class folders."""
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            cls_dir = data_dir / split / CLASS_DIR_NAMES[class_name]
            cls_dir.mkdir(parents=True, exist_ok=True)
            for item in cls_dir.iterdir():
                if item.is_file():
                    item.unlink()


def stratified_split(
    items: List[Tuple[str, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, Path]]]:
    """Deterministically split class items into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    if val_end >= total and total >= 3:
        train_end = total - 2
        val_end = total - 1

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def update_ui_samples(data_dir: Path, samples_dir: Path) -> None:
    """Populate the 3 UI sample chip PNGs using real images from test/val splits."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_targets = {
        "High Risk": samples_dir / "sample_high_risk.png",
        "Caution": samples_dir / "sample_caution.png",
        "Safe": samples_dir / "sample_safe.png",
    }

    for class_name, out_path in sample_targets.items():
        dir_name = CLASS_DIR_NAMES[class_name]
        chosen: Path | None = None
        for split in ("test", "val", "train"):
            candidates = sorted((data_dir / split / dir_name).glob("*"))
            if candidates:
                chosen = candidates[0]
                break
        if chosen is not None:
            with Image.open(chosen) as img:
                img.convert("RGB").save(out_path, format="PNG")


def prepare_dataset(
    mapped_csv: Path = DEFAULT_MAPPED_CSV,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    data_dir: Path = DATA_DIR,
    samples_dir: Path = SAMPLES_DIR,
    seed: int = 42,
) -> Dict:
    """Build stratified real pill dataset inside doping_classifier/data."""
    image_index = build_image_index(image_dir)
    by_class: Dict[str, List[Tuple[str, Path]]] = {c: [] for c in CLASS_NAMES}
    missing_image = 0

    with mapped_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            risk = (row.get("RISK_LEVEL") or "").strip()
            if risk not in by_class:
                continue

            item_seq = str(row.get("ITEM_SEQ") or "").strip()
            local_raw = (row.get("LOCAL_IMAGE_PATH") or "").strip()
            src_path = Path(local_raw) if local_raw else None
            if src_path is None or not src_path.exists() or src_path.stat().st_size == 0:
                src_path = image_index.get(item_seq)

            if src_path is None or not src_path.exists():
                missing_image += 1
                continue
            by_class[risk].append((item_seq, src_path))

    clear_split_directories(data_dir)

    counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    corrupt_count = 0

    for class_name in CLASS_NAMES:
        splits = stratified_split(by_class[class_name], seed=seed)
        dir_name = CLASS_DIR_NAMES[class_name]
        for split_name, entries in splits.items():
            dst_dir = data_dir / split_name / dir_name
            saved = 0
            for item_seq, src_path in entries:
                out = validate_or_convert_image(src_path, dst_dir, stem=item_seq)
                if out is None:
                    corrupt_count += 1
                else:
                    saved += 1
            counts[split_name][class_name] = saved

    update_ui_samples(data_dir, samples_dir)

    manifest = {
        "source_csv": str(mapped_csv),
        "source_image_dir": str(image_dir),
        "missing_images": missing_image,
        "corrupt_images": corrupt_count,
        "counts": counts,
    }
    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate doping_classifier/data with real MFDS pill images."
    )
    parser.add_argument("--mapped-csv", type=Path, default=DEFAULT_MAPPED_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = prepare_dataset(
        mapped_csv=args.mapped_csv,
        image_dir=args.image_dir,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
