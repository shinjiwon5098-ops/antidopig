"""Build a lightweight (<10 MB) L2-normalized embedding index for MFDS pill matching."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (  # noqa: E402
    CLASS_DIR_NAMES,
    DATA_DIR,
    DEFAULT_WEIGHTS_PATH,
    MAPPED_PILL_CSV_PATH,
    MODELS_DIR,
    SUPPORTED_EXTENSIONS,
)
from src.dataset import get_eval_transforms, load_rgb_fast, numpy_to_tensor  # noqa: E402
from src.model import load_trained_model  # noqa: E402

PRIORITY_SAFE_INGREDIENTS = (
    "아세트아미노펜",
    "이부프로펜",
    "덱시부프로펜",
    "나프록센",
    "세티리진",
    "로라타딘",
    "펙소페나딘",
)


def _index_split_images(data_dir: Path) -> Dict[str, Path]:
    """Map ITEM_SEQ -> image path across data/{train,val,test}/{High_Risk,Caution,Safe}."""
    index: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for dir_name in CLASS_DIR_NAMES.values():
            folder = data_dir / split / dir_name
            if not folder.exists():
                continue
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and p.stem not in index:
                    index[p.stem] = p
    return index


def _index_raw_images(raw_dir: Path) -> Dict[str, Path]:
    """Fallback map ITEM_SEQ -> image path inside raw pill_images directory."""
    index: Dict[str, Path] = {}
    if not raw_dir.exists():
        return index
    for p in raw_dir.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            seq = p.name.split("_", 1)[0]
            if seq and seq not in index:
                index[seq] = p
    return index


def _clean_ingredient(row: Dict[str, str]) -> str:
    extracted = (row.get("EXTRACTED_INGREDIENTS") or "").strip()
    if extracted:
        return extracted
    matched = (row.get("MATCHED_SUBSTANCES") or "").strip()
    if matched:
        return matched
    return (row.get("CLASS_NAME") or "성분 정보 미상").strip()


def _select_representative_entries(
    csv_path: Path,
    split_index: Dict[str, Path],
    raw_index: Dict[str, Path],
    max_total: int = 2000,
    seed: int = 42,
) -> List[Tuple[Dict[str, str], Path]]:
    """Keep all High Risk & Caution pills and stratified representative Safe pills."""
    high_caution: List[Tuple[Dict[str, str], Path]] = []
    safe_priority: List[Tuple[Dict[str, str], Path]] = []
    safe_buckets: Dict[Tuple[str, str], List[Tuple[Dict[str, str], Path]]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item_seq = (row.get("ITEM_SEQ") or "").strip()
            if not item_seq:
                continue

            img_path: Optional[Path] = split_index.get(item_seq)
            if img_path is None:
                local_raw = (row.get("LOCAL_IMAGE_PATH") or "").strip()
                if local_raw:
                    candidate = Path(local_raw)
                    if candidate.exists() and candidate.is_file():
                        img_path = candidate
            if img_path is None:
                img_path = raw_index.get(item_seq)
            if img_path is None or not img_path.exists():
                continue

            risk = (row.get("RISK_LEVEL") or "Safe").strip()
            if risk in ("High Risk", "Caution"):
                high_caution.append((row, img_path))
                continue

            ing_text = _clean_ingredient(row)
            if any(k in ing_text for k in PRIORITY_SAFE_INGREDIENTS):
                safe_priority.append((row, img_path))
            else:
                shape = (row.get("DRUG_SHAPE") or "기타").strip()
                color = (row.get("COLOR_CLASS1") or "기타").strip().split(",")[0].strip()
                safe_buckets[(shape, color)].append((row, img_path))

    selected: List[Tuple[Dict[str, str], Path]] = list(high_caution)
    seen_seqs = {(r.get("ITEM_SEQ") or "").strip() for r, _ in selected}

    rng = random.Random(seed)
    rng.shuffle(safe_priority)
    safe_quota = max(0, max_total - len(selected))

    # Keep up to ~25% of Safe quota for common OTC single/combo products.
    priority_cap = min(len(safe_priority), max(120, safe_quota // 4))
    for row, img_path in safe_priority[:priority_cap]:
        seq = (row.get("ITEM_SEQ") or "").strip()
        if seq not in seen_seqs:
            selected.append((row, img_path))
            seen_seqs.add(seq)

    # Round-robin stratified sampling across (shape, color) buckets for visual diversity.
    bucket_keys = sorted(safe_buckets.keys())
    for key in bucket_keys:
        rng.shuffle(safe_buckets[key])

    cursor = 0
    while len(selected) < max_total:
        added_this_round = False
        for key in bucket_keys:
            bucket = safe_buckets[key]
            while bucket:
                row, img_path = bucket.pop()
                seq = (row.get("ITEM_SEQ") or "").strip()
                if seq not in seen_seqs:
                    selected.append((row, img_path))
                    seen_seqs.add(seq)
                    added_this_round = True
                    break
            if len(selected) >= max_total:
                break
        cursor += 1
        if not added_this_round:
            break

    # Fill any remaining quota from leftover priority items if needed.
    if len(selected) < max_total:
        for row, img_path in safe_priority[priority_cap:]:
            seq = (row.get("ITEM_SEQ") or "").strip()
            if seq not in seen_seqs:
                selected.append((row, img_path))
                seen_seqs.add(seq)
                if len(selected) >= max_total:
                    break

    return selected


def build_pill_index(
    csv_path: Path = MAPPED_PILL_CSV_PATH,
    data_dir: Path = DATA_DIR,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    output_dir: Path = MODELS_DIR,
    max_total: int = 2000,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Extract L2-normalized embeddings and save compact NPZ + JSON index under 10 MB."""
    if not csv_path.exists():
        raise FileNotFoundError(f"MFDS mapping CSV not found: {csv_path}")

    split_index = _index_split_images(data_dir)
    raw_index = _index_raw_images(BASE_DIR.parent / "pill_images")
    entries = _select_representative_entries(
        csv_path=csv_path,
        split_index=split_index,
        raw_index=raw_index,
        max_total=max_total,
        seed=seed,
    )
    print(f"Selected {len(entries)} representative pills for indexing.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_trained_model(weights_path=weights_path, device=device)
    model.eval()
    eval_tf = get_eval_transforms()

    all_embeddings: List[np.ndarray] = []
    metadata_records: List[Dict[str, str]] = []
    item_seqs: List[str] = []

    batch_tensors: List[torch.Tensor] = []
    batch_meta: List[Dict[str, str]] = []

    def _flush_batch() -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            feats = model.forward_features(x)
            emb = model.forward_head(feats, pre_logits=True)
            emb_norm = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb_norm.cpu().numpy().astype(np.float16))
        for m in batch_meta:
            metadata_records.append(m)
            item_seqs.append(m["item_seq"])
        batch_tensors.clear()
        batch_meta.clear()

    for idx, (row, img_path) in enumerate(entries, start=1):
        try:
            rgb = load_rgb_fast(img_path)
            tensor = numpy_to_tensor(eval_tf(image=rgb)["image"])
        except Exception:
            continue

        batch_tensors.append(tensor)
        batch_meta.append(
            {
                "item_seq": (row.get("ITEM_SEQ") or "").strip(),
                "item_name": (row.get("ITEM_NAME") or "").strip(),
                "ingredient": _clean_ingredient(row),
                "print_front": (row.get("PRINT_FRONT") or "").strip(),
                "print_back": (row.get("PRINT_BACK") or "").strip(),
                "color": (row.get("COLOR_CLASS1") or "").strip(),
                "shape": (row.get("DRUG_SHAPE") or "").strip(),
                "risk_level": (row.get("RISK_LEVEL") or "Safe").strip(),
                "wada_code": (row.get("WADA_CODES") or "").strip(),
            }
        )
        if len(batch_tensors) >= batch_size:
            _flush_batch()
        if idx % 400 == 0:
            print(f"Processed {idx}/{len(entries)} images...", flush=True)

    _flush_batch()

    if not all_embeddings:
        raise RuntimeError("No valid pill embeddings were extracted.")

    emb_matrix = np.concatenate(all_embeddings, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "pill_reference_index.npz"
    meta_path = output_dir / "pill_metadata.json"

    np.savez_compressed(
        npz_path,
        embeddings=emb_matrix,
        item_seqs=np.array(item_seqs, dtype=np.str_),
    )
    meta_path.write_text(
        json.dumps(metadata_records, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    total_bytes = npz_path.stat().st_size + meta_path.stat().st_size
    print(
        f"Indexed {len(metadata_records)} pills -> "
        f"{npz_path.name} ({npz_path.stat().st_size / (1024*1024):.2f} MB) + "
        f"{meta_path.name} ({meta_path.stat().st_size / (1024*1024):.2f} MB) = "
        f"{total_bytes / (1024*1024):.2f} MB total",
        flush=True,
    )
    return npz_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight MFDS pill embedding index.")
    parser.add_argument("--csv", type=Path, default=MAPPED_PILL_CSV_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--max-total", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_pill_index(
        csv_path=args.csv,
        data_dir=args.data_dir,
        weights_path=args.weights,
        output_dir=args.output_dir,
        max_total=args.max_total,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
