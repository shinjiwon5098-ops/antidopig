"""Transfer learning training loop and Recall-centric evaluation for doping risk classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.config import CLASS_NAMES, DATA_DIR, DEFAULT_WEIGHTS_PATH, MODELS_DIR
from src.dataset import (
    PillImageDataset,
    generate_mvp_pill_dataset,
    get_eval_transforms,
    get_train_transforms,
)
from src.model import create_doping_classifier


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Compute accuracy, High Risk recall, per-class metrics, and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = float(accuracy_score(all_labels, all_preds))
    per_class_recall = recall_score(all_labels, all_preds, labels=[0, 1, 2], average=None, zero_division=0)
    high_risk_recall = float(per_class_recall[0])
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()
    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "high_risk_recall": high_risk_recall,
        "per_class_recall": {
            CLASS_NAMES[i]: float(per_class_recall[i]) for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm,
        "classification_report": report,
    }


def _has_populated_splits(data_dir: Path) -> bool:
    """Return True when train/val/test already contain images for every class."""
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            cls_dir = Path(data_dir) / split / class_name.replace(" ", "_")
            if not cls_dir.exists() or not any(cls_dir.iterdir()):
                return False
    return True


def _build_val_epoch_dataset(val_ds: PillImageDataset, safe_cap: int, seed: int) -> Subset | PillImageDataset:
    """Keep all minority val samples and cap Safe samples during fast epoch selection."""
    if safe_cap <= 0:
        return val_ds
    safe_idx = CLASS_NAMES.index("Safe")
    selected: list[int] = []
    safe_indices: list[int] = []
    for idx, (_, label) in enumerate(val_ds.samples):
        if label == safe_idx:
            safe_indices.append(idx)
        else:
            selected.append(idx)
    if len(safe_indices) <= safe_cap:
        return val_ds
    rng = np.random.default_rng(seed)
    chosen_safe = rng.choice(safe_indices, size=safe_cap, replace=False).tolist()
    selected.extend(chosen_safe)
    selected.sort()
    return Subset(val_ds, selected)


def train_model(
    data_dir: Path = DATA_DIR,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    backbone: str = "efficientnet_b0",
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 4e-4,
    seed: int = 42,
    samples_per_epoch: int = 2400,
    val_safe_cap: int = 300,
    num_workers: int = 4,
    skip_synthetic: bool = False,
) -> Dict:
    """Train transfer-learning model and save the best checkpoint by High Risk Recall & Accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not skip_synthetic and not _has_populated_splits(Path(data_dir)):
        generate_mvp_pill_dataset(data_dir=data_dir)

    train_ds = PillImageDataset(data_dir, "train", transform=get_train_transforms())
    val_ds = PillImageDataset(data_dir, "val", transform=get_eval_transforms())
    test_ds = PillImageDataset(data_dir, "test", transform=get_eval_transforms())
    val_epoch_ds = _build_val_epoch_dataset(val_ds, safe_cap=val_safe_cap, seed=seed)

    label_counts = Counter(label for _, label in train_ds.samples)
    min_count = min(label_counts.values()) if label_counts else 1
    max_count = max(label_counts.values()) if label_counts else 1
    use_balanced_sampler = max_count > 3 * max(1, min_count)

    loader_kwargs = (
        {"num_workers": num_workers, "persistent_workers": True}
        if num_workers > 0
        else {"num_workers": 0}
    )

    if use_balanced_sampler:
        # Balance classes across epochs under severe real-world imbalance while capping CPU epoch cost.
        sample_weights = [1.0 / label_counts[label] for _, label in train_ds.samples]
        epoch_samples = min(len(train_ds), max(samples_per_epoch, 3 * max(1, min_count)))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=epoch_samples,
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    else:
        epoch_samples = len(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)

    val_epoch_loader = DataLoader(val_epoch_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    val_full_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_doping_classifier(backbone=backbone, pretrained=True).to(device)

    # Slightly upweight High Risk class (index 0) to prioritize High Risk Recall >= 92%.
    weights_list = [1.3, 1.05, 1.0] if use_balanced_sampler else [1.4, 1.1, 1.0]
    class_weights = torch.tensor(weights_list, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_val_metrics: Dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)

        scheduler.step()
        avg_loss = running_loss / max(1, epoch_samples)
        val_metrics = evaluate_split(model, val_epoch_loader, device)

        # Composite selection score prioritizing High Risk Recall + overall Accuracy.
        score = 0.6 * val_metrics["high_risk_recall"] + 0.4 * val_metrics["accuracy"]
        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"loss={avg_loss:.4f} | val_acc={val_metrics['accuracy']*100:.2f}% | "
            f"val_high_risk_recall={val_metrics['high_risk_recall']*100:.2f}%",
            flush=True,
        )

        if score >= best_score:
            best_score = score
            best_val_metrics = val_metrics
            torch.save(
                {
                    "backbone": backbone,
                    "class_names": CLASS_NAMES,
                    "state_dict": model.state_dict(),
                    "metrics": {"val": val_metrics},
                },
                weights_path,
            )

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    full_val_metrics = evaluate_split(model, val_full_loader, device)
    test_metrics = evaluate_split(model, test_loader, device)

    checkpoint["metrics"] = {
        "val": full_val_metrics,
        "test": test_metrics,
    }
    torch.save(checkpoint, weights_path)

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "backbone": backbone,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "test_samples": len(test_ds),
                "val": full_val_metrics,
                "test": test_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Test Evaluation Summary ===")
    print(f"Accuracy         : {test_metrics['accuracy']*100:.2f}% (Target >= 85%)")
    print(f"High Risk Recall : {test_metrics['high_risk_recall']*100:.2f}% (Target >= 92%)")
    print(f"Confusion Matrix : {test_metrics['confusion_matrix']}")
    print(f"Saved weights to : {weights_path}")
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WADA Doping Risk Classifier")
    parser.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=2400)
    parser.add_argument("--val-safe-cap", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    train_model(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_epoch=args.samples_per_epoch,
        val_safe_cap=args.val_safe_cap,
        num_workers=args.num_workers,
        skip_synthetic=args.skip_synthetic,
    )


if __name__ == "__main__":
    main()
