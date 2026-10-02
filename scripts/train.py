"""Train, validate, and test the 58-class Chinese TSRD CNN."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

from app.labels import DATASET_INFO, LABEL_BY_ID
from app.model import build_model
from scripts.tsrd_dataset import TSRDDataset


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 96
NUM_CLASSES = len(LABEL_BY_ID)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Chinese TSRD traffic-sign CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tsrd/full_classes"))
    parser.add_argument("--output", type=Path, default=Path("models/tsrd_cnn.pt"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--architecture",
        choices=("mobilenet_v3_small", "custom_cnn"),
        default="mobilenet_v3_small",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialize MobileNetV3 with ImageNet weights",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE + 12, IMAGE_SIZE + 12)),
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE), padding=5, padding_mode="edge"),
            transforms.RandomRotation(9, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.88, 1.12),
                shear=(-4, 4),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.12,
                p=0.25,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.16, hue=0.025),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.12),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, eval_transform


def stratified_split(targets: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        by_class[label].append(index)

    generator = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_class.values():
        generator.shuffle(indices)
        val_count = min(len(indices) - 1, max(1, round(len(indices) * val_ratio)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    return train_indices, val_indices


def build_loaders(args: argparse.Namespace, device: torch.device):
    train_transform, eval_transform = build_transforms()
    train_augmented = TSRDDataset(args.data_dir / "train", transform=train_transform)
    train_evaluation = TSRDDataset(args.data_dir / "train", transform=eval_transform)
    test_dataset = TSRDDataset(args.data_dir / "test", transform=eval_transform)
    train_indices, val_indices = stratified_split(
        train_augmented.targets, args.val_ratio, args.seed
    )

    train_targets = [train_augmented.targets[index] for index in train_indices]
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    sample_weights = [1.0 / max(1, class_counts[label]) for label in train_targets]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(Subset(train_augmented, train_indices), sampler=sampler, **loader_args)
    val_loader = DataLoader(Subset(train_evaluation, val_indices), shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    print(
        f"TSRD ready: train={len(train_indices)}, val={len(val_indices)}, "
        f"test={len(test_dataset)}, classes={NUM_CLASSES}",
        flush=True,
    )
    return train_loader, val_loader, test_loader, class_counts.tolist()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += batch_size
    return total_loss / samples, correct / samples


@torch.inference_mode()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total_loss = 0.0
    samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        samples += batch_size
        encoded = labels.cpu() * NUM_CLASSES + predictions.cpu()
        confusion += torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
            NUM_CLASSES, NUM_CLASSES
        )

    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1)
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    supported = support > 0
    return {
        "loss": total_loss / samples,
        "accuracy": true_positive.sum().item() / samples,
        "macro_f1": f1[supported].mean().item(),
        "supported_class_count": int(supported.sum().item()),
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_support": support.tolist(),
    }


def save_history_csv(path: Path, history: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training device: {device}", flush=True)

    train_loader, val_loader, test_loader, class_counts = build_loaders(args, device)
    model = build_model(
        num_classes=NUM_CLASSES,
        architecture=args.architecture,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
    )

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2%} "
            f"val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.architecture,
                    "model_version": "tsrd-v1",
                    "dataset": DATASET_INFO,
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "labels": LABEL_BY_ID,
                },
                args.output,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    checkpoint["test_accuracy"] = test_metrics["accuracy"]
    checkpoint["test_macro_f1"] = test_metrics["macro_f1"]
    torch.save(checkpoint, args.output)

    elapsed = time.perf_counter() - started
    report = {
        "dataset": DATASET_INFO,
        "model": {
            "name": args.architecture,
            "version": "tsrd-v1",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "image_size": IMAGE_SIZE,
            "mean": MEAN,
            "std": STD,
        },
        "training": {
            "seed": args.seed,
            "best_epoch": best_epoch,
            "completed_epochs": len(history),
            "elapsed_seconds": round(elapsed, 2),
            "class_counts_after_split": class_counts,
            "balanced_sampling": True,
        },
        "validation": {
            key: value
            for key, value in evaluate(model, val_loader, criterion, device).items()
            if key != "confusion_matrix"
        },
        "test": test_metrics,
    }
    save_history_csv(args.reports_dir / "tsrd_training_history.csv", history)
    (args.reports_dir / "tsrd_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Best epoch={best_epoch}; test accuracy={test_metrics['accuracy']:.2%}; "
        f"test macro-F1={test_metrics['macro_f1']:.4f}; saved to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
