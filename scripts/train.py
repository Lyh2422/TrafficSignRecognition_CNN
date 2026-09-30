"""Train, validate, and test the GTSRB CNN with reproducible settings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.labels import LABEL_BY_ID
from app.model import TrafficSignCNN


MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)
NUM_CLASSES = 43


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GTSRB traffic-sign CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data/gtsrb"))
    parser.add_argument("--output", type=Path, default=Path("models/gtsrb_cnn.pt"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
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
            transforms.Resize((36, 36)),
            transforms.RandomCrop((32, 32), padding=3, padding_mode="edge"),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, eval_transform


def build_loaders(args: argparse.Namespace, device: torch.device):
    train_transform, eval_transform = build_transforms()
    print("Preparing GTSRB dataset (download is skipped when files already exist)...", flush=True)
    train_augmented = datasets.GTSRB(args.data_dir, split="train", transform=train_transform, download=True)
    train_evaluation = datasets.GTSRB(args.data_dir, split="train", transform=eval_transform, download=False)
    test_dataset = datasets.GTSRB(args.data_dir, split="test", transform=eval_transform, download=True)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_augmented), generator=generator).tolist()
    val_size = max(1, int(len(indices) * args.val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(train_augmented, train_indices)
    val_dataset = Subset(train_evaluation, val_indices)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    print(
        f"Dataset ready: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}",
        flush=True,
    )
    return train_loader, val_loader, test_loader


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
        confusion += torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "loss": total_loss / samples,
        "accuracy": true_positive.sum().item() / samples,
        "macro_f1": f1.mean().item(),
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": recall.tolist(),
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

    train_loader, val_loader, test_loader = build_loaders(args, device)
    model = TrafficSignCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.18,
        anneal_strategy="cos",
    )

    best_accuracy = -1.0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        validation = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(validation["loss"], 6),
            "val_accuracy": round(validation["accuracy"], 6),
            "val_macro_f1": round(validation["macro_f1"], 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.perf_counter() - epoch_started, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train {train_accuracy:.2%} loss {train_loss:.4f} | "
            f"val {validation['accuracy']:.2%} F1 {validation['macro_f1']:.4f} | "
            f"{row['seconds']:.1f}s",
            flush=True,
        )

        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "TrafficSignCNN",
                    "model_version": 2,
                    "epoch": epoch,
                    "val_accuracy": validation["accuracy"],
                    "val_macro_f1": validation["macro_f1"],
                    "input_size": 32,
                    "mean": MEAN,
                    "std": STD,
                    "classes": NUM_CLASSES,
                },
                args.output,
            )
            print(f"  Saved new best checkpoint: {args.output}", flush=True)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} epochs without improvement.", flush=True)
                break

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    elapsed = time.perf_counter() - started
    report = {
        "model": "TrafficSignCNN",
        "model_version": 2,
        "device": str(device),
        "seed": args.seed,
        "training_seconds": round(elapsed, 2),
        "best_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["val_accuracy"],
        "validation_macro_f1": checkpoint["val_macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "history": history,
        "per_class": [
            {
                **LABEL_BY_ID[class_id],
                "recall": test_metrics["per_class_recall"][class_id],
            }
            for class_id in range(NUM_CLASSES)
        ],
        "confusion_matrix": test_metrics["confusion_matrix"],
    }
    report_path = args.reports_dir / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_history_csv(args.reports_dir / "training_history.csv", history)
    print(
        f"Training complete | best epoch {checkpoint['epoch']} | "
        f"test accuracy {test_metrics['accuracy']:.2%} | "
        f"macro F1 {test_metrics['macro_f1']:.4f} | total {elapsed / 60:.1f} min",
        flush=True,
    )
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()

