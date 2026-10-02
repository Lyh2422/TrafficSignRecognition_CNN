"""Evaluate a trained Chinese TSRD checkpoint on the official test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.labels import LABEL_BY_ID
from app.model import build_model
from scripts.train import build_transforms, evaluate, select_device
from scripts.tsrd_dataset import TSRDDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Chinese TSRD CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tsrd/full_classes"))
    parser.add_argument("--model", type=Path, default=Path("models/tsrd_cnn.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    _, transform = build_transforms()
    dataset = TSRDDataset(args.data_dir / "test", transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = build_model(
        num_classes=len(LABEL_BY_ID),
        architecture=checkpoint.get("model_name", "mobilenet_v3_small"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(label_smoothing=0.05), device)
    print(f"device={device}")
    print(f"samples={len(dataset)}")
    print(f"supported_classes={metrics['supported_class_count']}/{len(LABEL_BY_ID)}")
    print(f"accuracy={metrics['accuracy']:.4%}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
