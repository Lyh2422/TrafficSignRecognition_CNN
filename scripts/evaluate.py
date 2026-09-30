"""Evaluate the saved CNN checkpoint on the official GTSRB test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from app.model import TrafficSignCNN
from scripts.train import NUM_CLASSES, build_transforms, evaluate, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/gtsrb"))
    parser.add_argument("--weights", type=Path, default=Path("models/gtsrb_cnn.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    _, eval_transform = build_transforms()
    dataset = datasets.GTSRB(args.data_dir, split="test", transform=eval_transform, download=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    model = TrafficSignCNN(NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(label_smoothing=0.05), device)
    print(f"device={device}")
    print(f"samples={len(dataset)}")
    print(f"accuracy={metrics['accuracy']:.4%}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()

