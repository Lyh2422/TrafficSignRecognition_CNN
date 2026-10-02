"""Lightweight CNN architecture for Chinese traffic-sign classification."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except ImportError:  # The MVP can run in explicitly labelled demo mode without PyTorch.
    torch = None
    nn = None
    MobileNet_V3_Small_Weights = None
    mobilenet_v3_small = None


if nn is not None:

    class TrafficSignCNN(nn.Module):
        """CNN that preserves spatial detail for fine-grained TSRD classes."""

        def __init__(self, num_classes: int = 43) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.08),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.12),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.16),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.38),
                nn.Linear(256, num_classes),
            )

        def forward(self, inputs):
            return self.classifier(self.features(inputs))

else:
    TrafficSignCNN = None


def build_model(
    num_classes: int,
    architecture: str = "mobilenet_v3_small",
    pretrained: bool = False,
):
    """Build the selected CNN without downloading weights during inference."""
    if nn is None:
        return None
    if architecture == "custom_cnn":
        return TrafficSignCNN(num_classes=num_classes)
    if architecture == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model architecture: {architecture}")
