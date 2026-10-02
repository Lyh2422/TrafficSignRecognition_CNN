"""Dataset utilities for the flat-file TSRD train and test directories."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from app.labels import LABEL_BY_ID


class TSRDDataset(Dataset):
    """Load TSRD images whose first three filename characters encode the class."""

    extensions = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, directory: Path | str, transform=None) -> None:
        self.directory = Path(directory)
        self.transform = transform
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"TSRD directory not found: {self.directory}. "
                "Run `python -m scripts.prepare_tsrd` first."
            )

        self.samples: list[tuple[Path, int]] = []
        for path in sorted(self.directory.iterdir()):
            if path.suffix.lower() not in self.extensions:
                continue
            try:
                class_id = int(path.name[:3])
            except ValueError as exc:
                raise ValueError(f"Invalid TSRD filename: {path.name}") from exc
            if class_id not in LABEL_BY_ID:
                raise ValueError(f"Unknown TSRD class {class_id} in {path.name}")
            self.samples.append((path, class_id))

        if not self.samples:
            raise RuntimeError(f"No TSRD images found in {self.directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]

    @property
    def class_counts(self) -> Counter:
        return Counter(self.targets)
