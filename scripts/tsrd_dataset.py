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

    def __init__(
        self,
        directory: Path | str,
        transform=None,
        crop_to_bbox: bool = True,
        crop_padding: float = 0.04,
    ) -> None:
        self.directory = Path(directory)
        self.transform = transform
        self.crop_to_bbox = crop_to_bbox
        self.crop_padding = crop_padding
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"TSRD directory not found: {self.directory}. "
                "Run `python -m scripts.prepare_tsrd` first."
            )

        annotations = self._load_annotations()
        self.samples: list[tuple[Path, int, tuple[int, int, int, int] | None]] = []
        for path in sorted(self.directory.iterdir()):
            if path.suffix.lower() not in self.extensions:
                continue
            try:
                class_id = int(path.name[:3])
            except ValueError as exc:
                raise ValueError(f"Invalid TSRD filename: {path.name}") from exc
            if class_id not in LABEL_BY_ID:
                raise ValueError(f"Unknown TSRD class {class_id} in {path.name}")
            annotation = annotations.get(path.name)
            if annotation is not None and annotation[1] != class_id:
                raise ValueError(f"Label mismatch for {path.name}")
            bbox = annotation[0] if annotation is not None else None
            self.samples.append((path, class_id, bbox))

        if not self.samples:
            raise RuntimeError(f"No TSRD images found in {self.directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, bbox = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.crop_to_bbox and bbox is not None:
            image = self._crop(image, bbox)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> list[int]:
        return [label for _, label, _ in self.samples]

    @property
    def class_counts(self) -> Counter:
        return Counter(self.targets)

    def _load_annotations(self) -> dict[str, tuple[tuple[int, int, int, int], int]]:
        split = self.directory.name.lower()
        annotation_name = {
            "train": "TsignRecgTrainAnnotation.txt",
            "test": "TsignRecgTestAnnotation.txt",
        }.get(split)
        if annotation_name is None:
            return {}
        annotation_path = self.directory.parent / annotation_name
        if not annotation_path.exists():
            return {}

        annotations = {}
        with annotation_path.open(encoding="utf-8-sig") as file:
            for line in file:
                fields = line.strip().split(";")
                if len(fields) < 8:
                    continue
                filename = fields[0]
                bbox = tuple(int(value) for value in fields[3:7])
                annotations[filename] = (bbox, int(fields[7]))
        return annotations

    def _crop(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
        left, top, right, bottom = bbox
        padding = round(max(right - left, bottom - top) * self.crop_padding)
        return image.crop(
            (
                max(0, left - padding),
                max(0, top - padding),
                min(image.width, right + padding + 1),
                min(image.height, bottom + padding + 1),
            )
        )
