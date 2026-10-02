"""Export one correctly recognized, tightly cropped Chinese sign per TSRD class."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

from app.labels import LABEL_BY_ID
from app.model import build_model
from scripts.train import build_transforms, select_device
from scripts.tsrd_dataset import TSRDDataset


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def find_best_samples(model, dataset, device, best, source_split: str) -> None:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            for local_index, label in enumerate(labels.tolist()):
                confidence = probabilities[local_index, label].item()
                if predictions[local_index].item() != label or confidence <= best[label]["confidence"]:
                    continue
                best[label] = {
                    "confidence": confidence,
                    "index": offset + local_index,
                    "dataset": dataset,
                    "source_split": source_split,
                }
            offset += labels.size(0)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "sample_images"
    output_dir.mkdir(exist_ok=True)
    device = select_device("auto")
    _, transform = build_transforms()
    test_dataset = TSRDDataset(root / "data/tsrd/full_classes/test", transform=transform)
    train_dataset = TSRDDataset(root / "data/tsrd/full_classes/train", transform=transform)

    checkpoint = torch.load(root / "models/tsrd_cnn.pt", map_location=device, weights_only=True)
    model = build_model(
        len(LABEL_BY_ID), checkpoint["model_name"], pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best = {
        class_id: {"confidence": -1.0, "index": None}
        for class_id in LABEL_BY_ID
    }
    find_best_samples(model, test_dataset, device, best, "test")
    missing = {class_id for class_id, item in best.items() if item["index"] is None}
    if missing:
        train_best = {
            class_id: {"confidence": -1.0, "index": None}
            for class_id in LABEL_BY_ID
        }
        find_best_samples(model, train_dataset, device, train_best, "train")
        for class_id in missing:
            best[class_id] = train_best[class_id]

    manifest = []
    exported_paths = []
    for class_id, item in best.items():
        if item["index"] is None:
            continue
        label = LABEL_BY_ID[class_id]
        dataset = item["dataset"]
        source_path, _, bbox = dataset.samples[item["index"]]
        filename = f"{class_id:02d}-{slugify(label['name_en'])}.png"
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        if bbox is not None:
            image = dataset._crop(image, bbox)
        image.save(output_dir / filename, format="PNG", optimize=True)
        exported_paths.append((class_id, output_dir / filename))
        manifest.append(
            {
                **label,
                "filename": filename,
                "confidence": round(item["confidence"], 6),
                "source_split": item["source_split"],
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cell_width, cell_height, columns = 142, 132, 7
    rows = (len(exported_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "#f2f0e7")
    draw = ImageDraw.Draw(sheet)
    for position, (class_id, path) in enumerate(exported_paths):
        x = (position % columns) * cell_width
        y = (position // columns) * cell_height
        with Image.open(path) as source:
            thumbnail = ImageOps.contain(source.convert("RGB"), (96, 96))
        image_x = x + (cell_width - thumbnail.width) // 2
        image_y = y + 5 + (96 - thumbnail.height) // 2
        sheet.paste(thumbnail, (image_x, image_y))
        draw.text((x + 8, y + 106), f"{class_id:02d} {LABEL_BY_ID[class_id]['name_zh']}", fill="#151712")
    sheet.save(output_dir / "contact-sheet.png", format="PNG", optimize=True)

    print(f"Exported {len(manifest)} verified Chinese samples to {output_dir}")
    print(f"Official test samples: {sum(item['source_split'] == 'test' for item in manifest)}")
    print(f"Training fallback samples: {sum(item['source_split'] == 'train' for item in manifest)}")


if __name__ == "__main__":
    main()
