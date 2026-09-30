"""Export high-confidence, correctly classified GTSRB test images as PNG samples."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets

from app.labels import LABEL_BY_ID
from app.model import TrafficSignCNN
from scripts.train import build_transforms, select_device


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


TARGETS = {
    class_id: slugify(LABEL_BY_ID[class_id]["name_en"])
    for class_id in sorted(LABEL_BY_ID)
}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "sample_images"
    output_dir.mkdir(exist_ok=True)
    device = select_device("auto")
    _, transform = build_transforms()
    dataset = datasets.GTSRB(root / "data/gtsrb", split="test", transform=transform, download=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    checkpoint = torch.load(root / "models/gtsrb_cnn.pt", map_location=device, weights_only=True)
    model = TrafficSignCNN(43).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best = {class_id: {"confidence": -1.0, "index": None} for class_id in TARGETS}
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            for local_index, label in enumerate(labels.tolist()):
                if label not in TARGETS or predictions[local_index].item() != label:
                    continue
                confidence = probabilities[local_index, label].item()
                if confidence > best[label]["confidence"]:
                    best[label] = {"confidence": confidence, "index": offset + local_index}
            offset += labels.size(0)

    manifest = []
    exported_paths = []
    for class_id, slug in TARGETS.items():
        selected = best[class_id]
        if selected["index"] is None:
            continue
        source_path, _ = dataset._samples[selected["index"]]
        filename = f"{class_id:02d}-{slug}.png"
        with Image.open(source_path) as image:
            image.convert("RGB").save(output_dir / filename, format="PNG", optimize=True)
        exported_paths.append((class_id, output_dir / filename))
        manifest.append(
            {
                **LABEL_BY_ID[class_id],
                "filename": filename,
                "confidence": round(selected["confidence"], 6),
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cell_width, cell_height, columns = 132, 126, 7
    rows = (len(exported_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "#f2f0e7")
    draw = ImageDraw.Draw(sheet)
    for position, (class_id, path) in enumerate(exported_paths):
        x = (position % columns) * cell_width
        y = (position // columns) * cell_height
        with Image.open(path) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), (92, 92))
        image_x = x + (cell_width - thumbnail.width) // 2
        sheet.paste(thumbnail, (image_x, y + 6))
        draw.text((x + 10, y + 103), f"Class {class_id:02d}", fill="#151712")
    sheet.save(output_dir / "contact-sheet.png", format="PNG", optimize=True)

    print(f"Exported {len(manifest)} verified samples to {output_dir}")
    for item in manifest:
        print(f"{item['filename']}: {item['name_zh']} ({item['confidence']:.2%})")


if __name__ == "__main__":
    main()
