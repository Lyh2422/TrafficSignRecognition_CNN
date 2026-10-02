"""Download, verify, and extract the 58-class TSRD image dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = (
    "https://huggingface.co/datasets/EvanDu/Traffic_Sign_Recogntion_Database/"
    "resolve/main/TSRD/full_classes.zip?download=true"
)
EXPECTED_SHA256 = "6503cb960aae6c8ad52ebe690adbeeeb83220ef38c2a0188ae7b45a180e8a52c"
EXPECTED_COUNTS = {"train": 4170, "test": 1994}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Chinese TSRD dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--url", default=DEFAULT_URL, help="Override the dataset archive URL")
    parser.add_argument("--skip-sha256", action="store_true", help="Skip archive checksum validation")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (destination / item.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe path in archive: {item.filename}")
        zipped.extractall(destination)


def count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / "full_classes.zip"
    extracted = args.data_dir / "full_classes"

    if not archive.exists():
        print(f"Downloading TSRD from {args.url}", flush=True)
        with urllib.request.urlopen(args.url) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)

    if not args.skip_sha256:
        actual = sha256(archive)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"TSRD checksum mismatch: expected {EXPECTED_SHA256}, got {actual}"
            )

    if not extracted.exists():
        safe_extract(archive, args.data_dir)

    for split, expected in EXPECTED_COUNTS.items():
        actual = count_images(extracted / split)
        if actual != expected:
            raise RuntimeError(
                f"TSRD {split} count mismatch: expected {expected}, got {actual}"
            )
        print(f"{split}: {actual} images", flush=True)
    print(f"TSRD ready at {extracted}", flush=True)


if __name__ == "__main__":
    main()
