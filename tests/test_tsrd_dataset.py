from pathlib import Path

from PIL import Image

from app.labels import DATASET_INFO, LABEL_BY_ID, TSRD_LABELS
from scripts.tsrd_dataset import TSRDDataset


def test_tsrd_label_ids_are_complete_and_contiguous():
    assert len(TSRD_LABELS) == 58
    assert list(LABEL_BY_ID) == list(range(58))
    assert DATASET_INFO["region"] == "中国"
    assert LABEL_BY_ID[0]["name_zh"] == "限速5km/h"
    assert LABEL_BY_ID[55]["name_zh"] == "禁止驶入"


def test_tsrd_dataset_reads_class_id_from_filename(tmp_path: Path):
    Image.new("RGB", (32, 32), "red").save(tmp_path / "055_0001.png")

    dataset = TSRDDataset(tmp_path)

    image, label = dataset[0]
    assert image.mode == "RGB"
    assert label == 55
    assert dataset.targets == [55]
    assert dataset.class_counts[55] == 1


def test_tsrd_dataset_crops_to_annotation_box(tmp_path: Path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    Image.new("RGB", (100, 80), "red").save(train_dir / "003_0001.png")
    (tmp_path / "TsignRecgTrainAnnotation.txt").write_text(
        "003_0001.png;100;80;20;10;79;69;3;\n", encoding="utf-8"
    )

    dataset = TSRDDataset(train_dir, crop_padding=0)
    image, label = dataset[0]

    assert label == 3
    assert image.size == (60, 60)
