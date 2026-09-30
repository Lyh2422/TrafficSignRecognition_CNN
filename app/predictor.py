"""Prediction service with real-CNN and clearly marked demo modes."""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .labels import LABEL_BY_ID
from .model import TrafficSignCNN, torch


class InvalidImageError(ValueError):
    pass


class Predictor:
    image_size = 32
    mean = np.array([0.3403, 0.3121, 0.3214], dtype=np.float32)
    std = np.array([0.2724, 0.2608, 0.2669], dtype=np.float32)

    def __init__(self) -> None:
        default_path = Path(__file__).resolve().parent.parent / "models" / "gtsrb_cnn.pt"
        self.model_path = Path(os.getenv("TSR_MODEL_PATH", str(default_path)))
        self.model = None
        self.metadata = {}
        self.mode = "demo"
        self.load_message = "未检测到训练权重，当前结果仅用于演示系统流程。"
        self._try_load_model()

    def _try_load_model(self) -> None:
        if torch is None or TrafficSignCNN is None:
            self.load_message = "PyTorch 未安装，当前结果仅用于演示系统流程。"
            return
        if not self.model_path.exists():
            return

        try:
            model = TrafficSignCNN(num_classes=len(LABEL_BY_ID))
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            if isinstance(checkpoint, dict):
                self.metadata = {
                    key: checkpoint[key]
                    for key in ("model_name", "model_version", "epoch", "val_accuracy", "val_macro_f1")
                    if key in checkpoint
                }
            self.mode = "cnn"
            accuracy = self.metadata.get("val_accuracy")
            suffix = f"，验证准确率 {accuracy:.2%}" if isinstance(accuracy, (int, float)) else ""
            self.load_message = f"已加载 CNN 权重：{self.model_path.name}{suffix}"
        except Exception as exc:  # Keep the site usable while surfacing the exact model problem.
            self.load_message = f"权重加载失败，已进入演示模式：{type(exc).__name__}"

    @staticmethod
    def decode_image(payload: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(payload))
            image.verify()
            image = Image.open(io.BytesIO(payload))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("文件不是有效的图片或图片已经损坏。") from exc

        if image.width < 16 or image.height < 16:
            raise InvalidImageError("图片尺寸过小，请上传至少 16×16 像素的图片。")
        if image.width * image.height > 20_000_000:
            raise InvalidImageError("图片像素过大，请上传不超过 2000 万像素的图片。")
        return image

    def predict(self, image: Image.Image, source_bytes: bytes) -> dict:
        started = time.perf_counter()
        if self.mode == "cnn" and self.model is not None:
            probabilities = self._cnn_probabilities(image)
        else:
            probabilities = self._demo_probabilities(image, source_bytes)

        indices = np.argsort(probabilities)[::-1][:3]
        predictions = []
        for index in indices:
            label = LABEL_BY_ID[int(index)]
            predictions.append({**label, "confidence": round(float(probabilities[index]), 6)})

        if self.mode != "cnn":
            warning = self.load_message
        elif predictions[0]["confidence"] < 0.55:
            warning = "模型置信度较低，请上传主体居中、裁剪紧凑且清晰的单个交通标志图片。"
        else:
            warning = None

        return {
            "prediction": predictions[0],
            "top3": predictions,
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
            "mode": self.mode,
            "warning": warning,
            "image": {"width": image.width, "height": image.height},
        }

    def _cnn_probabilities(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(tensor)
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    @staticmethod
    def _demo_probabilities(image: Image.Image, source_bytes: bytes) -> np.ndarray:
        """Return deterministic, visually plausible demo scores; never presented as CNN output."""
        thumbnail = image.resize((48, 48), Image.Resampling.BILINEAR)
        pixels = np.asarray(thumbnail, dtype=np.float32) / 255.0
        channel_mean = pixels.mean(axis=(0, 1))
        red, green, blue = channel_mean.tolist()

        seed = int.from_bytes(hashlib.sha256(source_bytes).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        logits = rng.normal(-2.1, 0.35, len(LABEL_BY_ID)).astype(np.float32)

        red_strength = max(0.0, red - (green + blue) / 2)
        blue_strength = max(0.0, blue - (red + green) / 2)
        yellow_strength = max(0.0, min(red, green) - blue)

        for class_id in [0, 1, 2, 3, 4, 5, 7, 8, 13, 14, 17]:
            logits[class_id] += 1.6 + red_strength * 8
        for class_id in [33, 34, 35, 36, 37, 38, 39, 40]:
            logits[class_id] += 1.5 + blue_strength * 9
        logits[12] += 1.2 + yellow_strength * 8

        # The hash makes repeated uploads stable while the color features affect the candidate family.
        logits[seed % len(LABEL_BY_ID)] += 0.9
        logits -= logits.max()
        exp = np.exp(logits)
        probabilities = exp / exp.sum()
        return probabilities
