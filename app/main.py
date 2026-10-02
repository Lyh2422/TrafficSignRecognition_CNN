"""FastAPI entry point for the traffic sign recognition MVP."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .database import HistoryRepository
from .labels import DATASET_INFO, TSRD_LABELS
from .predictor import InvalidImageError, Predictor

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = FastAPI(
    title="中国交通标志识别系统",
    description="基于卷积神经网络和 TSRD 数据集的中国交通标志识别接口",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predictor = Predictor()
history = HistoryRepository()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": predictor.mode,
        "model_message": predictor.load_message,
        "model_metadata": predictor.metadata,
        "class_count": len(TSRD_LABELS),
        "dataset": DATASET_INFO,
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="仅支持 JPG、PNG、WebP 等图片文件。")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="上传的文件为空。")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 8 MB。")

    try:
        image = predictor.decode_image(payload)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = predictor.predict(image, payload)
    result["record_id"] = history.add(file.filename or "未命名图片", result)
    return result


@app.get("/api/history")
def list_history(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return {"items": history.list(limit)}


@app.delete("/api/history")
def clear_history() -> dict:
    return {"deleted": history.clear()}
