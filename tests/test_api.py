import io
import os
from pathlib import Path

from PIL import Image

TEST_DB = Path(__file__).parent / "test_recognitions.db"
os.environ["TSR_DB_PATH"] = str(TEST_DB)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def make_png(color=(220, 40, 30)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


def setup_function():
    client.delete("/api/history")


def teardown_module():
    TEST_DB.unlink(missing_ok=True)


def test_health_reports_58_chinese_classes():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["class_count"] == 58


def test_predict_and_history_round_trip():
    response = client.post("/api/predict", files={"file": ("sign.png", make_png(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert len(result["top3"]) == 3
    assert result["mode"] in {"demo", "cnn"}
    assert 0 <= result["prediction"]["confidence"] <= 1
    assert result["decision"] in {"recognized", "uncertain"}
    assert isinstance(result["recognized"], bool)
    assert result["prediction"]["category"] in {"限速", "禁令", "指示", "警告"}

    history = client.get("/api/history").json()["items"]
    assert len(history) == 1
    assert history[0]["filename"] == "sign.png"


def test_rejects_non_image_file():
    response = client.post("/api/predict", files={"file": ("note.txt", b"not an image", "text/plain")})
    assert response.status_code == 415


def test_rejects_corrupted_image():
    response = client.post("/api/predict", files={"file": ("bad.png", b"broken", "image/png")})
    assert response.status_code == 422
