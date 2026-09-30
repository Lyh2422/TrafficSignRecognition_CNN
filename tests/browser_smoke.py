"""Headless browser smoke test for the complete upload-to-history flow."""

import io
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / ".artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def sample_sign() -> bytes:
    output = io.BytesIO()
    image = Image.new("RGB", (320, 320), "#ece9df")
    draw = ImageDraw.Draw(image)
    draw.ellipse((35, 35, 285, 285), fill="#f8f7f1", outline="#e23b25", width=34)
    draw.text((115, 125), "50", fill="#171914", stroke_width=2)
    image.save(output, format="PNG")
    return output.getvalue()


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    console_errors = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)

    page.goto("http://127.0.0.1:8000", wait_until="networkidle")
    mode_badge = page.locator("#modeBadge").inner_text()
    assert mode_badge in {"DEMO MODE", "CNN ONLINE"}
    assert page.locator("h1").is_visible()

    page.locator("#fileInput").set_input_files(
        {"name": "speed-limit-50.png", "mimeType": "image/png", "buffer": sample_sign()}
    )
    page.locator("#predictButton").click()
    page.locator("#resultContent").wait_for(state="visible")
    page.locator("#historyBody tr").first.wait_for(state="visible")

    assert page.locator("#className").inner_text() != "—"
    if mode_badge == "CNN ONLINE":
        assert "CNN" in page.locator("#resultMode").inner_text()
        if page.locator("#modeWarning").is_visible():
            assert "置信度较低" in page.locator("#modeWarning").inner_text()
    else:
        assert "演示模式" in page.locator("#resultMode").inner_text()
        assert page.locator("#modeWarning").is_visible()
    assert "speed-limit-50.png" in page.locator("#historyBody tr").first.inner_text()
    page.wait_for_timeout(600)
    page.screenshot(path=ARTIFACTS / "mvp-after-predict.png", full_page=True)

    mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    mobile.goto("http://127.0.0.1:8000", wait_until="networkidle")
    overflow = mobile.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
    assert not overflow
    assert mobile.locator("#dropZone").is_visible()
    mobile.screenshot(path=ARTIFACTS / "mvp-mobile.png", full_page=True)

    browser.close()
    assert not console_errors, f"Browser console errors: {console_errors}"

print("Browser smoke test passed: desktop upload flow and mobile layout verified.")
