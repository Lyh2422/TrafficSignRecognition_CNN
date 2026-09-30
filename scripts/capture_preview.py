"""Capture a deterministic README screenshot with a real GTSRB sample."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent.parent

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1)
    page.goto("http://127.0.0.1:8000", wait_until="networkidle")
    page.locator("#fileInput").set_input_files(ROOT / "sample_images/14-stop.png")
    page.locator("#predictButton").click()
    page.locator("#resultContent").wait_for(state="visible")
    page.wait_for_timeout(700)
    page.screenshot(path=ROOT / "docs/system-preview.png", full_page=True)
    browser.close()

print("Saved docs/system-preview.png")

