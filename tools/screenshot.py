"""Capture docs/report.png from a briefly-started Streamlit app (needs `pip install playwright`
and `playwright install chromium`)."""
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(root / "app/streamlit_app.py"),
                         "--server.headless", "true", "--server.port", "8766"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1280, "height": 900})
        for _ in range(30):
            try:
                page.goto("http://localhost:8766"); break
            except Exception:
                time.sleep(1)
        page.get_by_text("Effect on order value").wait_for(timeout=60000)
        time.sleep(2)
        page.screenshot(path=str(root / "docs/report.png"))
        b.close()
finally:
    proc.terminate(); proc.wait(timeout=10)
