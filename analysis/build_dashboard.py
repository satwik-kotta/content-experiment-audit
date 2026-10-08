"""Inject outputs/*.json into the dashboard template -> dashboard/index.html"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "dashboard/template.html").read_text()
html = html.replace("__EXP__", (ROOT / "outputs/experiments.json").read_text())
html = html.replace("__HEAD__", (ROOT / "outputs/headlines.json").read_text())
(ROOT / "dashboard/index.html").write_text(html)
print("wrote dashboard/index.html")
