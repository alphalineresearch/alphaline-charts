"""
generate_manifest.py
Run this at the end of your existing cron/workflow script.
Scans docs/ for all HTML files (excluding index.html),
writes docs/manifest.json for the dashboard to consume.
"""

import json
import os
import glob
from datetime import date

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
# Retired charts: their docs/ page is now a redirect kept for old links
RETIRED = ("eth_stable_model_compact.html",)

def title_from_filename(filename):
    name = os.path.basename(filename).replace(".html", "")
    return name.replace("_", " ").replace("-", " ").title()

charts = []
for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.html"))):
    basename = os.path.basename(path)
    if basename in ("index.html", *RETIRED):
        continue
    charts.append({
        "file": basename,
        "title": title_from_filename(basename),
        "updated": str(date.today())
    })

manifest = {
    "updated": str(date.today()),
    "count": len(charts),
    "charts": charts
}

# Homepage Signal Monitor — embedded here so it ships with manifest.json
# (already committed by the workflow). Never let it break the manifest.
try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import generate_signals
    _prev = None
    try:
        with open(os.path.join(DOCS_DIR, "manifest.json"), encoding="utf-8") as _f:
            _prev = json.load(_f).get("signals")
    except Exception:
        pass
    _cwd = os.getcwd()
    os.chdir(os.path.join(DOCS_DIR, ".."))          # generate_signals reads docs/ relative to repo root
    try:
        _signals, _failed = generate_signals.build(_prev)
    finally:
        os.chdir(_cwd)
    if len(_signals.get("items", [])) >= 3:
        manifest["signals"] = _signals
    elif _prev:
        manifest["signals"] = _prev
except Exception as e:
    print(f"signals skipped: {e}")

out_path = os.path.join(DOCS_DIR, "manifest.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"manifest.json written — {len(charts)} charts")
