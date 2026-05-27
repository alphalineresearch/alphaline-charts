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

def title_from_filename(filename):
    name = os.path.basename(filename).replace(".html", "")
    return name.replace("_", " ").replace("-", " ").title()

charts = []
for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.html"))):
    basename = os.path.basename(path)
    if basename == "index.html":
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

out_path = os.path.join(DOCS_DIR, "manifest.json")
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"manifest.json written — {len(charts)} charts")
