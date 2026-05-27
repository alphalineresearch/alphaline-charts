"""
Copies generated chart HTML files from docs/ to the website directory.
Run after the GitHub Actions workflow pushes new charts (git pull first),
or after running chart scripts locally.

Usage:
    python scripts/sync_to_website.py
"""

import shutil, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC  = REPO_ROOT / "docs"
DEST = Path(r"C:\Users\cwhit\Documents\Alphaline Research\Website\New")

CHARTS = [
    "eth_stable_model_compact.html",
    "eth_combined_tvl.html",
    "eth_stacked_tvl.html",
    "eth_proxy_signal.html",
    "eth_tvl_momentum.html",
    "eth_combined_components_tvl.html",
    "eth_model_zscore.html",
    "eth_stacked_tvl_mcap.html",
    "btc_rv7_rv30_proxy_signal.html",
    "mstr_pnav_heatmap.html",
    "btc_rsi_weekly.html",
    "eth_rsi_weekly.html",
    "btc_rsi_annual.html",
    "eth_rsi_annual.html",
    "btc_cost_momentum.html",
]

copied = 0
for name in CHARTS:
    s = SRC / name
    d = DEST / name
    if s.exists():
        shutil.copy2(s, d)
        print(f"  copied  {name}")
        copied += 1
    else:
        print(f"  MISSING {s}", file=sys.stderr)

print(f"\n{copied}/{len(CHARTS)} files synced to {DEST}")
