# Brief — chart catalog in manifest.json

Read CLAUDE.md first. Goal: the website's charts page builds itself from `docs/manifest.json`, so a new chart shows up on alphalineresearch.com/charts without editing the website.

## What to build
1. **A catalog file in the repo**: `scripts/catalog.py` (or `catalog.json`) with one entry per public chart, in display order:
   ```
   { "s": "btc", "g": "val", "id": "btc-cost", "file": "btc_price_vs_production_cost.html",
     "h": 800, "t": "BTC Price vs. Production Cost",
     "d": "Price against the estimated cost to mine one bitcoin, with a base-case cost path to 2032.",
     "read": "btc-production-cost-editorial.html" }
   ```
   - `s` segment: `btc` Bitcoin · `eth` Ethereum · `mkt` Total Crypto Market · `trs` Treasury Companies
   - `g` group: `val` Valuation · `sig` Signals · `mom` Momentum & trend · `flow` Flows & credit
   - `id` = the website anchor (`charts.html#<id>`). **Never change an existing id** — older links depend on them.
   - `h` = desktop iframe height in px; `read` = optional article page on the website.
   - Copy the current 25 entries exactly from the `CHARTS` list in the website's `charts.html` (Website\Drafts\charts-v3.html) — same ids, titles, descriptions, segments and groups.
2. **`generate_manifest.py`** writes that list to `manifest.json` as `catalog` (keep `charts` and `signals` as they are). Skip any entry whose HTML file is missing from `docs/`, and print a warning.
3. **Optional, nice to have:** `reading` on an entry — a short live line like `"vs. production cost: +30%"` — for charts that already have a Signal Monitor value. The page currently maps these itself, so leave it out if it adds work.
4. Adding a chart from now on = new script + workflow step + one catalog entry. Add that to CLAUDE.md.

## Check before pushing
- `python scripts/generate_manifest.py` → `catalog` has 25 entries, every `file` exists in `docs/`, every `docs/thumbs/<stem>.webp` exists.
- `signals.items` still 6, `signals.thumbs` still 6.
- Ask before pushing.

## Website side (already handled)
The draft charts page uses `manifest.catalog` when present and falls back to its built-in list otherwise, so nothing breaks before or after this lands.
