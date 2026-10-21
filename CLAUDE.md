# Alphaline Charts — project context

Daily-updated interactive Plotly charts for **alphalineresearch.com** (independent Bitcoin & Ethereum research by Christopher White, CFA). Charts are built by Python scripts, published to GitHub Pages at `https://alphalineresearch.github.io/alphaline-charts/`, and embedded in the website by iframe.

## How it fits together
- `scripts/generate_*.py` — one script per chart (or chart family). Each fetches data (yfinance, DeFiLlama, blockchain.com — no API keys), builds a Plotly figure and writes `docs/<name>.html` with `include_plotlyjs='cdn'`.
- `.github/workflows/update-chart.yml` — runs daily at 08:00 UTC: all chart scripts → `generate_manifest.py` → commits `docs/`. The commit step uses an explicit `git add` list: **a new output file must be added to that list or it will never be published.**
- `scripts/generate_manifest.py` — lists `docs/*.html` into `docs/manifest.json`, and also embeds the homepage **Signal Monitor** data by calling `generate_signals.build()` (key `signals`).
- `scripts/generate_signals.py` — reads the finished chart HTML files, extracts the latest readings by **trace name**, and builds the six homepage tiles plus `thumbs` (3-year mini-chart series for the homepage cards). Treasury-company mNAV inputs (holdings, shares, cash, debt, preferred) live in the `TREASURIES` table at the top — updated weekly by hand.
- The website (separate folder, deployed to Cloudflare) fetches `manifest.json` and loads charts from GitHub Pages. File names and chart URLs are public — **never rename an output file.**

## Do not break
- **Trace names** that `generate_signals.py` reads: `Composite Score (14d avg)`, `Hash Rate`, `Hash Rate ATH`, `BTC Price`, `Production Cost (hist)`, `ETH Price`, `Model (combined TVL)`, `Stablecoins`, `DeFi TVL`, `RWA TVL`, `mNAV (z-score heatmap)`, `RSI(14)`. If you must rename one, update `generate_signals.py` in the same change.
- Output file names in `docs/`.
- The data and model logic. Styling changes should not change any numbers.
- After any change, run the script(s), then `python scripts/generate_manifest.py`, and check `docs/manifest.json` still has 6 `signals.items` and 6 `signals.thumbs`.

## Brand (match the website)
Master copy: `C:\Users\cwhit\Documents\Alphaline Research\BRAND.md` — read it for the full design system and the public model names. Summary:
- Colours: navy `#0A1628` (paper), navy-mid `#102240` (plot), gold `#D4A843`, gold-light `#ECC96A`, gold-dim `#8A6B25`, white `#F8FAFB`, text-2 `#B4C2CD`, text-3 `#8FA2B2`, steel `#374D61` (borders only), green `#2ABF7A`, red `#E86A6A` / `#D64444`, ETH blue `#8FAEE6`.
- Fonts on the site: **IBM Plex Sans** (UI/body), **Cormorant Garamond** 700 (headings), IBM Plex Mono (numbers only, sparingly). Nothing smaller than 12px.
- Conventions: log scale on BTC/ETH price axes; regime shading green = opportunity, red = stretched; z-score bars red if positive, green if negative; finance-style negatives in labels, e.g. (3.9%).
- Gridlines `rgba(212,168,67,0.06)`; square corners; no heavy borders.

## Environment
- Conda env `alphaline` (Python 3.11): requests, pandas, numpy, plotly, yfinance, scikit-learn, scipy.
- DeFiLlama timestamps: `pd.Timestamp(int(x), unit='s').normalize()`; strip tz with `.dt.tz_convert(None)`.

## Weekly upkeep (Mondays)
- `generate_mstr_pnav_heatmap.py`: append Strategy's latest holdings / shares / capital rows (strategy.com ledger, 8-K).
- `generate_signals.py` `TREASURIES`: update MSTR, ASST, BMNR, SBET holdings, shares, cash, debt, preferred.
