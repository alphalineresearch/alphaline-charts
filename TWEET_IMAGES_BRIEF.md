# Brief — tweet images from the live charts

Read CLAUDE.md and BRAND.md first. Goal: one command turns a thread's image list into clean PNGs for X, built from the same chart files the website shows, so I never screenshot charts by hand.

## What to build: `scripts/tweet_images.py`

```
python scripts/tweet_images.py "C:\Users\cwhit\Documents\Alphaline Research\Tweets\2026-10 BTC Opportunity Signal\images.json"
```
Reads `images.json` and writes each PNG **next to it** (1.png, 2.png, ...).

### images.json format
```json
{
  "images": [
    { "file": "1.png",
      "chart": "btc_risk_score_history.html",
      "range": "all",
      "show": ["BTC Price", "Composite Score (14d avg)"],
      "title": "BTC Opportunity Signal",
      "subtitle": "Composite score, 0–100. 40+ marks a signal window." }
  ]
}
```
- `chart`: a file in `docs/`. Rebuild the figure from the Plotly JSON embedded in that HTML (data + layout) so no chart script needs changing.
- `range`: `all` | `5y` | `3y` | `1y` | `ytd` | `[start, end]` dates. Same windows as the range buttons; re-fit y like `_refit_y` does for thumbnails.
- `show`: trace names to keep, matched exactly as in the chart's legend (strip escape codes). Hide every other trace and its legend item. If `show` is missing, keep all. Unknown name → error listing the chart's trace names.
- Always keep shapes, bands and shading that belong to kept traces' panels; drop panels whose traces are all hidden and close the gap.
- `title` / `subtitle`: optional; default to the chart's own title/subtitle.

### Image style (BRAND.md)
- 1600 × 900 PNG, scale 2 (so 3200 × 1800 actual), navy `#0A1628` paper, `#102240` plot.
- Title top-left in Cormorant Garamond 700 ~44px, subtitle IBM Plex Sans ~22px `#B4C2CD`.
- Axis labels and legend ≥ 18px at 1600 wide (they get shrunk on phones). Legend only for kept traces, horizontal, under the plot.
- Footer: gold square + "ALPHALINE RESEARCH" left, "alphalineresearch.com · data to <manifest date>" right.
- No range buttons, no hover, no annotations except the latest-value gold marker.
- Render with the existing `_render_png` (kaleido first, Chrome fallback).

## Check before pushing
- Run it on the BTC Opportunity Signal thread's `images.json` (in the Tweets folder above): it should write 1.png–5.png.
- Open them: halving / hashrate / cost-proximity images show only that component plus BTC price; text readable when scaled to 600 px wide.
- Add a short "Tweet images" section to CLAUDE.md with the command and the JSON format.
- Show me the five images. Ask before pushing.
