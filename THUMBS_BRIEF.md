# Brief — chart thumbnails for the homepage

Read CLAUDE.md first. Goal: the homepage "More charts" cards should look like the real charts. Each chart also saves a small, clean image of its actual plot every day.

## What to build
1. **Export in `write_chart()`** (`scripts/alphaline_theme.py`) so every chart gets it with no per-script changes:
   - Save `docs/thumbs/<same stem as the chart file>.webp`, e.g. `docs/thumbs/eth_combined_tvl.webp`.
   - Size **960 × 540** (16:9), WebP quality ~80. Target **≤ 60 KB** each.
   - Render a **copy** of the figure — never modify the figure that goes into the HTML.
   - Show the chart's **default window** (the same range the HTML opens on), so the thumbnail matches what people see on click.
   - Strip the small print: no title, subtitle, legend, annotations, range buttons, axis titles or tick labels; keep gridlines faint. Keep every trace, band, fill, bar and shading shape in its real colours.
   - Multi-panel charts: keep all panels, but tighten vertical spacing and margins (~8px) so the plot fills the frame.
   - Background navy `#0A1628` / plot `#102240` as in the theme.
   - If export fails, print a warning and carry on — a thumbnail must never break a chart build.
2. **Dependencies**: add `kaleido` to `requirements.txt` (v1 uses a Chrome install; on GitHub's ubuntu runner Chrome is present — if needed call `kaleido.get_chrome_sync()` once). Confirm it works locally too.
3. **Workflow** (`.github/workflows/update-chart.yml`): add `docs/thumbs/` to the commit step (e.g. a separate line `git add docs/thumbs 2>/dev/null || true` before the commit).
4. Do **not** change chart HTML output, file names, trace names or data.

## Check before pushing
- Run all chart scripts, then `python scripts/generate_manifest.py`.
- `docs/thumbs/` should contain one `.webp` per chart (21), each ≤ 60 KB; open a few and confirm they match the charts' default view.
- `docs/manifest.json` still has 6 `signals.items` and 6 `signals.thumbs`.
- Show me 6 of them (the homepage set): btc_risk_score_history, btc_price_vs_production_cost, eth_combined_tvl, eth_stacked_tvl_mcap, mstr_pnav_heatmap, btc_rsi_weekly. Ask before pushing.

## Website side (already handled)
The homepage loads `https://alphalineresearch.github.io/alphaline-charts/thumbs/<stem>.webp` for each card and falls back to the drawn line sketch if an image is missing — no website change is needed when these appear.
