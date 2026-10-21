# Chart restyle brief

Restyle all the charts so they match the new alphalineresearch.com look and behave consistently. Read CLAUDE.md first.

## Plan
1. Create `scripts/alphaline_theme.py` — one shared module every chart script uses instead of its own `alphaline_layout()`. It should handle:
   - **Fonts:** IBM Plex Sans for axes/legend/hover, Cormorant Garamond 700 for the chart title. Load them from Google Fonts inside each output HTML so they actually render. Minimum 12px anywhere; axis ticks/legend 12–13px, hover 13px, title ~20px.
   - **Layout:** title + live-reading subtitle top-left, range buttons top-right, legend in one row under the plot, source line + small Alphaline mark at the bottom. Same margins, gridlines and hover format ("Oct 19, 2026 · $81,562") on every chart.
   - **Range buttons** on every time-series chart: YTD, 1Y, 3Y, 5Y, All — styled like the site (square, gold when active). Add a small `post_script` so the y-axis refits to the visible window when a button is pressed or the user zooms (log axes included). Skip buttons on the 30-day momentum charts.
   - **Default window:** 3Y for valuation/signal charts; All for long-cycle charts (annual RSI, BB squeeze × copper/gold).
   - **Embed mode:** when the chart is loaded inside an iframe, hide the in-chart title and tighten the top margin (the website already shows the title). Standalone / full screen keeps the title and branding.
   - **Phones:** fewer ticks, wrapping legend, hide the Plotly modebar on small screens.
   - A `write_chart(fig, path)` helper that does the font loading, post_script and config in one place.
2. **Pilot on two charts only:** `generate_btc_production_cost_charts.py` (BTC Opportunity Signal / risk score charts) and `generate_eth_combined_tvl.py` (ETH Model Price). Run them on real data, open the HTML, and show me screenshots at desktop and phone width before going further.
3. After I approve, convert the other chart scripts the same way, one at a time, running each and checking the output.

## Rules
- Don't change any data, model logic, output file names, or the trace names listed in CLAUDE.md.
- After each chart, run `generate_manifest.py` and confirm `docs/manifest.json` still has 6 `signals.items` and 6 `signals.thumbs`.
- Commit in small steps with clear messages, and ask me before pushing.
