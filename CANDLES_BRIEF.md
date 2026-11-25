# Brief — monthly candles (pilot on two charts)

Read CLAUDE.md and BRAND.md first. Goal: try monthly OHLC candles where they suit the signal, starting with one chart, then a second if it looks right. Everything else keeps its line.

## Step 1 — BTC Squeeze × Copper/Gold (`generate_btc_copper_gold_bb_squeeze.py` → `btc_copper_gold_bb_squeeze.html`)
The squeeze is measured on monthly Bollinger Bands, so draw the BTC price panel as **monthly candles inside those bands**.

1. **Monthly OHLC**: resample daily data to month-end bars — open = first, high = max, low = min, close = last.
   - From Yahoo's start onward, use real daily high/low.
   - Before that (blockchain.info, closes only), build high/low from daily closes. Add one line to the header note: *"Before 2014, monthly highs and lows use daily closes."*
   - Keep the current (partial) month as the last candle; its hover should say "month to date".
2. **Bands and squeeze**: unchanged. Bands should be computed on the same monthly closes the candles use, so they line up exactly.
3. **Candle style** (brand rules, square look):
   - Up month: body and wick `#2ABF7A`. Down month: `#D64444`. Thin wicks (1px), no gaps-for-weekends logic needed on monthly.
   - Keep the gold latest-value marker on the last close.
   - Price axis stays log.
   - Hover: date as "Oct 2026", then O / H / L / C in $ with commas, and the month's % change in finance format, e.g. (12.4%).
4. **Range buttons**: keep YTD / 1Y / 3Y / 5Y / All, but **default to All or 5Y**; on YTD and 1Y there are only 10–12 bars, which is fine.
5. **Thumbnail**: `write_chart()` already exports it; check the candles read at 960×540.

## Step 2 — only after I've seen Step 1: BTC Trend and ETH Trend (`generate_trend_charts.py`)
Same monthly candles on the price panel, but **colour each candle by its trend score** at month-end, not by up/down — that's the chart's message:
- 3–4 of 4 positive: `#2ABF7A` · 2 of 4: `#8FA2B2` · 0–1 of 4: `#D64444`.
- Keep the 200-day and 200-week MAs as lines, the Mayer panel as-is, and the shading zones.
- Same pre-2014 note for BTC. ETH has real OHLC from Aug 2015.
- Hover adds "Trend 3/4 · Uptrend".

## Don't change
- File names, chart ids, the website anchors, `generate_signals.py`, or any model chart (production cost, ETH model, Opportunity Signal, Recency). Those stay daily lines.

## Check before pushing
- Render at desktop (iframe ~1,100 × 1,110 for Squeeze) and phone (390 × 680) — no cut-off legend or footer, candles still visible.
- Show me screenshots of Step 1 (All view and 1Y view, desktop and phone). Ask before pushing, and before starting Step 2.
