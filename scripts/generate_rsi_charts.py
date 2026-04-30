"""
generate_rsi_charts.py
Alphaline Research — BTC & ETH Weekly RSI(14) + Annual RSI(52)

Data sources (no API keys required):
  - Yahoo Finance: BTC-USD and ETH-USD weekly price history (via yfinance)

Writes four files to docs/:
  docs/btc_rsi_weekly.html
  docs/eth_rsi_weekly.html
  docs/btc_rsi_annual.html
  docs/eth_rsi_annual.html

Usage:
    python generate_rsi_charts.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_BTC_WEEKLY = os.path.join('docs', 'btc_rsi_weekly.html')
OUTPUT_ETH_WEEKLY = os.path.join('docs', 'eth_rsi_weekly.html')
OUTPUT_BTC_ANNUAL = os.path.join('docs', 'btc_rsi_annual.html')
OUTPUT_ETH_ANNUAL = os.path.join('docs', 'eth_rsi_annual.html')

RSI_PERIOD     = 14
RSI_1YR_PERIOD = 52
PCT_RARE_OS    = 12
PCT_RARE_OB    = 88

# ════════════════════════════════════════════
# BRAND COLORS
# ════════════════════════════════════════════
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
AMBER     = '#C87820'
GREEN_LIT = '#2ABF7A'
RED_LIT   = '#D64444'

CHART_WIDTH  = 1100
CHART_HEIGHT = 820

def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'

# ════════════════════════════════════════════
# ALPHALINE LAYOUT
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT,
                     source='alphalineresearch.com  |  Yahoo Finance'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, autosize=True,
        title=dict(
            text=f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>',
            x=0.02, xanchor='left', y=0.98, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=60, r=120, t=70, b=145),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL, borderwidth=1,
                    font=dict(size=9, color=MIST), x=0.01, y=0.99,
                    xanchor='left', yanchor='top'),
        annotations=[
            dict(text=f'Source: {source}',
                 xref='paper', yref='paper', x=1.0, y=-0.13,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=MIST),
                 showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>',
                 xref='paper', yref='paper', x=0.0, y=-0.13,
                 xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD),
                 showarrow=False),
        ],
    )
    return fig

# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def build_episodes(mask_s, gap_weeks=1):
    """Group consecutive True weeks into (start, end) episodes."""
    episodes, dates = [], mask_s.index[mask_s].tolist()
    if not dates:
        return episodes
    ep_s = ep_e = dates[0]
    for dt in dates[1:]:
        if (dt - ep_e).days <= 7 * (gap_weeks + 1):
            ep_e = dt
        else:
            episodes.append((ep_s, ep_e))
            ep_s = ep_e = dt
    episodes.append((ep_s, ep_e))
    return episodes

def assign_rsi_colors(pct_series, slope_series):
    """4-state color scheme: GREEN/AMBER (OS), GOLD_LIT/RED_LIT (OB), STEEL (neutral)."""
    colors = []
    for pct, slope in zip(pct_series, slope_series):
        if pd.isna(pct) or pd.isna(slope):
            colors.append(STEEL)
        elif pct <= PCT_RARE_OS:
            colors.append(GREEN_LIT if slope > 0 else AMBER)
        elif pct >= PCT_RARE_OB:
            colors.append(RED_LIT if slope < 0 else GOLD_LIT)
        else:
            colors.append(STEEL)
    return colors

# ════════════════════════════════════════════
# DATA FETCHER
# ════════════════════════════════════════════
def fetch_weekly(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period='max', interval='1wk', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = 'date'
    return df.dropna(subset=['close'])

def build_dataframe(ticker: str) -> pd.DataFrame:
    df = fetch_weekly(ticker)
    df['rsi']           = compute_rsi(df['close'], period=RSI_PERIOD)
    df['rsi_1yr']       = compute_rsi(df['close'], period=RSI_1YR_PERIOD)
    df['rsi_pct']       = df['rsi'].rank(pct=True) * 100
    df['rsi_1yr_pct']   = df['rsi_1yr'].rank(pct=True) * 100
    df['rsi_slope']     = df['rsi'].diff(3)
    df['rsi_1yr_slope'] = df['rsi_1yr'].diff(3)
    return df

# ════════════════════════════════════════════
# CHART 1 & 2 — WEEKLY PRICE + RSI(14)
# ════════════════════════════════════════════
def plot_price_rsi(df: pd.DataFrame, ticker: str, price_color: str, rsi_range=None):
    if rsi_range is None:
        rsi_range = [0, 100]
    d = df.dropna(subset=['rsi', 'rsi_pct', 'rsi_slope']).copy()

    lvl_rare_os = float(d['rsi'].quantile(PCT_RARE_OS / 100))
    lvl_rare_ob = float(d['rsi'].quantile(PCT_RARE_OB / 100))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.03
    )

    # ── Vrect shading on price panel for rare extremes ─────────────
    for mask, color in [
        (d['rsi_pct'] <= PCT_RARE_OS, GREEN_LIT),
        (d['rsi_pct'] >= PCT_RARE_OB, RED_LIT),
    ]:
        for ep_s, ep_e in build_episodes(mask):
            fig.add_vrect(
                x0=ep_s, x1=ep_e,
                fillcolor=hex_to_rgba(color, 0.13),
                line_width=0.5, line_color=hex_to_rgba(color, 0.25),
                layer='below', row=1, col=1
            )

    # ── Panel 1: Price line ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=d.index, y=d['close'],
        mode='lines', name=f'{ticker} Price',
        line=dict(color=price_color, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>' + ticker + ': $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_annotation(
        x=d.index[-1], y=d['close'].iloc[-1],
        text=f'  ${d["close"].iloc[-1]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=price_color),
        row=1, col=1
    )

    # ── Colored triangles at rare RSI extremes ─────────────────────
    for mask, symbol, mult, label in [
        (d['rsi_pct'] <= PCT_RARE_OS, 'triangle-up',   0.95, f'Rare OS (pct ≤{PCT_RARE_OS})'),
        (d['rsi_pct'] >= PCT_RARE_OB, 'triangle-down', 1.05, f'Rare OB (pct ≥{PCT_RARE_OB})'),
    ]:
        sub = d[mask]
        if not sub.empty:
            tri_colors = assign_rsi_colors(sub['rsi_pct'], sub['rsi_slope'])
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub['close'] * mult,
                mode='markers', name=label,
                marker=dict(symbol=symbol, color=tri_colors, size=9,
                            line=dict(width=0)),
                showlegend=True,
                hovertemplate=label + '<br>%{x|%Y-%m-%d}<extra></extra>'
            ), row=1, col=1)

    # ── Panel 2: RSI background bands ─────────────────────────────
    fig.add_hrect(y0=rsi_range[0], y1=lvl_rare_os, fillcolor=hex_to_rgba(GREEN_LIT, 0.18), line_width=0, row=2, col=1)
    fig.add_hrect(y0=lvl_rare_ob,  y1=rsi_range[1], fillcolor=hex_to_rgba(RED_LIT,   0.15), line_width=0, row=2, col=1)

    for level, label, color, dash in [
        (lvl_rare_ob, f'Rare OB {lvl_rare_ob:.0f} ({PCT_RARE_OB}th pct)', RED_LIT,   'dot'),
        (50,          'Mid 50',                                              STEEL,     'dot'),
        (lvl_rare_os, f'Rare OS {lvl_rare_os:.0f} ({PCT_RARE_OS}th pct)', GREEN_LIT, 'dot'),
    ]:
        if rsi_range[0] <= level <= rsi_range[1]:
            fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=0.8, row=2, col=1)
            fig.add_annotation(
                x=d.index[-1], y=level,
                text=f'  {label}', showarrow=False, xanchor='left',
                font=dict(family='Courier New, monospace', size=8, color=color),
                row=2, col=1
            )

    # ── RSI(14) thin background line + colored dots ────────────────
    fig.add_trace(go.Scatter(
        x=d.index, y=d['rsi'],
        mode='lines', name='RSI(14)',
        line=dict(color=MIST, width=0.8), opacity=0.35,
        showlegend=True, hoverinfo='skip'
    ), row=2, col=1)

    dot_colors = assign_rsi_colors(d['rsi_pct'], d['rsi_slope'])
    fig.add_trace(go.Scatter(
        x=d.index, y=d['rsi'],
        mode='markers',
        marker=dict(color=dot_colors, size=5, line=dict(width=0)),
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>RSI(14): %{y:.1f}<extra></extra>'
    ), row=2, col=1)

    # ── Current RSI label + regime ─────────────────────────────────
    rsi_now   = d['rsi'].iloc[-1]
    pct_now   = d['rsi_pct'].iloc[-1]
    slope_now = d['rsi_slope'].iloc[-1]
    color_now = assign_rsi_colors([pct_now], [slope_now])[0]
    regime_label = (
        'RARE OS · RECOVERING' if pct_now <= PCT_RARE_OS and slope_now > 0 else
        'RARE OS · FALLING'    if pct_now <= PCT_RARE_OS else
        'RARE OB · TOPPING'    if pct_now >= PCT_RARE_OB and slope_now < 0 else
        'RARE OB · RUNNING'    if pct_now >= PCT_RARE_OB else
        'NEUTRAL'
    )

    fig.add_annotation(
        x=d.index[-1], y=rsi_now,
        text=f'  {rsi_now:.1f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=color_now),
        row=2, col=1
    )

    title = (
        f'{ticker} — Weekly Price & RSI(14)  |  '
        f'RSI: {rsi_now:.1f}  ·  {pct_now:.0f}th pct  ·  [{regime_label}]'
    )
    alphaline_layout(fig, title, height=CHART_HEIGHT)
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                    font=dict(size=9, color=MIST),
                    bgcolor='rgba(10,22,40,0.85)', bordercolor=STEEL, borderwidth=1)
    )
    fig.update_yaxes(type='log', tickprefix='$', title_text='Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='RSI(14)', range=rsi_range,
                     tickvals=[20, 30, 50, 70, 80, 90, 100], row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False)

    return fig, regime_label

# ════════════════════════════════════════════
# CHART 3 & 4 — ANNUAL RSI(52) CYCLE
# ════════════════════════════════════════════
def plot_annual_rsi(df: pd.DataFrame, ticker: str, price_color: str, rsi_range=None):
    if rsi_range is None:
        rsi_range = [0, 100]
    d = df.dropna(subset=['rsi_1yr', 'rsi_1yr_pct', 'rsi_1yr_slope']).copy()

    lvl_rare_os = float(d['rsi_1yr'].quantile(PCT_RARE_OS / 100))
    lvl_rare_ob = float(d['rsi_1yr'].quantile(PCT_RARE_OB / 100))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.03
    )

    # ── Vrect shading on price panel ──────────────────────────────
    for mask, color in [
        (d['rsi_1yr_pct'] <= PCT_RARE_OS, GREEN_LIT),
        (d['rsi_1yr_pct'] >= PCT_RARE_OB, RED_LIT),
    ]:
        for ep_s, ep_e in build_episodes(mask):
            fig.add_vrect(
                x0=ep_s, x1=ep_e,
                fillcolor=hex_to_rgba(color, 0.13),
                line_width=0.5, line_color=hex_to_rgba(color, 0.25),
                layer='below', row=1, col=1
            )

    # ── Panel 1: Price line ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=d.index, y=d['close'],
        mode='lines', name=f'{ticker} Price',
        line=dict(color=price_color, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>' + ticker + ': $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_annotation(
        x=d.index[-1], y=d['close'].iloc[-1],
        text=f'  ${d["close"].iloc[-1]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=price_color),
        row=1, col=1
    )

    # ── Colored triangles driven by RSI(52) pct + slope ───────────
    for mask, symbol, mult, label in [
        (d['rsi_1yr_pct'] <= PCT_RARE_OS, 'triangle-up',   0.95, f'Rare OS RSI(52) (pct ≤{PCT_RARE_OS})'),
        (d['rsi_1yr_pct'] >= PCT_RARE_OB, 'triangle-down', 1.05, f'Rare OB RSI(52) (pct ≥{PCT_RARE_OB})'),
    ]:
        sub = d[mask]
        if not sub.empty:
            tri_colors = assign_rsi_colors(sub['rsi_1yr_pct'], sub['rsi_1yr_slope'])
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub['close'] * mult,
                mode='markers', name=label,
                marker=dict(symbol=symbol, color=tri_colors, size=9,
                            line=dict(width=0)),
                showlegend=True,
                hovertemplate=label + '<br>%{x|%Y-%m-%d}<extra></extra>'
            ), row=1, col=1)

    # ── Panel 2: RSI(52) bands ─────────────────────────────────────
    fig.add_hrect(y0=rsi_range[0], y1=lvl_rare_os, fillcolor=hex_to_rgba(GREEN_LIT, 0.18), line_width=0, row=2, col=1)
    fig.add_hrect(y0=lvl_rare_ob,  y1=rsi_range[1], fillcolor=hex_to_rgba(RED_LIT,   0.15), line_width=0, row=2, col=1)

    for level, label, color, dash in [
        (lvl_rare_ob, f'Rare OB {lvl_rare_ob:.0f} ({PCT_RARE_OB}th pct)', RED_LIT,   'dot'),
        (50,          'Mid 50',                                              STEEL,     'dot'),
        (lvl_rare_os, f'Rare OS {lvl_rare_os:.0f} ({PCT_RARE_OS}th pct)', GREEN_LIT, 'dot'),
    ]:
        if rsi_range[0] <= level <= rsi_range[1]:
            fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=0.8, row=2, col=1)
            fig.add_annotation(
                x=d.index[-1], y=level,
                text=f'  {label}', showarrow=False, xanchor='left',
                font=dict(family='Courier New, monospace', size=8, color=color),
                row=2, col=1
            )

    # ── RSI(52) thin line + colored dots ──────────────────────────
    fig.add_trace(go.Scatter(
        x=d.index, y=d['rsi_1yr'],
        mode='lines', name='RSI(52)',
        line=dict(color=MIST, width=0.8), opacity=0.35,
        showlegend=True, hoverinfo='skip'
    ), row=2, col=1)

    dot_colors = assign_rsi_colors(d['rsi_1yr_pct'], d['rsi_1yr_slope'])
    fig.add_trace(go.Scatter(
        x=d.index, y=d['rsi_1yr'],
        mode='markers',
        marker=dict(color=dot_colors, size=5, line=dict(width=0)),
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>RSI(52): %{y:.1f}<extra></extra>'
    ), row=2, col=1)

    # ── Current labels ─────────────────────────────────────────────
    rsi_now   = d['rsi_1yr'].iloc[-1]
    pct_now   = d['rsi_1yr_pct'].iloc[-1]
    slope_now = d['rsi_1yr_slope'].iloc[-1]
    color_now = assign_rsi_colors([pct_now], [slope_now])[0]
    regime_label = (
        'RARE OS · RECOVERING' if pct_now <= PCT_RARE_OS and slope_now > 0 else
        'RARE OS · FALLING'    if pct_now <= PCT_RARE_OS else
        'RARE OB · TOPPING'    if pct_now >= PCT_RARE_OB and slope_now < 0 else
        'RARE OB · RUNNING'    if pct_now >= PCT_RARE_OB else
        'NEUTRAL'
    )

    fig.add_annotation(
        x=d.index[-1], y=rsi_now,
        text=f'  {rsi_now:.1f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=color_now),
        row=2, col=1
    )

    title = (
        f'{ticker} — Annual RSI(52) Cycle  |  '
        f'RSI(52): {rsi_now:.1f}  ·  {pct_now:.0f}th pct  ·  [{regime_label}]'
    )
    alphaline_layout(fig, title, height=CHART_HEIGHT)
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                    font=dict(size=9, color=MIST),
                    bgcolor='rgba(10,22,40,0.85)', bordercolor=STEEL, borderwidth=1)
    )
    fig.update_yaxes(type='log', tickprefix='$', title_text='Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='RSI(52)', range=rsi_range,
                     tickvals=[20, 30, 50, 70, 80, 90, 100], row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False)

    return fig, regime_label

# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building BTC & ETH RSI Charts ===')

    print('\nFetching BTC weekly...')
    btc = build_dataframe('BTC-USD')
    print(f'  BTC: {len(btc)} weeks | {btc.index[0].date()} → {btc.index[-1].date()} | ${btc["close"].iloc[-1]:,.0f}')

    print('Fetching ETH weekly...')
    eth = build_dataframe('ETH-USD')
    print(f'  ETH: {len(eth)} weeks | {eth.index[0].date()} → {eth.index[-1].date()} | ${eth["close"].iloc[-1]:,.0f}')

    print('\nGenerating BTC Weekly RSI(14)...')
    fig, regime = plot_price_rsi(btc, 'BTC', GOLD, rsi_range=[20, 100])
    fig.write_html(OUTPUT_BTC_WEEKLY, include_plotlyjs='cdn', config={'responsive': True})
    print(f'  Saved: {OUTPUT_BTC_WEEKLY}  [{regime}]')

    print('Generating ETH Weekly RSI(14)...')
    fig, regime = plot_price_rsi(eth, 'ETH', MIST, rsi_range=[20, 100])
    fig.write_html(OUTPUT_ETH_WEEKLY, include_plotlyjs='cdn', config={'responsive': True})
    print(f'  Saved: {OUTPUT_ETH_WEEKLY}  [{regime}]')

    print('Generating BTC Annual RSI(52)...')
    fig, regime = plot_annual_rsi(btc, 'BTC', GOLD, rsi_range=[20, 100])
    fig.write_html(OUTPUT_BTC_ANNUAL, include_plotlyjs='cdn', config={'responsive': True})
    print(f'  Saved: {OUTPUT_BTC_ANNUAL}  [{regime}]')

    print('Generating ETH Annual RSI(52)...')
    fig, regime = plot_annual_rsi(eth, 'ETH', MIST, rsi_range=[40, 90])
    fig.write_html(OUTPUT_ETH_ANNUAL, include_plotlyjs='cdn', config={'responsive': True})
    print(f'  Saved: {OUTPUT_ETH_ANNUAL}  [{regime}]')

    print('\nDone.')
