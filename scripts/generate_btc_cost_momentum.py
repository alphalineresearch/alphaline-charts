"""
generate_btc_cost_momentum.py
Alphaline Research — BTC Price + 30-Day Production Cost Momentum

Data sources (no API keys required):
  - mempool.space  : daily hashrate (primary, current)
  - blockchain.info: mining difficulty (fallback / historical)
  - Yahoo Finance  : BTC-USD daily close price

Writes docs/btc_cost_momentum.html

Usage:
    python generate_btc_cost_momentum.py
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_PATH = os.path.join('docs', 'btc_cost_momentum.html')
LOOKBACK    = 30          # days shown in chart

EFFICIENCY_SCHEDULE = {
    '2013-01-01': 1000, '2014-01-01': 500,  '2016-01-01': 200,
    '2018-01-01': 100,  '2019-01-01': 75,   '2020-01-01': 60,
    '2021-01-01': 40,   '2022-06-01': 32,   '2023-01-01': 25,
    '2024-01-01': 22,
}
ELECTRICITY_COST = 0.05
PUE              = 1.10
HALVING_SCHEDULE = [
    ('2009-01-03', 50.0), ('2012-11-28', 25.0),
    ('2016-07-09', 12.5), ('2020-05-11', 6.25),
    ('2024-04-20', 3.125),
]

# ─────────────────────────────────────────────
# BRAND COLORS
# ─────────────────────────────────────────────
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
NAVY_LIT  = '#1A3A6E'
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
GOLD_DIM  = '#8A6B25'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN_LIT = '#2ABF7A'
RED_LIT   = '#D64444'

CHART_WIDTH  = 1100
CHART_HEIGHT = 700


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ─────────────────────────────────────────────
# CHART TEMPLATE
# ─────────────────────────────────────────────
def alphaline_layout(fig, title, height=CHART_HEIGHT,
                     source='alphalineresearch.com  |  Yahoo Finance · mempool.space'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, width=CHART_WIDTH,
        title=dict(
            text=f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>',
            x=0.02, xanchor='left', y=0.98, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=55, r=38, t=60, b=120),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        legend=dict(
            orientation='h', yanchor='top', y=-0.08,
            xanchor='center', x=0.5,
            bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL,
            font=dict(size=9)
        ),
        annotations=[
            dict(text=f'Source: {source}',
                 xref='paper', yref='paper', x=1.0, y=-0.17,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=MIST),
                 showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>',
                 xref='paper', yref='paper', x=0.0, y=-0.17,
                 xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD),
                 showarrow=False),
        ],
    )
    return fig


# ─────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────
def _get(url, **kwargs):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ─────────────────────────────────────────────
# HELPERS: halving / efficiency lookups
# ─────────────────────────────────────────────
def get_block_reward(date):
    reward = 50.0
    for h_date, h_reward in HALVING_SCHEDULE:
        if date >= pd.Timestamp(h_date):
            reward = h_reward
    return reward


def get_efficiency(date):
    eff = 1000.0
    for ts_str, val in sorted(EFFICIENCY_SCHEDULE.items()):
        if date >= pd.Timestamp(ts_str):
            eff = val
    return eff


def compute_cost(df_hr):
    """Given a DataFrame with a 'hashrate_th' column, add production_cost."""
    df_hr = df_hr.copy()
    df_hr['block_reward']    = df_hr.index.map(get_block_reward)
    df_hr['btc_mined_daily'] = df_hr['block_reward'] * 144
    df_hr['efficiency_jth']  = df_hr.index.map(get_efficiency)
    df_hr['production_cost'] = (
        df_hr['hashrate_th'] * df_hr['efficiency_jth'] *
        PUE * 86400 / 3.6e6 * ELECTRICITY_COST / df_hr['btc_mined_daily']
    )
    return df_hr['production_cost']


# ─────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────
def fetch_price():
    """Fetch BTC-USD daily close from Yahoo Finance."""
    print('  Fetching BTC price from Yahoo Finance...')
    raw = yf.download('BTC-USD', period='max', interval='1d', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    return raw['close'].dropna().rename('btc_price')


def fetch_cost_mempool():
    """
    Primary cost source: mempool.space daily hashrate.
    Returns raw (unsmoothed) daily production cost series.
    """
    print('  Fetching hashrate from mempool.space...')
    resp = _get('https://mempool.space/api/v1/mining/hashrate/all')
    rows = resp.json()['hashrates']
    hr = pd.DataFrame(rows)
    hr['date'] = pd.to_datetime(hr['timestamp'], unit='s').dt.normalize()
    hr = hr.set_index('date').sort_index()
    hr = hr[~hr.index.duplicated(keep='last')]
    hr['hashrate_th'] = hr['avgHashrate'] / 1e12
    return compute_cost(hr).dropna().rename('cost_raw')


def fetch_cost_blockchain():
    """
    Fallback cost source: blockchain.com difficulty → hashrate.
    Returns 30-day smoothed production cost series.
    """
    print('  Fetching difficulty from blockchain.info (fallback)...')
    url = 'https://api.blockchain.info/charts/difficulty?timespan=all&format=json&sampled=false'
    resp = _get(url)
    rows = [{'date': pd.Timestamp(int(pt['x']), unit='s').normalize(),
              'value': float(pt['y'])} for pt in resp.json().get('values', [])]
    diff = pd.DataFrame(rows).set_index('date').sort_index()
    diff = diff[~diff.index.duplicated(keep='last')]
    diff.columns = ['avg_difficulty']
    diff.index = pd.to_datetime(diff.index).tz_localize(None)
    diff['hashrate_th'] = diff['avg_difficulty'] * (2 ** 32) / 600 / 1e12
    return compute_cost(diff).rolling(30).mean().dropna().rename('cost_30d')


def build_dataframe():
    """Fetch all required data and return the trimmed 30-day window."""
    btc_px = fetch_price()

    # — production cost (primary: mempool.space raw daily) —
    cost_raw = pd.Series(dtype=float, name='cost_raw')
    cost_30d = pd.Series(dtype=float, name='cost_30d')
    has_cost = False

    try:
        cost_raw = fetch_cost_mempool()
        has_cost = True
        print(f'    mempool.space: {len(cost_raw)} rows | latest ${cost_raw.iloc[-1]:,.0f}')
    except Exception as e:
        print(f'    mempool.space failed: {e}')

    try:
        cost_30d = fetch_cost_blockchain()
        has_cost = True
        print(f'    blockchain.info: {len(cost_30d)} rows | latest ${cost_30d.iloc[-1]:,.0f}')
    except Exception as e:
        print(f'    blockchain.info failed: {e}')

    if not has_cost:
        raise RuntimeError('Both production cost sources failed — cannot build chart.')

    # prefer raw daily (mempool); fall back to 30d-smoothed (blockchain)
    if len(cost_raw):
        cost_col  = 'cost_raw'
        cost_src  = cost_raw
    else:
        cost_col  = 'cost_30d'
        cost_src  = cost_30d

    d = pd.concat([btc_px, cost_30d, cost_src], axis=1)
    if cost_col == 'cost_30d':
        d.columns = ['btc_price', 'cost_30d']
    else:
        d.columns = ['btc_price', 'cost_30d', 'cost_raw']

    d = d.dropna(subset=['btc_price', cost_col])
    d = d.tail(LOOKBACK + 1)   # keep one extra row for pct_change

    d['cost_pct_chg'] = d[cost_col].pct_change() * 100
    d['cost_cum_ret']  = ((1 + d['cost_pct_chg'] / 100).cumprod() - 1) * 100
    d = d.iloc[1:]             # drop the NaN first row after pct_change

    print(f'  Window: {d.index[0].date()} → {d.index[-1].date()} ({len(d)} days)')
    return d


# ─────────────────────────────────────────────
# CHART FUNCTION  (verbatim logic from notebook)
# ─────────────────────────────────────────────
def plot_cost_momentum(d):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.38, 0.33, 0.29],
        vertical_spacing=0.025
    )

    # ── Panel 1: BTC price ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['btc_price'],
        mode='lines', line=dict(color=GOLD, width=2.0),
        name='BTC Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>BTC: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ── Panel 2: daily production cost % change bars (green/red) ──
    bar_colors = [GREEN_LIT if v >= 0 else RED_LIT for v in d['cost_pct_chg']]
    fig.add_trace(go.Bar(
        x=d.index, y=d['cost_pct_chg'],
        name='Daily Cost Δ%', showlegend=True,
        marker_color=bar_colors,
        marker_line=dict(width=0),
        hovertemplate='%{x|%Y-%m-%d}<br>Daily Δ: %{y:+.3f}%<extra></extra>'
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=WHITE, line_width=0.6,
                  opacity=0.25, row=2, col=1)

    # ── Panel 3: cumulative production cost return ──
    cum_final  = d['cost_cum_ret'].iloc[-1]
    cum_color  = GREEN_LIT if cum_final >= 0 else RED_LIT
    fill_color = hex_to_rgba(cum_color, 0.10)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['cost_cum_ret'],
        mode='lines', line=dict(color=cum_color, width=1.8),
        fill='tozeroy', fillcolor=fill_color,
        name='Cumul. Cost Δ%', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Cumul.: %{y:+.2f}%<extra></extra>'
    ), row=3, col=1)
    fig.add_hline(y=0, line_color=WHITE, line_width=0.6,
                  opacity=0.25, row=3, col=1)

    latest_cost = d['cost_30d'].iloc[-1]
    alphaline_layout(
        fig,
        f'BTC PRICE  +  PRODUCTION COST MOMENTUM  |  PAST 30 DAYS'
        f'  —  Cost: ${latest_cost:,.0f}',
        height=CHART_HEIGHT,
        source='alphalineresearch.com  |  Yahoo Finance · mempool.space'
    )

    # ── annotations added AFTER alphaline_layout ──
    latest_btc  = d['btc_price'].iloc[-1]
    btc_30d_ret = (d['btc_price'].iloc[-1] / d['btc_price'].iloc[0] - 1) * 100
    ret_color   = GREEN_LIT if btc_30d_ret >= 0 else RED_LIT

    fig.add_annotation(
        x=d.index[-1], y=latest_btc,
        xref='x', yref='y',
        text=f'  ${latest_btc:,.0f}  ({btc_30d_ret:+.1f}% 30d)',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=ret_color)
    )
    fig.add_annotation(
        x=d.index[-1], y=cum_final,
        xref='x3', yref='y3',
        text=f'  Cost  {cum_final:+.1f}% (30d)',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=cum_color)
    )

    fig.layout.shapes = []

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='left', x=0,
            font=dict(family='Courier New, monospace', size=9, color=MIST),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    fig.update_yaxes(title_text='BTC Price ($)',
                     title_font=dict(size=9, color=MIST), row=1, col=1)
    fig.update_yaxes(title_text='Daily Cost Δ%',
                     title_font=dict(size=9, color=MIST), row=2, col=1)
    fig.update_yaxes(title_text='Cumul. Cost Δ%',
                     title_font=dict(size=9, color=MIST), row=3, col=1)
    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building BTC Cost Momentum Chart ===')
    df = build_dataframe()
    fig = plot_cost_momentum(df)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn')
    print(f'Saved: {OUTPUT_PATH}')
