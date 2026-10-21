"""
generate_eth_tvl_momentum.py
Alphaline Research — ETH Price + Total Value Locked Momentum (Past 30 Days)

Data sources (no API keys required):
  - DeFiLlama stablecoins: stablecoins.llama.fi/stablecoincharts/Ethereum
  - DeFiLlama chain TVL:   api.llama.fi/v2/historicalChainTvl/Ethereum
  - DeFiLlama protocols:   api.llama.fi/protocols  (RWA aggregation)
  - Yahoo Finance:          ETH-USD daily close via yfinance

Writes docs/eth_tvl_momentum.html

Usage:
    python generate_eth_tvl_momentum.py
"""

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf

from alphaline_theme import apply_theme, write_chart, pct, date_long
import eth_rwa

# ════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════
OUTPUT_PATH = os.path.join('docs', 'eth_tvl_momentum.html')

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
ETH_SUPPLY        = 120_000_000
ETH_TVL_THRESHOLD = 1_000_000    # min $1M on Ethereum for RWA protocols
LOOKBACK          = 30           # days

# ════════════════════════════════════════════
# BRAND COLORS
# ════════════════════════════════════════════
GOLD     = '#D4A843'
GOLD_LIT  = '#ECC96A'
GOLD_DIM  = '#8A6B25'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN     = '#1A8A5A'
RED       = '#8A1A1A'
GREEN_LIT = '#2ABF7A'
RED_LIT   = '#D64444'

SOURCE = 'Yahoo Finance · DeFiLlama · Ethereum on-chain · CoinGecko'


# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════
def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


def _get(url, timeout=90, retries=3):
    """requests.get with retry + increasing timeout. DeFiLlama can be slow."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f'  Retry {attempt+1}/{retries-1} after {wait}s ({type(e).__name__})')
                time.sleep(wait)
            else:
                raise


# ════════════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════════════
def fetch_stablecoin_tvl():
    """DeFiLlama: daily stablecoin supply on Ethereum."""
    print('Fetching stablecoin TVL...')
    resp = _get('https://stablecoins.llama.fi/stablecoincharts/Ethereum')
    rows = []
    for entry in resp.json():
        ts  = pd.Timestamp(int(entry['date']), unit='s').normalize()
        usd = entry.get('totalCirculatingUSD', {}).get('peggedUSD', 0)
        rows.append({'date': ts, 'stable_tvl_usd': float(usd)})
    df_s = pd.DataFrame(rows).set_index('date').sort_index()
    df_s = df_s.iloc[:-1]
    print(f'  {len(df_s)} rows | latest: ${df_s["stable_tvl_usd"].iloc[-1]/1e9:.2f}B')
    return df_s


def fetch_defi_tvl():
    """DeFiLlama: daily total DeFi TVL on Ethereum mainnet."""
    print('Fetching Ethereum DeFi TVL...')
    r = _get('https://api.llama.fi/v2/historicalChainTvl/Ethereum')
    rows = [{'date': pd.Timestamp(int(e['date']), unit='s').normalize(),
              'defi_tvl_usd': float(e['tvl'])} for e in r.json()]
    df_d = pd.DataFrame(rows).set_index('date').sort_index()
    df_d.index = pd.to_datetime(df_d.index).tz_localize(None)
    df_d = df_d.iloc[:-1]
    print(f'  {len(df_d)} rows | latest: ${df_d["defi_tvl_usd"].iloc[-1]/1e9:.2f}B')
    return df_d


def fetch_rwa_tvl():
    """RWA on Ethereum: DefiLlama RWA protocols plus tokenized funds and gold read from
    their Ethereum contracts (DefiLlama stopped counting those on Sep 16, 2026).
    Shared with the other ETH charts — see scripts/eth_rwa.py."""
    return eth_rwa.fetch_rwa_tvl(_get)


def fetch_eth_price():
    """Yahoo Finance: ETH-USD daily close."""
    print('Fetching ETH price...')
    raw = yf.download('ETH-USD', period='max', interval='1d', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    eth = raw[['close']].rename(columns={'close': 'eth_price'}).dropna()
    print(f'  {len(eth)} rows | latest: ${eth["eth_price"].iloc[-1]:,.0f}')
    return eth


# ════════════════════════════════════════════
# BUILD DATAFRAME
# ════════════════════════════════════════════
def build_dataframe():
    stable_daily = fetch_stablecoin_tvl()
    defi_daily   = fetch_defi_tvl()
    rwa_daily    = fetch_rwa_tvl()
    eth          = fetch_eth_price()

    df = stable_daily.copy()
    df = df.join(defi_daily, how='left')
    df = df.join(rwa_daily,  how='left')
    df = df.join(eth[['eth_price']], how='left')
    df = df.dropna(subset=['eth_price'])

    df['defi_tvl_usd'] = df['defi_tvl_usd'].ffill().fillna(0)
    df['rwa_tvl_usd']  = df['rwa_tvl_usd'].ffill().fillna(0)

    df['total_secured_usd'] = df['stable_tvl_usd'] + df['defi_tvl_usd'] + df['rwa_tvl_usd']

    latest = df.iloc[-1]
    print(f'\nMerged: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}')
    print(f'  ETH Price:      ${latest["eth_price"]:,.0f}')
    print(f'  Total TVL:      ${latest["total_secured_usd"]/1e9:.2f}B')
    return df


# ════════════════════════════════════════════
# CHART — verbatim from notebook
# ════════════════════════════════════════════
def plot_tvl_momentum(df, lookback=30):
    d = df.dropna(subset=['eth_price', 'total_secured_usd']).copy()
    d['defi_tvl_usd'] = d['defi_tvl_usd'].fillna(0)
    d['rwa_tvl_usd']  = d['rwa_tvl_usd'].fillna(0)
    d = d.tail(lookback + 1)

    d['total_pct_chg'] = d['total_secured_usd'].pct_change() * 100
    d = d.iloc[1:]
    base_tvl = d['total_secured_usd'].iloc[0]
    d['tvl_cum_ret'] = (d['total_secured_usd'] / base_tvl - 1) * 100

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.38, 0.33, 0.29],
        vertical_spacing=0.025
    )

    # ── Panel 1: ETH price ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_price'],
        mode='lines', line=dict(color=GOLD, width=2.0),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ── Panel 2: daily TVL % change bars (green/red) ──
    bar_colors = [GREEN_LIT if v >= 0 else RED_LIT for v in d['total_pct_chg']]
    fig.add_trace(go.Bar(
        x=d.index, y=d['total_pct_chg'],
        name='Daily TVL \u0394%', showlegend=True,
        marker_color=bar_colors,
        marker_line=dict(width=0),
        hovertemplate='%{x|%Y-%m-%d}<br>Daily \u0394: %{y:+.3f}%<extra></extra>'
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=WHITE, line_width=0.6,
                  opacity=0.25, row=2, col=1)

    # ── Panel 3: cumulative TVL return ──
    cum_final  = d['tvl_cum_ret'].iloc[-1]
    cum_color  = GREEN_LIT if cum_final >= 0 else RED_LIT
    fill_color = hex_to_rgba(cum_color, 0.10)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['tvl_cum_ret'],
        mode='lines', line=dict(color=cum_color, width=1.8),
        fill='tozeroy', fillcolor=fill_color,
        name='Cumul. TVL Ret.', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Cumul.: %{y:+.2f}%<extra></extra>'
    ), row=3, col=1)
    fig.add_hline(y=0, line_color=WHITE, line_width=0.6,
                  opacity=0.25, row=3, col=1)

    # Latest readings go in the page subtitle (end-of-line labels clipped)
    latest_tvl  = d['total_secured_usd'].iloc[-1]
    latest_eth  = d['eth_price'].iloc[-1]
    eth_30d_ret = (d['eth_price'].iloc[-1] / d['eth_price'].iloc[0] - 1) * 100
    subtitle = (f'<b>ETH ${latest_eth:,.0f}</b>  ·  ${latest_tvl/1e9:.1f}B secured  ·  past 30 days: '
                f'ETH {pct(eth_30d_ret)}, secured value {pct(cum_final)}  ·  {date_long(d.index[-1])}')

    apply_theme(fig)
    fig.update_yaxes(title_text='ETH Price ($)', tickformat='$,.0f', row=1, col=1)
    fig.update_yaxes(title_text='Daily TVL\u0394%',
                     ticksuffix='%', row=2, col=1)
    fig.update_yaxes(title_text='Cumulative', ticksuffix='%', row=3, col=1)
    fig.update_xaxes(range=[d.index[0] - pd.Timedelta(days=0.6), d.index[-1] + pd.Timedelta(days=0.6)])
    return fig, subtitle


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building ETH Price + TVL Momentum ===')
    df  = build_dataframe()
    fig, subtitle = plot_tvl_momentum(df, lookback=LOOKBACK)
    write_chart(fig, OUTPUT_PATH,
                title='ETH Price & Secured Value Momentum',
                subtitle=subtitle,
                note='Past 30 days: daily % change in total value secured on Ethereum, and its cumulative return',
                source=SOURCE, ranges=False)
    print(f'Saved: {OUTPUT_PATH}')
