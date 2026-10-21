"""
generate_eth_stacked_tvl_mcap.py
Alphaline Research — ETH Price | Secured Value / Market Cap

Top panel   : ETH price (log)
Bottom panel: value secured on Ethereum (stablecoins + DeFi + RWA) ÷ ETH market cap,
              daily and 30-day average, with the running record

Data sources (no API keys required):
  - DeFiLlama stablecoins : stablecoins.llama.fi/stablecoincharts/Ethereum
  - DeFiLlama chain TVL   : api.llama.fi/v2/historicalChainTvl/Ethereum
  - DeFiLlama protocols   : api.llama.fi/protocols  (RWA aggregation)
  - Yahoo Finance          : ETH-USD daily close via yfinance

Writes docs/eth_stacked_tvl_mcap.html

Usage:
    python generate_eth_stacked_tvl_mcap.py
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from alphaline_theme import apply_theme, write_chart, date_long, ETH_BLUE, WHITE
sys.stdout.reconfigure(encoding='utf-8')

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
ETH_SUPPLY        = 120_000_000
ETH_TVL_THRESHOLD = 1_000_000

OUTPUT_PATH = os.path.join('docs', 'eth_stacked_tvl_mcap.html')

# ════════════════════════════════════════════
# BRAND COLORS
# ════════════════════════════════════════════
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
MIST      = '#7A8F9F'
GREEN_LIT = '#2ABF7A'

SOURCE = 'Yahoo Finance · DeFiLlama'


# ════════════════════════════════════════════
# HTTP HELPER
# ════════════════════════════════════════════
def _get(url, timeout=90, retries=3):
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
    print('Fetching stablecoin TVL...')
    resp = _get('https://stablecoins.llama.fi/stablecoincharts/Ethereum')
    rows = []
    for entry in resp.json():
        ts  = pd.Timestamp(int(entry['date']), unit='s').normalize()
        usd = entry.get('totalCirculatingUSD', {}).get('peggedUSD', 0)
        rows.append({'date': ts, 'stable_tvl_usd': float(usd)})
    df = pd.DataFrame(rows).set_index('date').sort_index().iloc[:-1]
    print(f'  {len(df)} rows | latest: ${df["stable_tvl_usd"].iloc[-1]/1e9:.2f}B')
    return df


def fetch_defi_tvl():
    print('Fetching Ethereum DeFi TVL...')
    r = _get('https://api.llama.fi/v2/historicalChainTvl/Ethereum')
    rows = [{'date': pd.Timestamp(int(e['date']), unit='s').normalize(),
             'defi_tvl_usd': float(e['tvl'])} for e in r.json()]
    df = pd.DataFrame(rows).set_index('date').sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.iloc[:-1]
    print(f'  {len(df)} rows | latest: ${df["defi_tvl_usd"].iloc[-1]/1e9:.2f}B')
    return df


def fetch_rwa_tvl():
    print('Fetching RWA protocol list...')
    r = _get('https://api.llama.fi/protocols')
    protocols = r.json()
    rwa_protocols = [
        (p['name'], p['slug'])
        for p in protocols
        if ('rwa' in p.get('category', '').lower() or
            'real world' in p.get('category', '').lower())
        and p.get('chainTvls', {}).get('Ethereum', 0) > ETH_TVL_THRESHOLD
    ]
    print(f'  Found {len(rwa_protocols)} RWA protocols on Ethereum')
    all_series = {}
    for name, slug in rwa_protocols:
        try:
            r2   = _get(f'https://api.llama.fi/protocol/{slug}')
            data = r2.json()
            if 'chainTvls' in data and 'Ethereum' in data['chainTvls']:
                hist = data['chainTvls']['Ethereum']['tvl']
                all_series[name] = pd.Series(
                    {pd.Timestamp(p['date'], unit='s'): p['totalLiquidityUSD'] for p in hist}
                )
            time.sleep(0.25)
        except Exception as e:
            print(f'  ! {name}: {e}')
    if not all_series:
        return pd.DataFrame(columns=['rwa_tvl_usd'])
    rwa_df = pd.DataFrame(all_series)
    rwa_df.index = pd.to_datetime(rwa_df.index).normalize()
    rwa_daily = rwa_df.resample('D').last().ffill()
    rwa_total = rwa_daily.sum(axis=1).to_frame(name='rwa_tvl_usd').iloc[:-1]
    print(f'  {len(rwa_total)} rows | latest: ${rwa_total["rwa_tvl_usd"].iloc[-1]/1e9:.3f}B')
    return rwa_total


def fetch_eth_price():
    print('Fetching ETH price...')
    raw = yf.download('ETH-USD', period='max', interval='1d', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    eth = raw[['close']].rename(columns={'close': 'eth_price'}).dropna()
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if len(eth) and eth.index[-1] >= today:
        eth = eth.iloc[:-1]
    eth['mcap'] = eth['eth_price'] * ETH_SUPPLY
    print(f'  {len(eth)} rows | latest: ${eth["eth_price"].iloc[-1]:,.0f}')
    return eth


# ════════════════════════════════════════════
# BUILD DATAFRAME
# ════════════════════════════════════════════
def build_dataframe():
    stable = fetch_stablecoin_tvl()
    defi   = fetch_defi_tvl()
    rwa    = fetch_rwa_tvl()
    eth    = fetch_eth_price()

    df = stable.copy()
    df = df.join(defi, how='left')
    df = df.join(rwa,  how='left')
    df = df.join(eth,  how='left')
    df = df.dropna(subset=['eth_price'])

    df['defi_tvl_usd'] = df['defi_tvl_usd'].ffill().fillna(0)
    df['rwa_tvl_usd']  = df['rwa_tvl_usd'].ffill().fillna(0)
    df['total_secured_usd'] = df['stable_tvl_usd'] + df['defi_tvl_usd'] + df['rwa_tvl_usd']
    df['total_to_mcap']     = df['total_secured_usd'] / df['mcap']

    latest = df.iloc[-1]
    print(f'Merged: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}')
    print(f'  ETH: ${latest["eth_price"]:,.0f}  |  Total TVL: ${latest["total_secured_usd"]/1e9:.1f}B'
          f'  |  TVL/Mcap: {latest["total_to_mcap"]:.2f}x')
    return df


# ════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════
def plot_stacked_tvl_mcap(df):
    """ETH price on top; value secured on Ethereum ÷ ETH market cap below.
    The ratio has trended up since 2019 (secured value has grown faster than ETH), and its
    level has not predicted ETH returns (research/eth/ratio_test.py), so it is shown as
    context: the 30-day line, the daily reading behind it, and the running record."""
    d = df.dropna(subset=['eth_price', 'stable_tvl_usd']).copy()
    d = d[d['total_secured_usd'] > 1e9]
    ratio        = d['total_to_mcap']
    ratio_smooth = ratio.rolling(30, min_periods=7).mean()
    record       = ratio_smooth.cummax()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_price'],
        mode='lines', line=dict(color=GOLD, width=1.8),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=ratio,
        mode='lines', line=dict(color='rgba(143,174,230,0.45)', width=1.0),
        name='Secured / market cap (daily)', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Secured / market cap: %{y:.2f}×<extra></extra>'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=ratio_smooth,
        mode='lines', name='30-day average', line=dict(color='#A9C4F5', width=2.6),
        showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>30-day average: %{y:.2f}×<extra></extra>'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=record,
        mode='lines', name='Record (30-day)', line=dict(color=WHITE, width=1.2, dash='dot'),
        opacity=0.75, showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Record so far: %{y:.2f}×<extra></extra>'
    ), row=2, col=1)

    latest = d.iloc[-1]
    rec_val = record.iloc[-1]
    rec_day = ratio_smooth.idxmax()
    subtitle = (
        f'<b>ETH ${latest["eth_price"]:,.0f}</b>  ·  '
        f'${latest["total_secured_usd"]/1e9:.0f}B secured vs. ${latest["mcap"]/1e9:.0f}B ETH market cap  ·  '
        f'<b>{latest["total_to_mcap"]:.2f}×</b> (30-day {ratio_smooth.iloc[-1]:.2f}×, '
        f'record {rec_val:.2f}× in {rec_day.strftime("%b %Y")})  ·  {date_long(d.index[-1])}'
    )

    apply_theme(fig)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='ETH Price', row=1, col=1)
    fig.update_yaxes(title_text='Secured / market cap', ticksuffix='×', rangemode='tozero', row=2, col=1)
    return fig, subtitle


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building ETH Price | Secured Value / Market Cap ===')
    df  = build_dataframe()
    fig, subtitle = plot_stacked_tvl_mcap(df)
    write_chart(fig, OUTPUT_PATH,
                title='ETH Secured Value vs. Market Cap',
                subtitle=subtitle,
                note='Value secured on Ethereum (stablecoins + DeFi + RWA) ÷ ETH market cap. '
                     'Above 1× Ethereum would secure more than it is worth',
                source=SOURCE, default='5Y')
    print(f'Saved: {OUTPUT_PATH}')
