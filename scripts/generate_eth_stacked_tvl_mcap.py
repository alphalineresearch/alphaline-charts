"""
generate_eth_stacked_tvl_mcap.py
Alphaline Research — ETH Stacked TVL + Price (right axis) | TVL / Market Cap

Top panel   : Stablecoins / DeFi / RWA stacked area (left axis)
              ETH Price (right axis, log scale)
Bottom panel: Total Secured TVL / ETH Market Cap ratio with 30d MA

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
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN_LIT = '#2ABF7A'

CHART_HEIGHT = 800


# ════════════════════════════════════════════
# CHART TEMPLATE
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT, subtitle='',
                     source='alphalineresearch.com  |  Yahoo Finance  |  DeFiLlama'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, autosize=True,
        showlegend=False,
        title=dict(
            text=(
                f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>'
                + '<br><span style="font-family:\'Courier New\',monospace; font-size:8px; color:' + GOLD + ';">ALPHALINE RESEARCH</span>'
                + (f'<br><span style="font-family:\'Courier New\',monospace; font-size:9px; color:' + MIST + f';">{subtitle}</span>' if subtitle else '')
            ),
            x=0.02, xanchor='left', y=0.985, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=60, r=80, t=80, b=95),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        xaxis2=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        annotations=[
            dict(text=f'Source: {source}', xref='paper', yref='paper',
                 x=0.0, y=-0.04, xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=8, color=STEEL), showarrow=False),
        ],
    )
    return fig


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
    d = df.dropna(subset=['eth_price', 'stable_tvl_usd']).copy()
    d = d[d['total_secured_usd'] > 1e9]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        specs=[[{'secondary_y': True}], [{'secondary_y': False}]],
        row_heights=[0.60, 0.40],
        vertical_spacing=0.04
    )

    # ── Row 1 left axis: stacked TVL areas ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['stable_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(212,168,67,0.22)',
        line=dict(color=GOLD, width=1.0),
        name='Stablecoins', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Stablecoins: $%{y:.1f}B<extra></extra>'
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['defi_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(122,143,159,0.22)',
        line=dict(color=MIST, width=1.0),
        name='DeFi TVL', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>DeFi: $%{y:.1f}B<extra></extra>'
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['rwa_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(42,191,122,0.18)',
        line=dict(color=GREEN_LIT, width=1.4),
        name='RWA TVL', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>RWA: $%{y:.1f}B<extra></extra>'
    ), row=1, col=1, secondary_y=False)

    # ── Row 1 right axis: ETH price (log) ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_price'],
        mode='lines', line=dict(color=GOLD_LIT, width=1.8),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1, secondary_y=True)

    # ── Row 2: TVL / Mcap ratio ──
    ratio        = d['total_to_mcap']
    ratio_smooth = ratio.rolling(30, min_periods=7).mean()

    fig.add_trace(go.Scatter(
        x=d.index, y=ratio,
        mode='lines', line=dict(color='rgba(212,168,67,0.25)', width=0.8),
        fill='tozeroy', fillcolor='rgba(212,168,67,0.07)',
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>Secured/Mcap: %{y:.2f}x<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=ratio_smooth,
        mode='lines', name='TVL/Mcap 30d MA', line=dict(color=GOLD, width=1.5),
        showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>30d MA: %{y:.2f}x<extra></extra>'
    ), row=2, col=1)

    # ── Annotations ──
    latest = d.iloc[-1]
    # ETH price label on right axis
    fig.add_annotation(
        x=d.index[-1], y=latest['eth_price'],
        xref='x', yref='y2',
        text=f'  ${latest["eth_price"]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD_LIT)
    )
    # Total TVL label on left axis
    fig.add_annotation(
        x=d.index[-1], y=latest['total_secured_usd'] / 1e9,
        xref='x', yref='y',
        text=f'  ${latest["total_secured_usd"]/1e9:.0f}B',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=9, color=GREEN_LIT)
    )
    # Ratio label on bottom panel
    fig.add_annotation(
        x=d.index[-1], y=ratio.iloc[-1],
        xref='x2', yref='y3',
        text=f'  {ratio.iloc[-1]:.2f}x',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=9, color=GOLD)
    )

    subtitle = (
        f'ETH: ${latest["eth_price"]:,.0f}  '
        f'Total TVL: ${latest["total_secured_usd"]/1e9:.0f}B  '
        f'TVL/Mcap: {latest["total_to_mcap"]:.2f}x'
    )
    alphaline_layout(fig, 'ETH Stacked TVL + Price  |  TVL / Market Cap', subtitle=subtitle)

    fig.update_layout(
        showlegend=True,
        legend=dict(
            bgcolor='rgba(10,22,40,0.0)', bordercolor='rgba(0,0,0,0)', borderwidth=0,
            font=dict(size=9, color=MIST),
            orientation='h', x=0.5, xanchor='center',
            y=-0.11, yanchor='top',
            tracegroupgap=0,
        ),
        shapes=[],
    )
    # y-axes
    fig.update_yaxes(tickprefix='$', ticksuffix='B', title_text='TVL ($B)',
                     range=[0, 350],
                     row=1, col=1, secondary_y=False)
    fig.update_yaxes(
        type='log',
        tickmode='array',
        tickvals=[100, 300, 1_000, 3_000, 10_000, 30_000],
        ticktext=['$0.1K', '$0.3K', '$1K', '$3K', '$10K', '$30K'],
        title_text='ETH Price',
        row=1, col=1, secondary_y=True, showgrid=False
    )
    fig.update_yaxes(title_text='TVL / Market Cap', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building ETH Stacked TVL + Price + TVL/Mcap ===')
    df  = build_dataframe()
    fig = plot_stacked_tvl_mcap(df)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
