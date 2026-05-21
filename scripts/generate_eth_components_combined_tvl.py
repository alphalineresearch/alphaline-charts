"""
generate_eth_combined_tvl.py
Alphaline Research — ETH Total Value Locked: Daily Component Flows (Past 30 Days)

Data sources (no API keys required):
  - DeFiLlama stablecoins : stablecoins.llama.fi/stablecoincharts/Ethereum
  - DeFiLlama chain TVL   : api.llama.fi/v2/historicalChainTvl/Ethereum
  - DeFiLlama protocols   : api.llama.fi/protocols  (RWA protocol list + per-slug TVL)
  - Yahoo Finance         : ETH-USD daily close via yfinance

Writes docs/eth_combined_components_tvl.html

Usage:
    python generate_eth_combined_tvl.py
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
OUTPUT_PATH       = os.path.join('docs', 'eth_combined_components_tvl.html')
LOOKBACK          = 30
ETH_SUPPLY        = 120_000_000
ETH_TVL_THRESHOLD = 1_000_000    # min $1M Ethereum TVL for RWA protocols
LIDO_POOL_ID      = '747c1d2a-c668-4682-b9f9-296708a3dd90'

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
GREEN     = '#1A8A5A'
RED       = '#8A1A1A'
GREEN_LIT = '#2ABF7A'
RED_LIT   = '#D64444'

CHART_WIDTH  = 1100
CHART_HEIGHT = 750


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ─────────────────────────────────────────────
# CHART TEMPLATE
# ─────────────────────────────────────────────
def alphaline_layout(fig, title, height=CHART_HEIGHT, subtitle='',
                     source='alphalineresearch.com  |  Yahoo Finance  |  DeFiLlama'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, autosize=True,
        showlegend=False,
        title=dict(
            text=(
                f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>' +
                '<br><span style="font-family:\'Courier New\',monospace; font-size:8px; color:' + GOLD + ';">ALPHALINE RESEARCH</span>' +
                (f'<br><span style="font-family:\'Courier New\',monospace; font-size:9px; color:' + MIST + ';">{subtitle}</span>' if subtitle else '')
            ),
            x=0.02, xanchor='left', y=0.985, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=60, r=80, t=80, b=80),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        xaxis2=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        yaxis2=dict(gridcolor='rgba(212,168,67,0.03)', gridwidth=0.4, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        annotations=[
            dict(text=f'Source: {source}',
                 xref='paper', yref='paper', x=1.0, y=-0.09,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=9, color=MIST),
                 showarrow=False),                 showarrow=False),
        ],
    )
    return fig


# ─────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────
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
    """DeFiLlama: aggregate RWA TVL on Ethereum across all tracked protocols."""
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
        print('  No RWA data fetched — returning zeros.')
        return pd.DataFrame(columns=['rwa_tvl_usd'])

    rwa_df = pd.DataFrame(all_series)
    rwa_df.index = pd.to_datetime(rwa_df.index).normalize()
    rwa_daily = rwa_df.resample('D').last().ffill()
    rwa_total = rwa_daily.sum(axis=1).to_frame(name='rwa_tvl_usd')
    rwa_total = rwa_total.iloc[:-1]
    print(f'  {len(rwa_total)} rows | latest: ${rwa_total["rwa_tvl_usd"].iloc[-1]/1e9:.3f}B')
    return rwa_total


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


# ─────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────
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

    # Forward-fill TVL series (not always daily)
    df['defi_tvl_usd'] = df['defi_tvl_usd'].ffill().fillna(0)
    df['rwa_tvl_usd']  = df['rwa_tvl_usd'].ffill().fillna(0)

    # Combined TVL
    df['total_secured_usd'] = df['stable_tvl_usd'] + df['defi_tvl_usd'] + df['rwa_tvl_usd']

    print(f'Merged: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}')
    latest = df.iloc[-1]
    print(f'  Stablecoins: ${latest["stable_tvl_usd"]/1e9:.2f}B')
    print(f'  DeFi TVL:    ${latest["defi_tvl_usd"]/1e9:.2f}B')
    print(f'  RWA TVL:     ${latest["rwa_tvl_usd"]/1e9:.3f}B')
    print(f'  Total:       ${latest["total_secured_usd"]/1e9:.2f}B')
    return df


# ─────────────────────────────────────────────
# CHART FUNCTION  (verbatim from notebook)
# ─────────────────────────────────────────────
def plot_tvl_daily_flows(df, lookback=30):
    d = df.dropna(subset=['eth_price', 'total_secured_usd']).copy()
    d['defi_tvl_usd'] = d['defi_tvl_usd'].fillna(0)
    d['rwa_tvl_usd']  = d['rwa_tvl_usd'].fillna(0)
    d = d.tail(lookback + 1)

    total_prev = d['total_secured_usd'].shift(1)
    d['stable_contrib'] = d['stable_tvl_usd'].diff() / total_prev * 100
    d['defi_contrib']   = d['defi_tvl_usd'].diff()   / total_prev * 100
    d['rwa_contrib']    = d['rwa_tvl_usd'].diff()     / total_prev * 100
    d['total_pct_chg']  = d['total_secured_usd'].pct_change() * 100
    d = d.iloc[1:]

    # Cumulative returns for panel 3
    d['eth_cum_ret'] = (d['eth_price'] / d['eth_price'].iloc[0] - 1) * 100
    d['tvl_cum_ret'] = (d['total_secured_usd'] / d['total_secured_usd'].iloc[0] - 1) * 100

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.28, 0.44, 0.28],
        vertical_spacing=0.03
    )

    # ── Panel 1: ETH price ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_price'],
        mode='lines', line=dict(color=GOLD, width=2.0),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ── Panel 2: stacked component contributions ──
    components = [
        ('stable_contrib', 'Stablecoins', GOLD),
        ('defi_contrib',   'DeFi TVL',    MIST),
        ('rwa_contrib',    'RWA TVL',     GREEN_LIT),
    ]
    for col, name, color in components:
        fig.add_trace(go.Bar(
            x=d.index, y=d[col],
            name=name, showlegend=True,
            marker_color=hex_to_rgba(color, 0.78),
            marker_line=dict(width=0.4, color=color),
            hovertemplate='%{x|%Y-%m-%d}<br>' + name + ': %{y:+.3f}%<extra></extra>'
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['total_pct_chg'],
        mode='lines+markers',
        line=dict(color=WHITE, width=1.0, dash='dot'),
        marker=dict(size=3, color=WHITE),
        name='Total TVL Δ%', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Total: %{y:+.3f}%<extra></extra>'
    ), row=2, col=1)

    fig.add_hline(y=0, line_color=WHITE, line_width=0.6,
                  opacity=0.25, row=2, col=1)

    # ── Panel 3: ETH vs TVL cumulative return ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_cum_ret'],
        mode='lines', line=dict(color=GOLD, width=1.8),
        name='ETH 30d Cum. Return', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: %{y:+.2f}%<extra></extra>'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['tvl_cum_ret'],
        mode='lines', line=dict(color=GREEN_LIT, width=1.8, dash='dot'),
        name='TVL 30d Cum. Return', showlegend=True,
        fill='tonexty',
        fillcolor=hex_to_rgba(GREEN_LIT, 0.10),
        hovertemplate='%{x|%Y-%m-%d}<br>TVL: %{y:+.2f}%<extra></extra>'
    ), row=3, col=1)

    fig.add_hline(y=0, line_color=WHITE, line_width=0.6,
                  opacity=0.25, row=3, col=1)

    alphaline_layout(
        fig,
        'ETH TOTAL VALUE LOCKED — DAILY COMPONENT FLOWS  |  PAST 30 DAYS',
        height=900
    )

    # ── Panel 1 annotation: ETH price + 30d return ──
    latest_eth  = d['eth_price'].iloc[-1]
    eth_30d_ret = (d['eth_price'].iloc[-1] / d['eth_price'].iloc[0] - 1) * 100
    p1_color    = GREEN_LIT if eth_30d_ret >= 0 else RED_LIT
    fig.add_annotation(
        x=d.index[-1], y=latest_eth,
        xref='x', yref='y',
        text=f'  ${latest_eth:,.0f}  ({eth_30d_ret:+.1f}% 30d)',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=p1_color)
    )

    # ── Panel 3 annotations: ETH cumul. and TVL cumul. ──
    eth_cum_val = d['eth_cum_ret'].iloc[-1]
    tvl_cum_val = d['tvl_cum_ret'].iloc[-1]
    eth_cum_col = GREEN_LIT if eth_cum_val >= 0 else RED_LIT
    tvl_cum_col = GREEN_LIT if tvl_cum_val >= 0 else RED_LIT

    fig.add_annotation(
        x=d.index[-1], y=eth_cum_val,
        xref='x3', yref='y3',
        text=f'  ETH cumul. {eth_cum_val:+.1f}%',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=eth_cum_col)
    )
    fig.add_annotation(
        x=d.index[-1], y=tvl_cum_val,
        xref='x3', yref='y3',
        text=f'  TVL cumul. {tvl_cum_val:+.1f}%',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=tvl_cum_col)
    )

    fig.update_layout(
        barmode='relative',
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='center', x=0.5,
            font=dict(family='Courier New, monospace', size=9, color=MIST),
            bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL, borderwidth=1
        )
    )
    fig.update_yaxes(title_text='ETH Price ($)',
                     title_font=dict(size=9, color=MIST), row=1, col=1)
    fig.update_yaxes(title_text='Daily Contribution (% of Total TVL)',
                     title_font=dict(size=9, color=MIST), row=2, col=1)
    fig.update_yaxes(title_text='Cumulative Return (%)',
                     title_font=dict(size=9, color=MIST), row=3, col=1)
    # Pad x-axis 3 days right so end-of-data annotations aren't clipped
    fig.update_xaxes(range=[d.index[0], d.index[-1] + pd.Timedelta(days=3)])
    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building ETH Combined TVL Daily Flows ===')
    df = build_dataframe()
    fig = plot_tvl_daily_flows(df, lookback=LOOKBACK)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
