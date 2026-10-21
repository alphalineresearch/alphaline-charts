"""
generate_eth_stacked_tvl.py
Alphaline Research — ETH Price + Stacked TVL (with rolling ATH lines)

Fetches data from DeFiLlama + Yahoo Finance (no API keys required),
builds the combined TVL series, and writes eth_stacked_tvl.html to docs/.

Usage:
    python generate_eth_stacked_tvl.py
"""

import os
import time
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from alphaline_theme import apply_theme, write_chart, date_long

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
ETH_SUPPLY        = 120_000_000
ETH_TVL_THRESHOLD = 1_000_000    # min $1M on Ethereum for RWA protocols

BEACON_CHAIN = '2020-12-01'
MERGE        = '2022-09-15'
SHANGHAI     = '2023-04-12'
EIP4844      = '2024-03-13'
ETH_ETF      = '2024-07-23'

OUTPUT_PATH = os.path.join('docs', 'eth_stacked_tvl.html')
RWA_NOTE = ' · RWA = on-chain RWA protocols only; DefiLlama stopped counting tokenized funds (BUIDL, gold tokens) on Sep 16, 2026'

# ════════════════════════════════════════════
# ALPHALINE BRAND COLORS
# ════════════════════════════════════════════
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
WHITE     = '#F8FAFB'
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
# All DeFiLlama — no API key required
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
        print('  No RWA data fetched — returning zeros.')
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
    print(f'  {len(eth)} rows | latest: ${eth["eth_price"].iloc[-1]:,.0f}')
    return eth


# ════════════════════════════════════════════
# MERGE + COMPUTE COMBINED TVL
# ════════════════════════════════════════════
def build_dataframe():
    stable = fetch_stablecoin_tvl()
    defi   = fetch_defi_tvl()
    rwa    = fetch_rwa_tvl()
    eth    = fetch_eth_price()

    df = stable.copy()
    df = df.join(defi, how='left')
    df = df.join(rwa,  how='left')
    df = df.join(eth[['eth_price']], how='left')
    df = df.dropna(subset=['eth_price'])

    df['defi_tvl_usd'] = df['defi_tvl_usd'].ffill().fillna(0)
    df['rwa_tvl_usd']  = df['rwa_tvl_usd'].ffill().fillna(0)
    df['total_secured_usd'] = df['stable_tvl_usd'] + df['defi_tvl_usd'] + df['rwa_tvl_usd']

    return df


# ════════════════════════════════════════════
# CHART — ETH PRICE + STACKED TVL WITH ROLLING ATH LINES
# ════════════════════════════════════════════
def plot_eth_vs_stacked_tvl_ath(df):
    d = df.dropna(subset=['eth_price', 'stable_tvl_usd']).copy()
    d = d[d['total_secured_usd'] > 1e9]
    d['defi_tvl_usd'] = d['defi_tvl_usd'].fillna(0)
    d['rwa_tvl_usd']  = d['rwa_tvl_usd'].fillna(0)

    # Rolling ATH (cumulative max)
    d['eth_ath']   = d['eth_price'].cummax()
    d['total_ath'] = d['total_secured_usd'].cummax()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.50],
        vertical_spacing=0.04
    )

    # ── Top panel: ETH price (log) ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_price'],
        mode='lines', line=dict(color=GOLD, width=1.8),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_ath'],
        mode='lines', line=dict(color=GOLD_LIT, width=1.2, dash='dot'),
        name='ETH ATH', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH ATH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    latest = d.iloc[-1]

    # ── Bottom panel: stacked TVL areas ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['stable_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(42,191,122,0.32)',
        line=dict(color='#2ABF7A', width=1.2),
        name='Stablecoins', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Stablecoins: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['defi_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(143,174,230,0.32)',
        line=dict(color='#8FAEE6', width=1.2),
        name='DeFi TVL', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>DeFi: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['rwa_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(236,201,106,0.55)',
        line=dict(color=GOLD_LIT, width=1.4),
        name='RWA TVL', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>RWA: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    # Combined TVL rolling ATH dotted line
    fig.add_trace(go.Scatter(
        x=d.index, y=d['total_ath'] / 1e9,
        mode='lines', line=dict(color=WHITE, width=1.2, dash='dot'),
        name='Total ATH', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Total ATH: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    # The latest readings live in the page subtitle (end-of-line labels clipped
    # at the right edge, and the price one sat on a log axis in linear units)
    subtitle = (
        f'<b>ETH ${latest["eth_price"]:,.0f}</b>  ·  '
        f'<b>${latest["total_secured_usd"]/1e9:.0f}B</b> secured: '
        f'stablecoins ${latest["stable_tvl_usd"]/1e9:.0f}B, DeFi ${latest["defi_tvl_usd"]/1e9:.0f}B, '
        f'RWA ${latest["rwa_tvl_usd"]/1e9:.1f}B  ·  {date_long(d.index[-1])}'
    )

    apply_theme(fig)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='', row=1, col=1)
    fig.update_yaxes(tickprefix='$', ticksuffix='B', title_text='', row=2, col=1)
    return fig, subtitle


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building ETH Stacked TVL chart ===')
    df  = build_dataframe()
    fig, subtitle = plot_eth_vs_stacked_tvl_ath(df)
    write_chart(fig, OUTPUT_PATH,
                title='Value Secured on Ethereum',
                subtitle=subtitle,
                note='ETH price and stacked stablecoin, DeFi and RWA value; dotted lines = running all-time highs' + RWA_NOTE,
                source=SOURCE, default='5Y',
                fit_y=('yaxis2',))   # secured value rescales to the window shown
    print(f'Saved: {OUTPUT_PATH}')
