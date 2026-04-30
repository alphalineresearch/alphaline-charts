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

# ════════════════════════════════════════════
# ALPHALINE BRAND COLORS
# ════════════════════════════════════════════
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN_LIT = '#2ABF7A'

CHART_WIDTH  = 1100
CHART_HEIGHT = 750


# ════════════════════════════════════════════
# ALPHALINE CHART TEMPLATE
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT,
                     source='alphalineresearch.com  |  Yahoo Finance  |  DeFiLlama'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, width=CHART_WIDTH,
        showlegend=False,
        title=dict(
            text=f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>',
            x=0.02, xanchor='left', y=0.98, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=60, r=80, t=70, b=110),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        xaxis2=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        yaxis2=dict(gridcolor='rgba(212,168,67,0.03)', gridwidth=0.4, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        annotations=[
            dict(text=f'Source: {source}', xref='paper', yref='paper',
                 x=1.0, y=-0.16, xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=MIST), showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>', xref='paper', yref='paper',
                 x=0.07, y=-0.16, xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD), showarrow=False),
        ],
        shapes=[
            dict(type='line', xref='paper', yref='paper', x0=0.010, y0=-0.21, x1=0.018, y1=-0.175,
                 line=dict(color=WHITE, width=2.0), layer='above'),
            dict(type='line', xref='paper', yref='paper', x0=0.018, y0=-0.175, x1=0.026, y1=-0.21,
                 line=dict(color=WHITE, width=2.0), layer='above'),
            dict(type='line', xref='paper', yref='paper', x0=0.012, y0=-0.194, x1=0.024, y1=-0.194,
                 line=dict(color=WHITE, width=1.4), layer='above'),
            dict(type='rect', xref='paper', yref='paper', x0=0.016, y0=-0.174, x1=0.020, y1=-0.166,
                 fillcolor=GOLD, line_width=0, layer='above'),
        ]
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
        name='ETH Rolling ATH', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH ATH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    latest = d.iloc[-1]
    fig.add_annotation(
        x=d.index[-1], y=latest['eth_price'],
        text=f'  ${latest["eth_price"]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD)
    )
    if latest['eth_price'] >= latest['eth_ath'] * 0.999:
        fig.add_annotation(
            x=d.index[-1], y=latest['eth_ath'],
            text='  ▲ ATH', showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=9, color=GOLD_LIT)
        )

    # ── Bottom panel: stacked TVL areas ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['stable_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(212,168,67,0.22)',
        line=dict(color=GOLD, width=1.0),
        name='Stablecoins', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Stablecoins: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['defi_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(122,143,159,0.22)',
        line=dict(color=MIST, width=1.0),
        name='DeFi TVL', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>DeFi: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['rwa_tvl_usd'] / 1e9,
        mode='lines', stackgroup='tvl',
        fillcolor='rgba(42,191,122,0.18)',
        line=dict(color=GREEN_LIT, width=1.4),
        name='RWA TVL', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>RWA: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    # Combined TVL rolling ATH dotted line
    fig.add_trace(go.Scatter(
        x=d.index, y=d['total_ath'] / 1e9,
        mode='lines', line=dict(color=WHITE, width=1.2, dash='dot'),
        name='Total Secured ATH', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Total ATH: $%{y:.1f}B<extra></extra>'
    ), row=2, col=1)

    # Annotations on bottom panel
    for val, color, label in [
        (latest['stable_tvl_usd'] / 2 / 1e9,                              GOLD,      f'Stablecoins  ${latest["stable_tvl_usd"]/1e9:.0f}B'),
        ((latest['stable_tvl_usd'] + latest['defi_tvl_usd'] / 2) / 1e9,  MIST,      f'DeFi  ${latest["defi_tvl_usd"]/1e9:.0f}B'),
        (latest['total_secured_usd'] / 1e9,                                GREEN_LIT, f'Total  ${latest["total_secured_usd"]/1e9:.0f}B'),
    ]:
        fig.add_annotation(
            x=d.index[-1], y=val, text=f'  {label}',
            showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=9, color=color)
        )

    # Event lines
    events = [
        (BEACON_CHAIN, 'Beacon',   STEEL),
        (MERGE,        'Merge',    MIST),
        (SHANGHAI,     'Shanghai', GOLD),
        (EIP4844,      '4844',     MIST),
        (ETH_ETF,      'ETFs',     GOLD_LIT),
    ]
    for date, label, evt_color in events:
        for row in [1, 2]:
            fig.add_vline(x=date, line_color=evt_color, line_width=1,
                          line_dash='dot', row=row, col=1)
        fig.add_annotation(x=date, y=1.01, xref='x', yref='paper',
            text=label, showarrow=False,
            font=dict(size=8, color=evt_color), xanchor='left')

    title = (
        f'ETH Price + Stacked TVL — Rolling All-Time Highs  |  '
        f'ETH: ${latest["eth_price"]:,.0f}  |  '
        f'Total Secured: ${latest["total_secured_usd"]/1e9:.0f}B'
    )
    alphaline_layout(fig, title, height=CHART_HEIGHT)

    # Shift footer annotations up to fit tighter bottom margin
    new_anns = []
    for ann in fig.layout.annotations:
        a = ann.to_plotly_json()
        if a.get('yref') == 'paper' and isinstance(a.get('y'), (int, float)) and a['y'] < -0.05:
            a['y'] = -0.09
        new_anns.append(a)

    fig.update_layout(
        showlegend=True,
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL, borderwidth=1,
                    font=dict(size=9, color=MIST), x=0.02, y=0.98),
        margin=dict(l=60, r=80, t=70, b=100),
        shapes=[],
        annotations=new_anns
    )
    fig.update_yaxes(type='log', tickprefix='$', title_text='ETH Price (log)', row=1, col=1)
    fig.update_yaxes(tickprefix='$', ticksuffix='B', title_text='TVL ($B)', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building ETH Stacked TVL chart ===')
    df  = build_dataframe()
    fig = plot_eth_vs_stacked_tvl_ath(df)

    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn')
    print(f'Saved: {OUTPUT_PATH}')
