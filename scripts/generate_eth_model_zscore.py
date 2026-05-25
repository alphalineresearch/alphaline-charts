"""
generate_eth_model_zscore.py
Alphaline Research — ETH Combined TVL Fair Value Model + Z-score

Top panel   : ETH price (log), model fair value, +/-1σ and +/-2σ bands
Bottom panel: ETH price / model Z-score (under/over valuation)

Data sources (no API keys required):
  - DeFiLlama stablecoins : stablecoins.llama.fi/stablecoincharts/Ethereum
  - DeFiLlama chain TVL   : api.llama.fi/v2/historicalChainTvl/Ethereum
  - DeFiLlama protocols   : api.llama.fi/protocols  (RWA aggregation)
  - Yahoo Finance          : ETH-USD daily close via yfinance

Writes docs/eth_model_zscore.html

Usage:
    python generate_eth_model_zscore.py
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import yfinance as yf
sys.stdout.reconfigure(encoding='utf-8')

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
ETH_SUPPLY        = 120_000_000
ETH_TVL_THRESHOLD = 1_000_000

OUTPUT_PATH = os.path.join('docs', 'eth_model_zscore.html')

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
RED_LIT   = '#D64444'
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
        margin=dict(l=40, r=55, t=80, b=110),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        xaxis2=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        yaxis2=dict(gridcolor='rgba(212,168,67,0.03)', gridwidth=0.4, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        annotations=[
            dict(text=f'Source: {source}', xref='paper', yref='paper',
                 x=0.0, y=-0.09, xanchor='left', yanchor='top',
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
    print(f'  Found {len(rwa_protocols)} RWA protocols')
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
    print(f'  latest: ${rwa_total["rwa_tvl_usd"].iloc[-1]/1e9:.3f}B')
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
# BUILD DATAFRAME + FIT MODEL
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

    print(f'Merged: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}')
    print(f'  ETH: ${df["eth_price"].iloc[-1]:,.0f}  |  Total TVL: ${df["total_secured_usd"].iloc[-1]/1e9:.1f}B')
    return df


def fit_model(df, x_col, min_val=1e9):
    reg = df.dropna(subset=[x_col, 'eth_price'])
    reg = reg[reg[x_col] > min_val]
    lx  = np.log(reg[x_col].values).reshape(-1, 1)
    ly  = np.log(reg['eth_price'].values)
    m   = LinearRegression().fit(lx, ly)
    r2  = m.score(lx, ly)
    resid_std = (ly - m.predict(lx)).std()
    return m, r2, resid_std


def apply_model(df, model, std, x_col, min_val=1e9):
    mask = df[x_col] > min_val
    df['eth_model'] = np.where(
        mask,
        np.exp(model.predict(np.log(df[x_col].clip(lower=min_val)).values.reshape(-1, 1))),
        np.nan
    )
    df['band_1up'] = df['eth_model'] * np.exp( 1 * std)
    df['band_1dn'] = df['eth_model'] * np.exp(-1 * std)
    df['band_2up'] = df['eth_model'] * np.exp( 2 * std)
    df['band_2dn'] = df['eth_model'] * np.exp(-2 * std)
    df['zscore']   = np.log(df['eth_price'] / df['eth_model']) / std
    return df


# ════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════
def plot_model_zscore(df, r2):
    d = df.dropna(subset=['eth_model']).copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.03
    )

    # ── Top panel: +/-2σ fill ──
    fig.add_trace(go.Scatter(
        x=pd.concat([d.index.to_series(), d.index.to_series()[::-1]]),
        y=pd.concat([d['band_2up'], d['band_2dn'][::-1]]),
        fill='toself', fillcolor='rgba(212,168,67,0.06)',
        line=dict(width=0), showlegend=True, name='+/-2σ Band', hoverinfo='skip'
    ), row=1, col=1)

    # +/-1σ fill
    fig.add_trace(go.Scatter(
        x=pd.concat([d.index.to_series(), d.index.to_series()[::-1]]),
        y=pd.concat([d['band_1up'], d['band_1dn'][::-1]]),
        fill='toself', fillcolor='rgba(212,168,67,0.13)',
        line=dict(width=0), showlegend=True, name='+/-1σ Band', hoverinfo='skip'
    ), row=1, col=1)

    # Model fair value line
    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_model'],
        mode='lines', line=dict(color='rgba(248,250,251,0.55)', width=1.4, dash='dash'),
        name='Model Fair Value', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Model: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ETH price
    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_price'],
        mode='lines', line=dict(color=GOLD, width=2.0),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ── Top panel annotations ──
    latest = d.iloc[-1]
    fig.add_annotation(
        x=d.index[-1], y=latest['eth_price'],
        xref='x', yref='y',
        text=f'  ${latest["eth_price"]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD)
    )
    fig.add_annotation(
        x=d.index[-1], y=latest['eth_model'],
        xref='x', yref='y',
        text=f'  Model ${latest["eth_model"]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=9, color=WHITE)
    )

    # ── Bottom panel: Z-score bars ──
    zscore  = d['zscore']
    zcolors = [
        RED_LIT                  if v >  2 else
        'rgba(214,68,68,0.55)'   if v >  1 else
        'rgba(214,68,68,0.30)'   if v >  0 else
        'rgba(42,191,122,0.30)'  if v > -1 else
        'rgba(42,191,122,0.55)'  if v > -2 else
        GREEN_LIT
        for v in zscore
    ]
    fig.add_trace(go.Bar(
        x=d.index, y=zscore,
        marker_color=zcolors, marker_opacity=1.0,
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>Z-score: %{y:+.2f}σ<extra></extra>'
    ), row=2, col=1)

    # 28d MA overlay
    zscore_ma = zscore.rolling(28, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=zscore_ma.index, y=zscore_ma, mode='lines',
        line=dict(color='rgba(248,250,251,0.40)', width=1.8),
        name='Z 28d MA', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>28d MA: %{y:+.2f}σ<extra></extra>'
    ), row=2, col=1)

    # Reference lines and labels
    for level, color in [(2, RED_LIT), (1, 'rgba(214,68,68,0.6)'),
                         (-1, 'rgba(42,191,122,0.6)'), (-2, GREEN_LIT)]:
        fig.add_hline(y=level, line_color=color, line_width=0.7, line_dash='dot', row=2, col=1)
        fig.add_annotation(
            x=d.index[-1], y=level,
            xref='x2', yref='y2',
            text=f'  {level:+d}σ', showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=8, color=color)
        )
    fig.add_hline(y=0, line_color=STEEL, line_width=0.8, row=2, col=1)

    # Current Z-score label
    fig.add_annotation(
        x=d.index[-1], y=zscore.iloc[-1],
        xref='x2', yref='y2',
        text=f'  {zscore.iloc[-1]:+.2f}σ', showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD_LIT)
    )

    subtitle = (
        f'ETH: ${latest["eth_price"]:,.0f}  '
        f'Model: ${latest["eth_model"]:,.0f}  '
        f'R²={r2:.3f}'
    )
    alphaline_layout(fig, 'ETH Combined TVL Model  |  Z-score', subtitle=subtitle)

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
    fig.update_yaxes(
        type='log',
        tickmode='array',
        tickvals=[100, 300, 1_000, 3_000, 10_000, 30_000],
        ticktext=['$0.1K', '$0.3K', '$1K', '$3K', '$10K', '$30K'],
        title_text='ETH Price',
        row=1, col=1
    )
    fig.update_yaxes(title_text='Z-score (σ)', row=2, col=1)
    fig.update_xaxes(title_text='', row=2, col=1)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building ETH Combined TVL Model + Z-score ===')
    df = build_dataframe()

    model, r2, std = fit_model(df, 'total_secured_usd', min_val=1e9)
    df = apply_model(df, model, std, 'total_secured_usd', min_val=1e9)

    print(f'Model R²: {r2:.3f} | Z-score: {df["zscore"].iloc[-1]:+.2f}σ')

    fig = plot_model_zscore(df, r2)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
