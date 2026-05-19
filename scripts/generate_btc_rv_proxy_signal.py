"""
generate_eth_combined_tvl.py
Alphaline Research — ETH Combined TVL Model (Compact 3-Panel)

Fetches stablecoin TVL, DeFi TVL, RWA TVL from DeFiLlama and ETH price
from Yahoo Finance (no API keys required), fits a log-log regression on
combined total secured value, and writes eth_combined_tvl.html to docs/.

Usage:
    python generate_eth_combined_tvl.py
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import yfinance as yf

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
ETH_SUPPLY         = 120_000_000
STAKING_RATE_INIT  = 0.28
ATTACK_THRESHOLD   = 0.33
ETH_TVL_THRESHOLD  = 1_000_000

LIDO_POOL_ID = '747c1d2a-c668-4682-b9f9-296708a3dd90'

OUTPUT_PATH = os.path.join('docs', 'eth_combined_tvl.html')

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
RED_LIT   = '#D64444'
GREEN_LIT = '#2ABF7A'

CHART_WIDTH  = 1100
CHART_HEIGHT = 920


# ════════════════════════════════════════════
# ALPHALINE CHART TEMPLATE
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT,
                     source='alphalineresearch.com  |  Yahoo Finance  |  DeFiLlama'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, autosize=True,
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
                 x=1.0, y=-0.133, xanchor='right', yanchor='middle',
                 font=dict(family='Courier New, monospace', size=10, color=MIST), showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>', xref='paper', yref='paper',
                 x=0.0, y=-0.133, xanchor='left', yanchor='middle',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD), showarrow=False),
        ],
    )
    return fig


# ════════════════════════════════════════════
# HTTP HELPER
# ════════════════════════════════════════════
def _get(url, timeout=90, retries=3, backoff=15):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            wait = backoff * (attempt + 1)
            print(f'  Retry {attempt+1}/{retries} after {wait}s ({type(e).__name__})')
            if attempt < retries - 1:
                time.sleep(wait)
    raise Exception(f'All {retries} attempts failed: {url}')


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
    print('Fetching RWA TVL...')
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


def fetch_staking_data():
    print('Fetching staked ETH (Lido)...')
    resp = _get('https://api.llama.fi/protocol/lido')
    eth_tvl_hist = resp.json().get('chainTvls', {}).get('Ethereum', {}).get('tvl', [])
    tvl_rows = [{'date': pd.Timestamp(int(pt['date']), unit='s').normalize(),
                 'staked_usd': float(pt['totalLiquidityUSD'])} for pt in eth_tvl_hist]
    tvl_df = pd.DataFrame(tvl_rows).set_index('date').sort_index()
    tvl_df = tvl_df[~tvl_df.index.duplicated(keep='last')].iloc[:-1]
    print(f'  latest: ${tvl_df["staked_usd"].iloc[-1]/1e9:.1f}B')
    return tvl_df


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
    stable  = fetch_stablecoin_tvl()
    defi    = fetch_defi_tvl()
    rwa     = fetch_rwa_tvl()
    staking = fetch_staking_data()
    eth     = fetch_eth_price()

    df = stable.copy()
    df = df.join(defi,    how='left')
    df = df.join(rwa,     how='left')
    df = df.join(staking, how='left')
    df = df.join(eth,     how='left')
    df = df.dropna(subset=['eth_price'])

    df['defi_tvl_usd'] = df['defi_tvl_usd'].ffill().fillna(0)
    df['rwa_tvl_usd']  = df['rwa_tvl_usd'].ffill().fillna(0)
    df['total_secured_usd'] = df['stable_tvl_usd'] + df['defi_tvl_usd'] + df['rwa_tvl_usd']

    df['stake_ratio_actual'] = STAKING_RATE_INIT
    df['total_to_mcap']      = df['total_secured_usd'] / df['mcap']

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


def apply_model(df, model, std, x_col, label, min_val=1e9):
    mask = df[x_col] > min_val
    df[f'eth_model_{label}'] = np.where(
        mask,
        np.exp(model.predict(np.log(df[x_col].clip(lower=min_val)).values.reshape(-1, 1))),
        np.nan
    )
    df[f'band_1up_{label}'] = df[f'eth_model_{label}'] * np.exp( 1 * std)
    df[f'band_1dn_{label}'] = df[f'eth_model_{label}'] * np.exp(-1 * std)
    df[f'band_2up_{label}'] = df[f'eth_model_{label}'] * np.exp( 2 * std)
    df[f'band_2dn_{label}'] = df[f'eth_model_{label}'] * np.exp(-2 * std)
    df[f'zscore_{label}']   = np.log(df['eth_price'] / df[f'eth_model_{label}']) / std
    return df


# ════════════════════════════════════════════
# CHART — ETH COMBINED TVL MODEL (3 panels)
# ════════════════════════════════════════════
def plot_model_compact(df, r2_a):
    model_df = df.dropna(subset=['eth_model'])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.22, 0.26],
        vertical_spacing=0.025
    )

    # ±2σ band
    fig.add_trace(go.Scatter(
        x=pd.concat([model_df.index.to_series(), model_df.index.to_series()[::-1]]),
        y=pd.concat([model_df['band_2up'], model_df['band_2dn'][::-1]]),
        fill='toself', fillcolor='rgba(212,168,67,0.06)',
        line=dict(width=0), showlegend=True, name='±2 Std Dev', hoverinfo='skip'
    ), row=1, col=1)

    # ±1σ band
    fig.add_trace(go.Scatter(
        x=pd.concat([model_df.index.to_series(), model_df.index.to_series()[::-1]]),
        y=pd.concat([model_df['band_1up'], model_df['band_1dn'][::-1]]),
        fill='toself', fillcolor='rgba(212,168,67,0.12)',
        line=dict(width=0), showlegend=True, name='±1 Std Dev', hoverinfo='skip'
    ), row=1, col=1)

    # Model fair value
    fig.add_trace(go.Scatter(
        x=model_df.index, y=model_df['eth_model'],
        mode='lines', line=dict(color='rgba(248,250,251,0.55)', width=1.4, dash='dash'),
        name='Model (combined TVL)', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Model: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ETH price
    fig.add_trace(go.Scatter(
        x=model_df.index, y=model_df['eth_price'],
        mode='lines', line=dict(color=GOLD, width=2.0),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_annotation(
        x=model_df.index[-1], y=model_df['eth_price'].iloc[-1],
        text=f'  ${model_df["eth_price"].iloc[-1]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD)
    )

    # Panel 2: Total Secured / Market Cap ratio
    ratio        = model_df['total_to_mcap']
    ratio_smooth = ratio.rolling(30, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=model_df.index, y=ratio,
        mode='lines', line=dict(color='rgba(212,168,67,0.25)', width=0.8),
        fill='tozeroy', fillcolor='rgba(212,168,67,0.07)',
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>Secured/Mcap: %{y:.2f}x<extra></extra>'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=model_df.index, y=ratio_smooth,
        mode='lines', name='Secured/Mcap 30d MA', line=dict(color=GOLD, width=1.5),
        showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>30d MA: %{y:.2f}x<extra></extra>'
    ), row=2, col=1)
    fig.add_annotation(
        x=model_df.index[-1], y=ratio.iloc[-1],
        text=f'  {ratio.iloc[-1]:.2f}x',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=9, color=GOLD)
    )

    # Panel 3: Z-score bars
    zscore  = model_df['zscore']
    zcolors = [
        RED_LIT                   if v >  2 else
        'rgba(214,68,68,0.55)'    if v >  1 else
        'rgba(214,68,68,0.30)'    if v >  0 else
        'rgba(42,191,122,0.30)'   if v > -1 else
        'rgba(42,191,122,0.55)'   if v > -2 else
        GREEN_LIT
        for v in zscore
    ]
    fig.add_trace(go.Bar(
        x=model_df.index, y=zscore, marker_color=zcolors, marker_opacity=1.0,
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>Z-score: %{y:+.2f}σ<extra></extra>'
    ), row=3, col=1)

    zscore_smooth = zscore.rolling(28, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=zscore_smooth.index, y=zscore_smooth, mode='lines',
        line=dict(color='rgba(248,250,251,0.40)', width=1.8),
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>28d MA: %{y:+.2f}σ<extra></extra>'
    ), row=3, col=1)

    for level, color in [(2, RED_LIT), (1, 'rgba(214,68,68,0.6)'),
                         (-1, 'rgba(42,191,122,0.6)'), (-2, GREEN_LIT)]:
        fig.add_hline(y=level, line_color=color, line_width=0.7,
                      line_dash='dot', row=3, col=1)
        fig.add_annotation(x=model_df.index[-1], y=level,
            text=f'  {level:+d}σ', showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=8, color=color))
    fig.add_hline(y=0, line_color=STEEL, line_width=0.8, row=3, col=1)

    fig.add_annotation(
        x=model_df.index[-1], y=zscore.iloc[-1],
        text=f'  {zscore.iloc[-1]:+.2f}σ', showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD_LIT)
    )

    latest = model_df.iloc[-1]
    title  = (
        f'ETH Combined TVL Model  |  '
        f'ETH: ${latest["eth_price"]:,.0f}  |  '
        f'Model: ${latest["eth_model"]:,.0f}  |  '
        f'Z: {latest["zscore"]:+.2f}σ  |  '
        f'R²={r2_a:.3f}'
    )
    alphaline_layout(fig, title, height=CHART_HEIGHT)

    # Shift footer annotations up to fit tighter margin
    new_anns = []
    for ann in fig.layout.annotations:
        a = ann.to_plotly_json()
        if a.get('yref') == 'paper' and isinstance(a.get('y'), (int, float)) and a['y'] < -0.05:
            a['y'] = -0.06
        new_anns.append(a)

    fig.update_layout(
        showlegend=True,
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL, borderwidth=1,
                    font=dict(size=9, color=MIST),
                    orientation='h', x=0.5, y=1.02,
                    xanchor='center', yanchor='bottom'),
        margin=dict(l=60, r=80, t=70, b=100),
        annotations=new_anns
    )
    fig.update_yaxes(type='log', tickprefix='$', title_text='ETH Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='Secured/Mcap', row=2, col=1)
    fig.update_yaxes(title_text='Z-score (σ)', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=1)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building ETH Combined TVL Model chart ===')
    df = build_dataframe()

    # Fit Model A — combined TVL (primary for this chart)
    model_a, r2_a, std_a = fit_model(df, 'total_secured_usd', min_val=1e9)
    df = apply_model(df, model_a, std_a, 'total_secured_usd', 'a', min_val=1e9)

    # Alias to generic column names expected by plot function
    df['eth_model'] = df['eth_model_a']
    df['band_1up']  = df['band_1up_a']
    df['band_1dn']  = df['band_1dn_a']
    df['band_2up']  = df['band_2up_a']
    df['band_2dn']  = df['band_2dn_a']
    df['zscore']    = df['zscore_a']

    print(f'Model R²: {r2_a:.3f} | Z-score: {df["zscore"].iloc[-1]:+.2f}σ')

    fig = plot_model_compact(df, r2_a)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
