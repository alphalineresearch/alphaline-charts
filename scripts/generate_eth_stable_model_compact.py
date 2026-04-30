"""
generate_eth_stable_model_compact.py
Alphaline Research — ETH vs Stablecoin Model (Compact 3-Panel)

Fetches stablecoin TVL, staking data, RWA TVL from DeFiLlama and ETH price
from Yahoo Finance (no API keys required), fits the log-log regression model,
and writes eth_stable_model_compact.html to docs/.

Usage:
    python generate_eth_stable_model_compact.py
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

OUTPUT_PATH = os.path.join('docs', 'eth_stable_model_compact.html')

# ════════════════════════════════════════════
# ALPHALINE BRAND COLORS
# ════════════════════════════════════════════
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GOLD      = '#D4A843'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
RED_LIT   = '#D64444'
GREEN_LIT = '#2ABF7A'

CHART_WIDTH  = 1100
CHART_HEIGHT = 750


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


def fetch_staking_data():
    print('Fetching staking data...')
    # APY — graceful fallback if slow
    apy_df = pd.DataFrame({'staking_apy': pd.Series(dtype=float)})
    apy_df.index.name = 'date'
    try:
        resp = _get(f'https://yields.llama.fi/chart/{LIDO_POOL_ID}', timeout=120, retries=3, backoff=20)
        apy_rows = []
        for entry in resp.json().get('data', []):
            ts = pd.Timestamp(entry['timestamp']).tz_localize(None).normalize()
            apy_rows.append({'date': ts, 'staking_apy': entry.get('apy', None)})
        apy_df = pd.DataFrame(apy_rows).set_index('date').sort_index()
        apy_df['staking_apy'] = apy_df['staking_apy'].rolling(7, min_periods=1).mean()
        print(f'  APY OK | latest: {apy_df["staking_apy"].dropna().iloc[-1]:.2f}%')
    except Exception as e:
        print(f'  APY fetch failed ({type(e).__name__}) — staking_apy will be NaN')

    # Staked ETH TVL via Lido
    resp2 = _get('https://api.llama.fi/protocol/lido')
    eth_tvl_hist = resp2.json().get('chainTvls', {}).get('Ethereum', {}).get('tvl', [])
    tvl_rows = [{'date': pd.Timestamp(int(pt['date']), unit='s').normalize(),
                 'staked_usd': float(pt['totalLiquidityUSD'])} for pt in eth_tvl_hist]
    tvl_df = pd.DataFrame(tvl_rows).set_index('date').sort_index()
    tvl_df = tvl_df[~tvl_df.index.duplicated(keep='last')]
    print(f'  Staked ETH OK | latest: ${tvl_df["staked_usd"].iloc[-1]/1e9:.1f}B')

    df = apy_df.join(tvl_df, how='outer').iloc[:-1]
    if 'staking_apy' not in df.columns:
        df['staking_apy'] = float('nan')
    return df


def fetch_eth_price():
    print('Fetching ETH price...')
    raw = yf.download('ETH-USD', period='max', interval='1d', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    eth = raw.dropna()
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if len(eth) and eth.index[-1] >= today:
        eth = eth.iloc[:-1]
    eth['mcap'] = eth['close'] * ETH_SUPPLY
    print(f'  {len(eth)} rows | latest: ${eth["close"].iloc[-1]:,.0f}')
    return eth


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
            time.sleep(0.3)
        except Exception as e:
            print(f'  ! {name}: {e}')

    if not all_series:
        return pd.DataFrame(columns=['rwa_tvl_usd'])

    rwa_df = pd.DataFrame(all_series)
    rwa_df.index = pd.to_datetime(rwa_df.index).normalize()
    rwa_daily = rwa_df.resample('D').last().ffill()
    rwa_total = rwa_daily.sum(axis=1).to_frame(name='rwa_tvl_usd').iloc[:-1]
    print(f'  latest: ${rwa_total["rwa_tvl_usd"].iloc[-1]/1e9:.2f}B')
    return rwa_total


# ════════════════════════════════════════════
# BUILD DATAFRAME + FIT MODEL
# ════════════════════════════════════════════
def build_dataframe():
    stable  = fetch_stablecoin_tvl()
    staking = fetch_staking_data()
    eth     = fetch_eth_price()
    rwa     = fetch_rwa_tvl()

    df = stable.copy()
    df = df.join(staking[['staking_apy', 'staked_usd']], how='left')
    df = df.join(eth[['close', 'mcap']].rename(columns={'close': 'eth_price'}), how='left')
    df = df.dropna(subset=['eth_price'])

    df['current_staked']  = df['staked_usd'] / df['eth_price']
    df['pct_staked']      = df['current_staked'] / ETH_SUPPLY * 100
    df['cir_supply']      = np.where(
        df['pct_staked'] > 0,
        df['current_staked'] / (df['pct_staked'] / 100),
        ETH_SUPPLY
    )
    df = df.join(rwa[['rwa_tvl_usd']], how='left')
    df['rwa_tvl_usd']      = df['rwa_tvl_usd'].fillna(0)
    df['total_secured_usd'] = df['stable_tvl_usd'] + df['rwa_tvl_usd']
    df['stable_to_mcap']   = df['stable_tvl_usd'] / df['mcap']
    df['stake_ratio_actual'] = df['pct_staked'].fillna(STAKING_RATE_INIT * 100) / 100
    df['supply_actual']    = df['cir_supply'].fillna(ETH_SUPPLY)

    return df


def fit_model(df, x_col, min_val=1e8):
    reg = df.dropna(subset=[x_col, 'eth_price'])
    reg = reg[reg[x_col] > min_val]
    lx  = np.log(reg[x_col].values).reshape(-1, 1)
    ly  = np.log(reg['eth_price'].values)
    m   = LinearRegression().fit(lx, ly)
    r2  = m.score(lx, ly)
    resid_std = (ly - m.predict(lx)).std()
    return m, r2, resid_std


def apply_model(df, model, std, x_col, label):
    mask = df[x_col] > 1e8
    df[f'eth_model_{label}'] = np.where(
        mask,
        np.exp(model.predict(np.log(df[x_col].clip(lower=1e8)).values.reshape(-1, 1))),
        np.nan
    )
    df[f'band_1up_{label}'] = df[f'eth_model_{label}'] * np.exp( 1 * std)
    df[f'band_1dn_{label}'] = df[f'eth_model_{label}'] * np.exp(-1 * std)
    df[f'band_2up_{label}'] = df[f'eth_model_{label}'] * np.exp( 2 * std)
    df[f'band_2dn_{label}'] = df[f'eth_model_{label}'] * np.exp(-2 * std)
    df[f'zscore_{label}']   = np.log(df['eth_price'] / df[f'eth_model_{label}']) / std
    return df


# ════════════════════════════════════════════
# CHART — ETH STABLE MODEL COMPACT (3 panels)
# ════════════════════════════════════════════
def plot_model_compact(df, r2):
    model_df = df.dropna(subset=['eth_model'])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.22, 0.26],
        vertical_spacing=0.025
    )

    # ── ±2 std band ──
    fig.add_trace(go.Scatter(
        x=pd.concat([model_df.index.to_series(), model_df.index.to_series()[::-1]]),
        y=pd.concat([model_df['band_2up'], model_df['band_2dn'][::-1]]),
        fill='toself', fillcolor='rgba(212,168,67,0.06)',
        line=dict(width=0), showlegend=True, name='±2 Std Dev', hoverinfo='skip'
    ), row=1, col=1)

    # ── ±1 std band ──
    fig.add_trace(go.Scatter(
        x=pd.concat([model_df.index.to_series(), model_df.index.to_series()[::-1]]),
        y=pd.concat([model_df['band_1up'], model_df['band_1dn'][::-1]]),
        fill='toself', fillcolor='rgba(212,168,67,0.12)',
        line=dict(width=0), showlegend=True, name='±1 Std Dev', hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=model_df.index, y=model_df['eth_model'],
        mode='lines', line=dict(color='rgba(248,250,251,0.55)', width=1.4, dash='dash'),
        name='Model', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Model: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=model_df.index, y=model_df['eth_price'],
        mode='lines', line=dict(color=GOLD, width=2.0),
        name='ETH Price', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ── Panel 2: TVL / Market Cap ratio ──
    ratio        = model_df['stable_to_mcap']
    ratio_smooth = ratio.rolling(30, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=model_df.index, y=ratio,
        mode='lines', line=dict(color='rgba(212,168,67,0.25)', width=0.8),
        fill='tozeroy', fillcolor='rgba(212,168,67,0.07)',
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>TVL/Mcap: %{y:.2f}x<extra></extra>'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=model_df.index, y=ratio_smooth,
        mode='lines', name='TVL/Mcap 30d', line=dict(color=GOLD, width=1.5),
        showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>30d MA: %{y:.2f}x<extra></extra>'
    ), row=2, col=1)
    fig.add_annotation(
        x=model_df.index[-1], y=ratio.iloc[-1],
        text=f'  {ratio.iloc[-1]:.2f}x',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=9, color=GOLD), row=2, col=1
    )

    # ── Panel 3: Z-score bars ──
    zscore  = model_df['zscore']
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
        x=model_df.index, y=zscore, marker_color=zcolors,
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>Z-score: %{y:+.2f}σ<extra></extra>'
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore.rolling(28, min_periods=7).mean(),
        mode='lines', line=dict(color='rgba(248,250,251,0.40)', width=1.8),
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>28d MA: %{y:+.2f}σ<extra></extra>'
    ), row=3, col=1)
    for level, color in [(2, RED_LIT), (1, 'rgba(214,68,68,0.6)'),
                         (-1, 'rgba(42,191,122,0.6)'), (-2, GREEN_LIT)]:
        fig.add_annotation(
            x=model_df.index[-1], y=level, text=f'  {level:+d}σ',
            showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=8, color=color), row=3, col=1
        )

    latest  = model_df.iloc[-1]
    z_label = f'{latest["zscore"]:+.2f}σ'
    title   = (
        f'ETH vs Stablecoin Model  |  '
        f'Price: ${latest["eth_price"]:,.0f}  |  '
        f'Model: ${latest["eth_model"]:,.0f}  |  '
        f'R²: {r2:.2f}  |  Z-score: {z_label}'
    )

    fig.update_layout(
        template='plotly_dark', paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=CHART_HEIGHT, autosize=True,
        title=dict(
            text=f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>',
            x=0.02, xanchor='left', y=0.98, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=55, r=38, t=65, b=90),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        showlegend=True,
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL, borderwidth=1,
                    font=dict(size=9, color=MIST),
                    orientation='h', x=0.5, y=1.02, xanchor='center', yanchor='bottom'),
        bargap=0,
        annotations=[
            dict(text='Source: alphalineresearch.com  |  Yahoo Finance  |  DeFiLlama',
                 xref='paper', yref='paper', x=1.0, y=-0.068,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=9, color=MIST), showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>',
                 xref='paper', yref='paper', x=0.075, y=-0.068,
                 xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=9, color=GOLD), showarrow=False),
        ],
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        xaxis2=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        xaxis3=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
    )
    fig.update_yaxes(title_text='ETH Price', row=1, col=1,
                     title_font=dict(size=10, color=MIST), type='log', tickformat='$,.0f')
    fig.update_yaxes(title_text='TVL / Mcap', row=2, col=1,
                     title_font=dict(size=10, color=MIST),
                     gridcolor='rgba(212,168,67,0.03)', gridwidth=0.4)
    fig.update_yaxes(title_text='Z-Score (σ)', row=3, col=1,
                     title_font=dict(size=10, color=MIST),
                     gridcolor='rgba(212,168,67,0.03)', gridwidth=0.4)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building ETH Stable Model Compact chart ===')
    df = build_dataframe()

    # Fit model A (stablecoins only) — used for this chart
    model_a, r2, std_a = fit_model(df, 'stable_tvl_usd')
    df = apply_model(df, model_a, std_a, 'stable_tvl_usd', 'a')

    # Alias to generic column names expected by plot function
    df['eth_model'] = df['eth_model_a']
    df['band_1up']  = df['band_1up_a']
    df['band_1dn']  = df['band_1dn_a']
    df['band_2up']  = df['band_2up_a']
    df['band_2dn']  = df['band_2dn_a']
    df['zscore']    = df['zscore_a']

    print(f'Model R²: {r2:.3f} | Z-score: {df["zscore"].iloc[-1]:+.2f}σ')

    fig = plot_model_compact(df, r2)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
