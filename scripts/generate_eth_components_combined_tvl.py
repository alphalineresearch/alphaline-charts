"""
generate_eth_proxy_signal.py
Alphaline Research — ETH RV7/RV30 Proxy Confluence Signal

Fetches ETH price from Yahoo Finance and stablecoin/DeFi/RWA TVL from
DeFiLlama (no API keys required), fits the combined TVL regression model,
computes the three-condition signal, and writes eth_proxy_signal.html to docs/.

Usage:
    python generate_eth_proxy_signal.py
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
# SIGNAL THRESHOLDS
# ════════════════════════════════════════════
RV_RATIO_THRESH   = 1.7    # RV7/RV30 > this = vol surprise
RV_EXTREME_THRESH = 2.5    # > this = extreme shock
DD_THRESH         = 20.0   # % below 90d high = in drawdown
ZSCORE_THRESH     = -1.0   # z-score <= this = below -1σ model

ETH_TVL_THRESHOLD = 1_000_000   # min $1M on Ethereum for RWA protocols

OUTPUT_PATH = os.path.join('docs', 'eth_proxy_signal.html')

# ════════════════════════════════════════════
# ALPHALINE BRAND COLORS
# ════════════════════════════════════════════
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GOLD      = '#D4A843'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN_LIT = '#2ABF7A'
TEAL      = '#2ABF7A'

CHART_WIDTH  = 1100
CHART_HEIGHT = 850


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ════════════════════════════════════════════
# ALPHALINE CHART TEMPLATE
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT,
                     source='alphalineresearch.com  |  Yahoo Finance · DeFiLlama'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, autosize=True,
        title=dict(
            text=f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>',
            x=0.02, xanchor='left', y=0.98, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=60, r=120, t=70, b=110),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL, borderwidth=1,
                    font=dict(size=9, color=MIST),
                    orientation='h', x=0.5, y=1.02,
                    xanchor='center', yanchor='bottom'),
        annotations=[
            dict(text=f'Source: {source}', xref='paper', yref='paper',
                 x=1.0, y=-0.16, xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=MIST), showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>', xref='paper', yref='paper',
                 x=0.07, y=-0.16, xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD), showarrow=False),
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
def fetch_eth_price():
    print('Fetching ETH price...')
    raw = yf.download('ETH-USD', period='max', interval='1d', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    px = raw['close'].dropna().rename('eth_price')
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if len(px) and px.index[-1] >= today:
        px = px.iloc[:-1]
    print(f'  {len(px)} rows | latest: ${px.iloc[-1]:,.0f}')
    return px


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
        print('  No RWA data — using zeros.')
        return pd.DataFrame(columns=['rwa_tvl_usd'])
    rwa_df = pd.DataFrame(all_series)
    rwa_df.index = pd.to_datetime(rwa_df.index).normalize()
    rwa_daily = rwa_df.resample('D').last().ffill()
    rwa_total = rwa_daily.sum(axis=1).to_frame(name='rwa_tvl_usd').iloc[:-1]
    print(f'  latest: ${rwa_total["rwa_tvl_usd"].iloc[-1]/1e9:.3f}B')
    return rwa_total


# ════════════════════════════════════════════
# BUILD DATAFRAME + MODEL + SIGNALS
# ════════════════════════════════════════════
def build_dataframe():
    eth_px = fetch_eth_price()
    stable = fetch_stablecoin_tvl()
    defi   = fetch_defi_tvl()
    rwa    = fetch_rwa_tvl()

    # RV metrics
    log_ret   = np.log(eth_px / eth_px.shift(1))
    rv7_full  = log_ret.rolling(7,  min_periods=5).std()  * np.sqrt(365) * 100
    rv30_full = log_ret.rolling(30, min_periods=20).std() * np.sqrt(365) * 100
    ratio_full = (rv7_full / rv30_full).replace([np.inf, -np.inf], np.nan)

    high_90d = eth_px.rolling(90).max()
    dd_90d   = (eth_px / high_90d - 1) * 100

    df = pd.concat([
        eth_px,
        rv7_full.rename('rv7'),
        rv30_full.rename('rv30'),
        ratio_full.rename('rv_ratio'),
        high_90d.rename('high_90d'),
        dd_90d.rename('dd_90d'),
    ], axis=1).dropna(subset=['rv_ratio'])

    # Join TVL components
    df = df.join(stable['stable_tvl_usd'], how='left')
    df = df.join(defi['defi_tvl_usd'],     how='left')
    if len(rwa) > 0:
        df = df.join(rwa['rwa_tvl_usd'], how='left')
    else:
        df['rwa_tvl_usd'] = 0.0

    df['stable_tvl_usd'] = df['stable_tvl_usd'].ffill()
    df['defi_tvl_usd']   = df['defi_tvl_usd'].ffill().fillna(0)
    df['rwa_tvl_usd']    = df['rwa_tvl_usd'].ffill().fillna(0)
    df['total_secured_usd'] = df['stable_tvl_usd'] + df['defi_tvl_usd'] + df['rwa_tvl_usd']

    # Fit combined TVL model
    reg_df = df.dropna(subset=['total_secured_usd', 'eth_price'])
    reg_df = reg_df[reg_df['total_secured_usd'] > 1e9]
    lx = np.log(reg_df['total_secured_usd'].values).reshape(-1, 1)
    ly = np.log(reg_df['eth_price'].values)
    model     = LinearRegression().fit(lx, ly)
    resid_std = (ly - model.predict(lx)).std()
    r2        = model.score(lx, ly)

    mask = df['total_secured_usd'] > 1e9
    df['eth_model'] = np.where(
        mask,
        np.exp(model.predict(np.log(df['total_secured_usd'].clip(lower=1e9)).values.reshape(-1, 1))),
        np.nan
    )
    df['band_1dn'] = df['eth_model'] * np.exp(-1 * resid_std)
    df['band_2up'] = df['eth_model'] * np.exp( 2 * resid_std)
    df['band_2dn'] = df['eth_model'] * np.exp(-2 * resid_std)
    df['zscore']   = np.log(df['eth_price'] / df['eth_model']) / resid_std

    print(f'Combined TVL model R²={r2:.3f} | Z-score: {df["zscore"].iloc[-1]:+.2f}σ')

    # Signal conditions
    is_vol_surprise = df['rv_ratio'] > RV_RATIO_THRESH
    is_drawdown     = df['dd_90d']   <= -DD_THRESH
    is_cheap        = df['zscore']   <= ZSCORE_THRESH

    sig_all = is_vol_surprise & is_drawdown & is_cheap

    return df, sig_all, r2


# ════════════════════════════════════════════
# CHART — ETH PROXY CONFLUENCE SIGNAL
# ════════════════════════════════════════════
def plot_eth_proxy_signal(df, sig_all):
    d = df.dropna(subset=['eth_model'])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.03
    )

    # Confluence episode bands
    s_all = d[sig_all.reindex(d.index, fill_value=False)]
    if len(s_all) > 0:
        conf_dates_list = s_all.index.tolist()
        episodes = []
        ep_s = ep_e = conf_dates_list[0]
        for j in range(1, len(conf_dates_list)):
            if (conf_dates_list[j] - conf_dates_list[j-1]).days > 14:
                episodes.append((ep_s, ep_e))
                ep_s = conf_dates_list[j]
            ep_e = conf_dates_list[j]
        episodes.append((ep_s, ep_e))

        for ep_s, ep_e in episodes:
            for _row in [1, 2]:
                fig.add_vrect(
                    x0=ep_s - pd.Timedelta(days=1),
                    x1=ep_e + pd.Timedelta(days=1),
                    fillcolor=hex_to_rgba(GREEN_LIT, 0.11),
                    line_width=0.7, line_color=hex_to_rgba(GREEN_LIT, 0.30),
                    layer='below', row=_row, col=1
                )
            # Solid green vertical lines at episode boundaries
            for ep_date in [ep_s, ep_e]:
                fig.add_vline(
                    x=str(ep_date.date()),
                    line_color=GREEN_LIT,
                    line_width=1.2,
                    opacity=0.5,
                )

    # ── Panel 1: ETH price + model bands + signals ──
    fig.add_trace(go.Scatter(
        x=pd.concat([d.index.to_series(), d.index.to_series()[::-1]]),
        y=pd.concat([d['band_2up'], d['band_2dn'][::-1]]),
        fill='toself', fillcolor='rgba(42,191,122,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['band_1dn'],
        mode='lines', name='-1σ band  (signal threshold)',
        line=dict(color=TEAL, width=1.0, dash='dot'),
        hovertemplate='%{x|%Y-%m-%d}<br>-1σ: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_model'],
        mode='lines', name='Combined TVL model (fair value)',
        line=dict(color=TEAL, width=1.5, dash='dash'),
        hovertemplate='%{x|%Y-%m-%d}<br>Model: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['eth_price'],
        mode='lines', name='ETH Price',
        line=dict(color=GOLD, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_annotation(
        x=d.index[-1], y=d['eth_price'].iloc[-1],
        text=f'  ${d["eth_price"].iloc[-1]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD)
    )

    if len(s_all):
        fig.add_trace(go.Scatter(
            x=s_all.index, y=s_all['eth_price'],
            mode='markers', name='● Confluence signal (Vol + DD + Model)',
            marker=dict(symbol='circle', size=7, color=GREEN_LIT, opacity=0.90,
                        line=dict(color=WHITE, width=0.6)),
            hovertemplate='%{x|%Y-%m-%d}<br>ETH: $%{y:,.0f}<br>Confluence: all 3<extra></extra>'
        ), row=1, col=1)

    # ── Panel 2: RV7/RV30 ratio ──
    rv    = d['rv_ratio'].dropna()
    rv_ma = rv.rolling(14, min_periods=5).mean()

    fig.add_hrect(y0=RV_RATIO_THRESH, y1=RV_EXTREME_THRESH,
                  fillcolor=hex_to_rgba(GOLD, 0.08),
                  line_width=0, layer='below', row=2, col=1)

    fig.add_trace(go.Scatter(
        x=rv.index, y=rv,
        mode='lines', name='RV7/RV30 ratio',
        line=dict(color=MIST, width=1.0), opacity=0.60,
        hovertemplate='%{x|%Y-%m-%d}<br>RV7/RV30: %{y:.2f}<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=rv_ma.index, y=rv_ma,
        mode='lines', name='14d MA ratio',
        line=dict(color=GOLD, width=1.6),
        hovertemplate='%{x|%Y-%m-%d}<br>14d MA: %{y:.2f}<extra></extra>'
    ), row=2, col=1)

    for thresh, color, label in [
        (1.0,             STEEL, '1.0×'),
        (RV_RATIO_THRESH, GOLD,  f'{RV_RATIO_THRESH}×'),
    ]:
        fig.add_shape(type='line',
            x0=rv.index[0], x1=rv.index[-1],
            y0=thresh, y1=thresh,
            line=dict(color=color, width=0.8, dash='dot'),
            row=2, col=1)
        fig.add_annotation(
            x=rv.index[-1], y=thresh,
            text=f'  {label}', showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=8, color=color),
            row=2, col=1)

    if len(s_all):
        s_all_rv = rv.reindex(s_all.index).dropna()
        fig.add_trace(go.Scatter(
            x=s_all_rv.index, y=s_all_rv,
            mode='markers', showlegend=False,
            marker=dict(symbol='circle', size=7, color=GREEN_LIT, opacity=0.90),
            hovertemplate='%{x|%Y-%m-%d}<br>RV ratio: %{y:.2f}<br>Confluence: all 3<extra></extra>'
        ), row=2, col=1)

    # Layout
    title = 'ETH RV7/RV30 Proxy Signal  |  ● Confluence = Vol Surprise + DD + Near Model'
    alphaline_layout(fig, title, height=CHART_HEIGHT)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation='h', x=0.5, y=1.02, xanchor='center', yanchor='bottom',
                    font=dict(size=9, color=MIST),
                    bgcolor='rgba(10,22,40,0.85)', bordercolor=STEEL, borderwidth=1),
        margin=dict(b=150)
    )
    fig.update_yaxes(type='log', tickprefix='$', title_text='ETH Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='RV7/RV30 ratio', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building ETH Proxy Confluence Signal chart ===')
    df, sig_all, r2 = build_dataframe()
    fig = plot_eth_proxy_signal(df, sig_all)

    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
