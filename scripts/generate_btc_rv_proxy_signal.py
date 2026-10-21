"""
generate_btc_rv_proxy_signal.py
Alphaline Research — BTC RV7/RV30 Proxy Confluence Signal

Fetches BTC price from Yahoo Finance and mining difficulty from blockchain.com
(no API keys required), computes the RV7/RV30 ratio signal, and writes
btc_rv7_rv30_proxy_signal.html to docs/.

Usage:
    python generate_btc_rv_proxy_signal.py
"""

import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from alphaline_theme import apply_theme, write_chart, pct, date_long

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
RV_RATIO_THRESH   = 1.8    # RV7/RV30 > this = vol surprise
RV_EXTREME_THRESH = 2.5    # > this = extreme shock
DD_THRESH_PROXY   = 20.0   # % below 90d high
COST_THRESH_PROXY = 1.5    # price/cost multiple ceiling for confluence

EFFICIENCY_SCHEDULE = {
    '2013-01-01': 1000, '2014-01-01': 500,  '2016-01-01': 200,
    '2018-01-01': 100,  '2019-01-01': 75,   '2020-01-01': 60,
    '2021-01-01': 40,   '2022-06-01': 32,   '2023-01-01': 25,
    '2024-01-01': 22,
}
ELECTRICITY_COST = 0.05
PUE              = 1.10
HALVING_SCHEDULE = [
    ('2009-01-03', 50.0),  ('2012-11-28', 25.0),
    ('2016-07-09', 12.5),  ('2020-05-11', 6.25),
    ('2024-04-20', 3.125),
]

OUTPUT_PATH = os.path.join('docs', 'btc_rv7_rv30_proxy_signal.html')

# ════════════════════════════════════════════
# ALPHALINE BRAND COLORS
# ════════════════════════════════════════════
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN_LIT = '#2ABF7A'


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ════════════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════════════
def fetch_btc_price():
    print('Fetching BTC price...')
    raw = yf.download('BTC-USD', period='max', interval='1d', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    px = raw['close'].dropna().rename('btc_price')
    # Drop today's incomplete candle so all charts show the same last closed day
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if len(px) and px.index[-1] >= today:
        px = px.iloc[:-1]
    print(f'  {len(px)} rows | latest: ${px.iloc[-1]:,.0f}')
    return px


def fetch_production_cost():
    """Estimate BTC production cost from blockchain.com mining difficulty."""
    print('Fetching mining difficulty from blockchain.com...')
    try:
        url = ('https://api.blockchain.info/charts/difficulty'
               '?timespan=all&format=json&sampled=false')
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        rows = [{'date': pd.Timestamp(int(pt['x']), unit='s').normalize(),
                 'value': float(pt['y'])} for pt in resp.json().get('values', [])]
        diff_raw = pd.DataFrame(rows).set_index('date').sort_index()
        diff_raw = diff_raw[~diff_raw.index.duplicated(keep='last')]
        diff_raw.columns = ['avg_difficulty']
        diff_raw['hashrate_th'] = diff_raw['avg_difficulty'] * (2 ** 32) / 600 / 1e12
        diff_raw.index = pd.to_datetime(diff_raw.index).tz_localize(None)

        def get_block_reward(date):
            reward = 50.0
            for h_date, h_reward in HALVING_SCHEDULE:
                if date >= pd.Timestamp(h_date):
                    reward = h_reward
            return reward

        def get_efficiency(date):
            eff = 1000.0
            for ts_str, val in sorted(EFFICIENCY_SCHEDULE.items()):
                if date >= pd.Timestamp(ts_str):
                    eff = val
            return eff

        diff_raw['block_reward']    = diff_raw.index.map(get_block_reward)
        diff_raw['btc_mined_daily'] = diff_raw['block_reward'] * 144
        diff_raw['efficiency_jth']  = diff_raw.index.map(get_efficiency)
        diff_raw['production_cost'] = (
            diff_raw['hashrate_th'] * diff_raw['efficiency_jth'] *
            PUE * 86400 / 3.6e6 * ELECTRICITY_COST / diff_raw['btc_mined_daily']
        )
        cost_series = diff_raw['production_cost'].rolling(30).mean().dropna()
        cost_series.name = 'cost_30d'
        print(f'  {len(cost_series)} rows | latest: ${cost_series.iloc[-1]:,.0f}')
        return cost_series, True
    except Exception as e:
        print(f'  Production cost fetch failed: {e} — proceeding without cost signal')
        return pd.Series(dtype=float, name='cost_30d'), False


# ════════════════════════════════════════════
# BUILD DATAFRAME
# ════════════════════════════════════════════
def build_dataframe():
    btc_px = fetch_btc_price()
    cost_series, has_cost = fetch_production_cost()

    log_ret   = np.log(btc_px / btc_px.shift(1))
    rv7_full  = log_ret.rolling(7,  min_periods=5).std()  * np.sqrt(365) * 100
    rv30_full = log_ret.rolling(30, min_periods=20).std() * np.sqrt(365) * 100
    ratio_full = (rv7_full / rv30_full).replace([np.inf, -np.inf], np.nan)

    df = pd.concat([
        btc_px.rename('btc_price'),
        rv7_full.rename('rv7'),
        rv30_full.rename('rv30'),
        ratio_full.rename('rv_ratio'),
    ], axis=1).dropna()

    df['high_90d'] = df['btc_price'].rolling(90).max()
    df['dd_90d']   = (df['btc_price'] / df['high_90d'] - 1) * 100

    if has_cost:
        df = df.join(cost_series.rename('cost_30d'), how='left')
        df['cost_30d']     = df['cost_30d'].ffill()
        df['cost_multiple'] = df['btc_price'] / df['cost_30d']
        df['near_cost']    = df['cost_multiple'] <= COST_THRESH_PROXY
    else:
        df['cost_30d']     = np.nan
        df['cost_multiple'] = np.nan
        df['near_cost']    = False

    return df, has_cost


# ════════════════════════════════════════════
# CHART — BTC RV7/RV30 PROXY CONFLUENCE SIGNAL
# ════════════════════════════════════════════
def plot_rv_proxy(df, has_cost):
    d = df.copy()

    is_vol_surprise      = d['rv_ratio'] > RV_RATIO_THRESH
    in_drawdown          = d['dd_90d']  <= -DD_THRESH_PROXY
    near_cost_p          = d['near_cost'].fillna(False)
    sig_proxy_confluence = is_vol_surprise & in_drawdown & near_cost_p

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.03
    )

    # ── Panel 1: BTC price (log) ──
    fig.add_trace(go.Scatter(
        x=d.index, y=d['btc_price'],
        mode='lines', name='BTC Price',
        line=dict(color=GOLD, width=1.4),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # Production cost lines
    if has_cost:
        cost_p = d['cost_30d'].dropna()
        fig.add_trace(go.Scatter(
            x=cost_p.index, y=cost_p,
            mode='lines', name='Production Cost (30d)',
            line=dict(color=GREEN_LIT, width=0.9, dash='dot'),
            opacity=0.55,
            hovertemplate='%{x|%Y-%m-%d}<br>Cost: $%{y:,.0f}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=cost_p.index, y=cost_p * COST_THRESH_PROXY,
            mode='lines', name=f'{COST_THRESH_PROXY}× Cost',
            line=dict(color=GREEN_LIT, width=0.6, dash='dot'),
            opacity=0.30,
            hovertemplate='%{x|%Y-%m-%d}<br>' + f'{COST_THRESH_PROXY}×' + ' Cost: $%{y:,.0f}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=list(cost_p.index) + list(cost_p.index[::-1]),
            y=list(cost_p * COST_THRESH_PROXY) + list(cost_p[::-1]),
            fill='toself',
            fillcolor=hex_to_rgba(GREEN_LIT, 0.05),
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

    # Confluence episode bands + dots
    prx_conf_days = d[sig_proxy_confluence]
    if len(prx_conf_days) > 0:
        conf_dates_list = prx_conf_days.index.tolist()
        episodes = []
        ep_s = ep_e = conf_dates_list[0]
        for j in range(1, len(conf_dates_list)):
            if (conf_dates_list[j] - conf_dates_list[j - 1]).days > 14:
                episodes.append((ep_s, ep_e))
                ep_s = conf_dates_list[j]
            ep_e = conf_dates_list[j]
        episodes.append((ep_s, ep_e))

        for ep_s, ep_e in episodes:
            for band_row in [1, 2]:
                fig.add_vrect(
                    x0=ep_s - pd.Timedelta(days=1),
                    x1=ep_e + pd.Timedelta(days=1),
                    fillcolor=hex_to_rgba(GREEN_LIT, 0.11),
                    line_width=0.7, line_color=hex_to_rgba(GREEN_LIT, 0.30),
                    layer='below',
                    row=band_row, col=1
                )

        fig.add_trace(go.Scatter(
            x=prx_conf_days.index, y=prx_conf_days['btc_price'],
            mode='markers',
            name='Confluence signal (DD + Vol Spike + Near Cost)',
            marker=dict(color=GREEN_LIT, size=7, opacity=0.90,
                        symbol='circle',
                        line=dict(color=WHITE, width=0.6)),
            hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Confluence</extra>'
        ), row=1, col=1)

    # ── Panel 2: RV7/RV30 ratio ──
    ratio = d['rv_ratio'].dropna()

    fig.add_hrect(y0=RV_RATIO_THRESH, y1=RV_EXTREME_THRESH,
                  fillcolor=hex_to_rgba(GOLD, 0.08),
                  line_width=0, layer='below', row=2, col=1)

    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio,
        mode='lines', name='RV7/RV30 ratio',
        line=dict(color=MIST, width=1.0),
        opacity=0.60,
        hovertemplate='%{x|%Y-%m-%d}<br>RV ratio: %{y:.2f}×<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.rolling(14).mean(),
        mode='lines', name='14d MA ratio',
        line=dict(color=GOLD, width=1.6),
        hovertemplate='%{x|%Y-%m-%d}<br>14d MA: %{y:.2f}×<extra></extra>'
    ), row=2, col=1)

    for thresh, color, label in [
        (1.0,             STEEL, '1.0×'),
        (RV_RATIO_THRESH, GOLD,  f'{RV_RATIO_THRESH}× signal threshold'),
    ]:
        fig.add_shape(type='line',
            x0=ratio.index[0], x1=ratio.index[-1],
            y0=thresh, y1=thresh,
            line=dict(color=color, width=0.8, dash='dot'),
            row=2, col=1)
        # label inside the right edge so it never clips
        fig.add_annotation(
            xref='x domain', x=0.995, y=thresh, yshift=8,
            text=label, showarrow=False, xanchor='right',
            font=dict(color=MIST if color == STEEL else color),
            row=2, col=1)

    if len(prx_conf_days) > 0:
        fig.add_trace(go.Scatter(
            x=prx_conf_days.index,
            y=d.loc[prx_conf_days.index, 'rv_ratio'],
            mode='markers', name='Confluence on ratio',
            marker=dict(color=GREEN_LIT, size=7, symbol='circle'),
            showlegend=False,
            hovertemplate='%{x|%Y-%m-%d}<br>RV ratio: %{y:.2f}×<extra></extra>'
        ), row=2, col=1)

    # ── Live reading for the page subtitle ──
    last = d.iloc[-1]
    on_now = bool(sig_proxy_confluence.iloc[-1])
    parts = [f'<b>RV7/RV30 {last["rv_ratio"]:.2f}×</b> (fires above {RV_RATIO_THRESH}×)',
             f'{pct(last["dd_90d"])} from 90-day high']
    if has_cost:
        parts.append(f'{last["cost_multiple"]:.2f}× production cost')
    parts.append('confluence on' if on_now else
                 (f'last confluence {date_long(prx_conf_days.index[-1])}' if len(prx_conf_days) else 'no confluence yet'))
    parts.append(date_long(d.index[-1]))
    subtitle = '  ·  '.join(parts)

    apply_theme(fig)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='BTC Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='RV7/RV30 ratio', ticksuffix='×', row=2, col=1)
    fig.update_xaxes(title_text='', row=2, col=1)
    return fig, subtitle


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building BTC RV7/RV30 Proxy Signal chart ===')
    df, has_cost = build_dataframe()
    fig, subtitle = plot_rv_proxy(df, has_cost)
    write_chart(fig, OUTPUT_PATH,
                title='BTC Volatility Shock Signal',
                subtitle=subtitle,
                note=(f'Confluence (green) = 7-day realised vol above {RV_RATIO_THRESH}× the 30-day, '
                      f'price {DD_THRESH_PROXY:.0f}%+ below its 90-day high, and within {COST_THRESH_PROXY}× production cost'),
                source='Yahoo Finance · blockchain.info', default='5Y', min_plot_height=420)
    print(f'Saved: {OUTPUT_PATH}')
