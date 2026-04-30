"""
generate_mstr_pnav_heatmap.py
Alphaline Research — MSTR P/NAV Z-Score Heatmap + BTC Production Cost

Data sources (no API keys required):
  - Yahoo Finance: MSTR price, BTC price, shares outstanding (via yfinance)
  - blockchain.com: mining difficulty → BTC production cost
  - Strategy (MSTR) SEC filings: BTC holdings lookup table (hardcoded, update periodically)

Writes docs/mstr_pnav_heatmap.html.

Usage:
    python generate_mstr_pnav_heatmap.py
"""

import os
import warnings
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_PATH   = os.path.join('docs', 'mstr_pnav_heatmap.html')
START         = '2020-08-01'
ZSCORE_START  = '2022-01-01'
SPLIT_DATE    = pd.Timestamp('2024-08-01')
SPLIT_FACTOR  = 10
BULL_THRESH   = 2.0
BEAR_THRESH   = 1.2
SMOOTH_SPAN   = 14
COST_BAND_MULT = 1.5

EFFICIENCY_SCHEDULE = {
    '2013-01-01': 1000, '2014-01-01': 500,  '2016-01-01': 200,
    '2018-01-01': 100,  '2019-01-01': 75,   '2020-01-01': 60,
    '2021-01-01': 40,   '2022-06-01': 32,   '2023-01-01': 25,
    '2024-01-01': 22,
}
ELECTRICITY_COST = 0.05   # $/kWh
PUE              = 1.10
HALVING_SCHEDULE = [
    ('2009-01-03', 50.0), ('2012-11-28', 25.0),
    ('2016-07-09', 12.5), ('2020-05-11', 6.25),
    ('2024-04-20', 3.125),
]

# ════════════════════════════════════════════
# BRAND COLORS
# ════════════════════════════════════════════
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
CHART_HEIGHT = 870

def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'

# ════════════════════════════════════════════
# ALPHALINE LAYOUT
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT,
                     source='alphalineresearch.com  |  Yahoo Finance · Strategy (MSTR) · blockchain.com'):
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
            dict(text=f'Source: {source}',
                 xref='paper', yref='paper', x=1.0, y=-0.13,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=MIST),
                 showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>',
                 xref='paper', yref='paper', x=0.0, y=-0.13,
                 xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD),
                 showarrow=False),
        ],
    )
    return fig

# ════════════════════════════════════════════
# BTC HOLDINGS LOOKUP TABLE
# Source: Strategy (MSTR) SEC filings / press releases
# Verify & update from: saylortracker.com  or  sec.gov
# ════════════════════════════════════════════
BTC_HOLDINGS_RAW = [
    ('2020-08-11',   21_454),
    ('2020-09-14',   38_250),
    ('2020-12-04',   40_824),
    ('2020-12-21',   70_470),
    ('2021-01-22',   70_784),
    ('2021-02-02',   71_079),
    ('2021-02-24',   90_531),
    ('2021-03-12',   91_326),
    ('2021-05-18',   92_079),
    ('2021-06-21',  105_085),
    ('2021-08-24',  108_992),
    ('2021-09-13',  114_042),
    ('2021-11-01',  121_044),
    ('2021-12-09',  122_478),
    ('2021-12-30',  124_391),
    ('2022-01-31',  125_051),
    ('2022-03-29',  129_218),
    ('2022-06-28',  129_699),
    ('2022-12-27',  132_500),
    ('2023-03-27',  138_955),
    ('2023-06-27',  152_333),
    ('2023-09-24',  158_245),
    ('2023-12-26',  189_150),
    ('2024-03-18',  214_278),
    ('2024-06-20',  226_500),
    ('2024-09-13',  252_220),
    ('2024-11-11',  279_420),
    ('2024-11-18',  331_200),
    ('2024-11-25',  386_700),
    ('2024-12-02',  402_100),
    ('2024-12-09',  423_650),
    ('2024-12-22',  446_400),
    ('2025-01-27',  471_107),
    ('2025-02-24',  499_096),
    ('2025-03-31',  506_137),
    ('2025-04-07',  528_185),
]

def _build_btc_holdings():
    _btc = pd.DataFrame(BTC_HOLDINGS_RAW, columns=['date', 'btc_held'])
    _btc['date'] = pd.to_datetime(_btc['date'])
    _btc = _btc.set_index('date').sort_index()
    _daily_idx = pd.date_range(_btc.index[0], datetime.utcnow().date(), freq='D')
    return _btc.reindex(_daily_idx).ffill().rename_axis('date')

# ════════════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════════════
def fetch_price_data():
    """Fetch MSTR price, BTC price, and shares outstanding via yfinance."""
    print('Fetching MSTR price...')
    mstr_raw = yf.download('MSTR', start=START, interval='1d', progress=False)
    if isinstance(mstr_raw.columns, pd.MultiIndex):
        mstr_raw.columns = [c[0].lower() for c in mstr_raw.columns]
    else:
        mstr_raw.columns = [c.lower() for c in mstr_raw.columns]
    mstr_raw.index = pd.to_datetime(mstr_raw.index).tz_localize(None)
    mstr_px = mstr_raw['close'].dropna().rename('mstr_price')
    print(f'  MSTR: {len(mstr_px)} rows | latest ${mstr_px.iloc[-1]:,.2f}')

    print('Fetching BTC price...')
    btc_raw = yf.download('BTC-USD', start=START, interval='1d', progress=False)
    if isinstance(btc_raw.columns, pd.MultiIndex):
        btc_raw.columns = [c[0].lower() for c in btc_raw.columns]
    else:
        btc_raw.columns = [c.lower() for c in btc_raw.columns]
    btc_raw.index = pd.to_datetime(btc_raw.index).tz_localize(None)
    btc_px = btc_raw['close'].dropna().rename('btc_price')
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if len(btc_px) and btc_px.index[-1] >= today:
        btc_px = btc_px.iloc[:-1]
    print(f'  BTC:  {len(btc_px)} rows | latest ${btc_px.iloc[-1]:,.0f}')

    print('Fetching MSTR shares outstanding...')
    ticker = yf.Ticker('MSTR')
    shares_daily = None
    try:
        shares_raw = ticker.get_shares_full(start=START)
        if shares_raw is not None and len(shares_raw) > 5:
            shares_raw.index = pd.to_datetime(shares_raw.index).tz_localize(None)
            shares_raw = shares_raw[~shares_raw.index.duplicated(keep='last')].sort_index()
            shares_raw[shares_raw.index < SPLIT_DATE] *= SPLIT_FACTOR
            _sidx = pd.date_range(shares_raw.index[0], datetime.utcnow().date(), freq='D')
            shares_daily = shares_raw.reindex(_sidx).ffill().rename_axis('date').rename('shares')
            print(f'  Shares: {len(shares_raw)} obs | latest {shares_daily.iloc[-1]/1e6:.1f}M (split-adj)')
    except Exception as e:
        print(f'  get_shares_full failed: {e}')

    if shares_daily is None:
        print('  Using fallback quarterly shares table (split-adjusted)...')
        _fallback = [
            ('2020-08-01',  105_000_000), ('2021-01-01',  107_000_000),
            ('2021-07-01',  109_000_000), ('2022-01-01',  110_000_000),
            ('2022-07-01',  111_000_000), ('2023-01-01',  133_000_000),
            ('2023-07-01',  150_000_000), ('2024-01-01',  163_000_000),
            ('2024-08-01',  210_000_000), ('2024-10-01',  230_000_000),
            ('2025-01-01',  280_000_000),
        ]
        _sf = pd.DataFrame(_fallback, columns=['date', 'shares'])
        _sf['date'] = pd.to_datetime(_sf['date'])
        _sf = _sf.set_index('date').sort_index()
        _sidx = pd.date_range(_sf.index[0], datetime.utcnow().date(), freq='D')
        shares_daily = _sf.reindex(_sidx).ffill()['shares'].rename_axis('date').rename('shares')

    return mstr_px, btc_px, shares_daily


def fetch_production_cost():
    """Fetch BTC mining difficulty from blockchain.com and compute production cost."""
    print('Fetching mining difficulty from blockchain.com...')
    try:
        url = ('https://api.blockchain.info/charts/difficulty'
               '?timespan=all&format=json&sampled=false')
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        rows = [{'date': pd.Timestamp(int(pt['x']), unit='s').normalize(),
                 'value': float(pt['y'])} for pt in resp.json().get('values', [])]
        diff_raw = pd.DataFrame(rows).set_index('date').sort_index()
        diff_raw = diff_raw[~diff_raw.index.duplicated(keep='last')]
        diff_raw.index = pd.to_datetime(diff_raw.index).tz_localize(None)
        diff_raw.columns = ['avg_difficulty']
        diff_raw['hashrate_th'] = diff_raw['avg_difficulty'] * (2**32) / 600 / 1e12

        def _block_reward(date):
            r = 50.0
            for h_date, h_r in HALVING_SCHEDULE:
                if date >= pd.Timestamp(h_date):
                    r = h_r
            return r

        def _efficiency(date):
            e = 1000.0
            for ts, val in sorted(EFFICIENCY_SCHEDULE.items()):
                if date >= pd.Timestamp(ts):
                    e = val
            return e

        diff_raw['block_reward']    = diff_raw.index.map(_block_reward)
        diff_raw['btc_mined_daily'] = diff_raw['block_reward'] * 144
        diff_raw['efficiency_jth']  = diff_raw.index.map(_efficiency)
        diff_raw['production_cost'] = (
            diff_raw['hashrate_th'] * diff_raw['efficiency_jth'] *
            PUE * 86400 / 3.6e6 * ELECTRICITY_COST / diff_raw['btc_mined_daily']
        )
        cost_series = diff_raw['production_cost'].rolling(30).mean().dropna()
        cost_series.name = 'cost_30d'
        print(f'  Production cost: {len(cost_series)} rows | latest ${cost_series.iloc[-1]:,.0f}')
        return cost_series, True
    except Exception as e:
        print(f'  Production cost fetch failed: {e} — proceeding without')
        return pd.Series(dtype=float, name='cost_30d'), False


# ════════════════════════════════════════════
# BUILD DATAFRAME
# ════════════════════════════════════════════
def build_dataframe():
    btc_holdings = _build_btc_holdings()
    mstr_px, btc_px, shares_daily = fetch_price_data()
    cost_series, has_cost = fetch_production_cost()

    df = pd.concat([mstr_px, btc_px], axis=1).dropna()
    df = df.join(btc_holdings['btc_held'], how='left')
    df = df.join(shares_daily.rename('shares'), how='left')
    df['btc_held'] = df['btc_held'].ffill()
    df['shares']   = df['shares'].ffill()
    df = df.dropna(subset=['btc_held', 'shares'])

    df['btc_nav']     = df['btc_held'] * df['btc_price']
    df['mstr_mktcap'] = df['mstr_price'] * df['shares']
    df['pnav']        = df['mstr_mktcap'] / df['btc_nav']
    df['pnav_smooth'] = df['pnav'].ewm(span=SMOOTH_SPAN, adjust=False).mean()

    # Z-score (calibrated to post-2022 era only)
    cal       = df.loc[ZSCORE_START:, 'pnav_smooth']
    pnav_mu   = cal.mean()
    pnav_std  = cal.std()
    df['pnav_zscore']      = (df['pnav_smooth'] - pnav_mu) / pnav_std
    df['pnav_zscore_clip'] = df['pnav_zscore'].clip(-3, 3)

    print(f'\nDataFrame: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}')
    print(f'Z-score: mu={pnav_mu:.2f}x  sigma={pnav_std:.2f}x  '
          f'current={df["pnav_zscore"].iloc[-1]:+.2f}s ({df["pnav_smooth"].iloc[-1]:.2f}x)')

    return df, cost_series, has_cost, pnav_mu, pnav_std


# ════════════════════════════════════════════
# REGIME HELPERS
# ════════════════════════════════════════════
def build_episodes(df, mask, gap_days=5):
    """Merge consecutive True days into (start, end) episodes."""
    dates = df.index[mask].tolist()
    if not dates:
        return []
    eps, ep_s, ep_e = [], dates[0], dates[0]
    for d in dates[1:]:
        if (d - ep_e).days <= gap_days + 1:
            ep_e = d
        else:
            eps.append((ep_s, ep_e))
            ep_s = ep_e = d
    eps.append((ep_s, ep_e))
    return eps


# ════════════════════════════════════════════
# CHART FUNCTION
# ════════════════════════════════════════════
def plot_heatmap_chart(df, cost_series, has_cost, pnav_mu, pnav_std):
    is_bull    = df['pnav_smooth'] >= BULL_THRESH
    is_bear    = df['pnav_smooth'] <= BEAR_THRESH
    bull_eps   = build_episodes(df, is_bull)
    bear_eps   = build_episodes(df, is_bear)
    regime_now = 'BULL' if is_bull.iloc[-1] else ('BEAR' if is_bear.iloc[-1] else 'NEUTRAL')
    pnav_now   = df['pnav_smooth'].iloc[-1]
    zscore_now = df['pnav_zscore'].iloc[-1]

    # Diverging colorscale: RED (bear) -> STEEL (neutral) -> GREEN (bull)
    CSCALE = [
        [0.00, RED_LIT],
        [0.30, '#7A2A2A'],
        [0.48, STEEL],
        [0.52, STEEL],
        [0.70, '#1A6A45'],
        [1.00, GREEN_LIT],
    ]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.03
    )

    # ── Bull/bear vrects on BTC panel only ──────────────────
    for ep_s, ep_e in bull_eps:
        fig.add_vrect(
            x0=ep_s - pd.Timedelta(days=0.5),
            x1=ep_e + pd.Timedelta(days=0.5),
            fillcolor=hex_to_rgba(GREEN_LIT, 0.12),
            line_width=0.5, line_color=hex_to_rgba(GREEN_LIT, 0.25),
            layer='below', row=1, col=1
        )
    for ep_s, ep_e in bear_eps:
        fig.add_vrect(
            x0=ep_s - pd.Timedelta(days=0.5),
            x1=ep_e + pd.Timedelta(days=0.5),
            fillcolor=hex_to_rgba(RED_LIT, 0.10),
            line_width=0.5, line_color=hex_to_rgba(RED_LIT, 0.20),
            layer='below', row=1, col=1
        )

    # ── Production cost band (panel 1) ──────────────────────
    if has_cost:
        cost_p = cost_series.reindex(df.index, method='ffill').dropna()

        fig.add_trace(go.Scatter(
            x=list(cost_p.index) + list(cost_p.index[::-1]),
            y=list(cost_p * COST_BAND_MULT) + list(cost_p[::-1]),
            fill='toself',
            fillcolor=hex_to_rgba(GREEN_LIT, 0.08),
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=cost_p.index, y=cost_p * COST_BAND_MULT,
            mode='lines',
            name=f'Cost x{COST_BAND_MULT} (near-cost zone)',
            line=dict(color=hex_to_rgba(GREEN_LIT, 0.5), width=0.8, dash='dot'),
            hovertemplate='%{x|%Y-%m-%d}<br>Cost x' + str(COST_BAND_MULT) +
                          ': $%{y:,.0f}<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=cost_p.index, y=cost_p,
            mode='lines',
            name='BTC Production Cost (30d)',
            line=dict(color=GREEN_LIT, width=1.6),
            hovertemplate='%{x|%Y-%m-%d}<br>Cost: $%{y:,.0f}<extra></extra>',
        ), row=1, col=1)

        fig.add_annotation(
            x=cost_p.index[-1], y=cost_p.iloc[-1],
            text=f'  cost ${cost_p.iloc[-1]:,.0f}',
            showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=9, color=GREEN_LIT),
        )

    # ── Panel 1: BTC price (log) ─────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df['btc_price'],
        mode='lines', name='BTC Price',
        line=dict(color=GOLD, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>BTC: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_annotation(
        x=df.index[-1], y=df['btc_price'].iloc[-1],
        text=f'  ${df["btc_price"].iloc[-1]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD)
    )

    # ── Panel 2: P/NAV heatmap bars ─────────────────────────
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['pnav_smooth'],
        name='P/NAV (z-score heatmap)',
        showlegend=False,
        marker=dict(
            color=df['pnav_zscore_clip'],
            colorscale=CSCALE,
            cmin=-3, cmax=3,
            colorbar=dict(
                title=dict(
                    text='z-score',
                    font=dict(family='Courier New, monospace', size=9, color=MIST),
                    side='right',
                ),
                tickfont=dict(family='Courier New, monospace', size=8, color=MIST),
                tickvals=[-3, -2, -1, 0, 1, 2, 3],
                ticktext=['-3s', '-2s', '-1s', '  0', '+1s', '+2s', '+3s'],
                len=0.38, thickness=11,
                x=1.03, y=0.0, yanchor='bottom',
                bgcolor=NAVY, bordercolor=STEEL, borderwidth=1,
            ),
            line_width=0,
        ),
        hovertemplate=(
            '%{x|%Y-%m-%d}<br>'
            'P/NAV: %{y:.2f}x<br>'
            'z-score: %{marker.color:+.2f}s'
            '<extra></extra>'
        ),
    ), row=2, col=1)

    # mu and sigma band lines (panel 2)
    pnav_max = df['pnav_smooth'].max()
    band_lines = [
        (pnav_mu + 2*pnav_std, '+2s', GREEN_LIT, 'dot'),
        (pnav_mu + 1*pnav_std, '+1s', GREEN_LIT, 'dot'),
        (pnav_mu,              ' mu', MIST,       'dash'),
        (pnav_mu - 1*pnav_std, '-1s', RED_LIT,   'dot'),
        (1.0,                  '1x',  STEEL,      'dot'),
    ]
    for thresh, label, color, dash in band_lines:
        if 0 < thresh < pnav_max * 1.15:
            fig.add_shape(
                type='line',
                x0=df.index[0], x1=df.index[-1],
                y0=thresh, y1=thresh,
                line=dict(color=color, width=0.7, dash=dash),
                row=2, col=1
            )
            fig.add_annotation(
                x=df.index[-1], y=thresh,
                text=f'  {label}', showarrow=False, xanchor='left',
                font=dict(family='Courier New, monospace', size=8, color=color),
                row=2, col=1
            )

    # Current value annotation (panel 2)
    c_now = GREEN_LIT if zscore_now > 0.5 else (RED_LIT if zscore_now < -0.5 else MIST)
    fig.add_annotation(
        x=df.index[-1], y=pnav_now,
        text=f'  {pnav_now:.2f}x ({zscore_now:+.1f}s)',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=9, color=c_now),
        row=2, col=1
    )

    # ── Layout ──────────────────────────────────────────────
    cost_note = '' if has_cost else '  (cost unavailable)'
    title = (
        f'MSTR P/NAV — Z-Score Heatmap + BTC Production Cost{cost_note}  |  '
        f'{regime_now}  ·  {pnav_now:.2f}x  ·  {zscore_now:+.1f}s'
    )
    alphaline_layout(fig, title, height=CHART_HEIGHT)
    fig.update_layout(
        bargap=0,
        showlegend=True,
        legend=dict(
            orientation='h', x=0.5, y=1.02, xanchor='center', yanchor='bottom',
            font=dict(size=9, color=MIST),
            bgcolor='rgba(10,22,40,0.85)', bordercolor=STEEL, borderwidth=1
        )
    )
    fig.update_yaxes(type='log', tickprefix='$', title_text='BTC Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='P/NAV (x)', ticksuffix='x', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building MSTR P/NAV Z-Score Heatmap + Production Cost ===')
    df, cost_series, has_cost, pnav_mu, pnav_std = build_dataframe()
    fig = plot_heatmap_chart(df, cost_series, has_cost, pnav_mu, pnav_std)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
