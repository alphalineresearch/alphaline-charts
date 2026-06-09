"""
generate_btc_copper_gold_bb_squeeze.py
Alphaline Research — BTC Monthly BB Squeeze × Copper/Gold Ratio

Fetches BTC daily price from blockchain.info (resampled to monthly OHLC),
Copper (HG=F) and Gold (GC=F) monthly futures from Yahoo Finance.
No API keys required.
Writes btc_copper_gold_bb_squeeze.html to docs/.

Usage:
    python scripts/generate_btc_copper_gold_bb_squeeze.py
"""

import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
BB_PERIOD    = 20
BB_MULT      = 2.0
KC_PERIOD    = 20
KC_MULT      = 1.5
ENTRY_PCT    = 0.20   # bottom 20th pct = deep compression entry zone
BOTTOM_PCT   = 0.30   # bottom 30th pct for Cu/Au cycle low detection
ROLL_MIN_WIN = 9      # months for rolling minimum in cycle low detection
MIN_LOW_GAP  = 180    # days between cycle lows

OUTPUT_PATH = os.path.join('docs', 'btc_copper_gold_bb_squeeze.html')

# ════════════════════════════════════════════
# ALPHALINE BRAND COLORS
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
GREEN_LIT = '#2ABF7A'
RED_LIT   = '#D64444'
CYAN      = '#22D3EE'
ORANGE    = '#F97316'

CHART_WIDTH  = 1100
CHART_HEIGHT = 1080


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ════════════════════════════════════════════
# ALPHALINE CHART TEMPLATE
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT,
                     source='alphalineresearch.com  |  Yahoo Finance · blockchain.info'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, autosize=True,
        title=dict(
            text=(
                f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>'
                f'<br><span style="font-family:Courier New,monospace; font-size:8px; color:{GOLD};">ALPHALINE RESEARCH</span>'
            ),
            x=0.02, xanchor='left', y=0.985, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=60, r=40, t=80, b=120),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        legend=dict(
            orientation='h', yanchor='top', y=-0.08,
            xanchor='center', x=0.5,
            bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL,
            font=dict(size=9),
        ),
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
# BOLLINGER BAND SQUEEZE CALCULATION
# ════════════════════════════════════════════
def compute_squeeze(df, bb_period=BB_PERIOD, bb_mult=BB_MULT,
                    kc_period=KC_PERIOD, kc_mult=KC_MULT):
    px = df['close']; hi = df['high']; lo = df['low']

    bb_mid   = px.rolling(bb_period).mean()
    bb_std   = px.rolling(bb_period).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    prev_close = px.shift(1)
    tr = pd.concat(
        [(hi - lo), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1
    ).max(axis=1)
    atr      = tr.rolling(kc_period).mean()
    kc_mid   = px.ewm(span=kc_period, adjust=False).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    squeeze_on  = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    bb_width    = bb_upper - bb_lower
    kc_width    = kc_upper - kc_lower
    squeeze_pct = (bb_width - kc_width) / kc_width * 100

    return pd.DataFrame({
        'bb_mid': bb_mid, 'bb_upper': bb_upper, 'bb_lower': bb_lower,
        'kc_mid': kc_mid, 'kc_upper': kc_upper, 'kc_lower': kc_lower,
        'squeeze_on': squeeze_on, 'squeeze_pct': squeeze_pct,
    }, index=df.index)


# ════════════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════════════
def fetch_blockchain_chart(chart_name):
    url = (f'https://api.blockchain.info/charts/{chart_name}'
           f'?timespan=all&format=json&sampled=false')
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = [
        {'date': pd.Timestamp(int(pt['x']), unit='s').normalize(),
         'value': float(pt['y'])}
        for pt in resp.json().get('values', [])
    ]
    df = pd.DataFrame(rows).set_index('date').sort_index()
    return df[~df.index.duplicated(keep='last')]


def fetch_yf_monthly(ticker):
    raw = yf.download(ticker, period='max', interval='1mo',
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    avail = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in raw.columns]
    return raw[avail].dropna(subset=['close'])


# ════════════════════════════════════════════
# BUILD DATAFRAME
# ════════════════════════════════════════════
def build_dataframe():
    print('Fetching BTC daily price (blockchain.info)...')
    btc_daily = fetch_blockchain_chart('market-price')
    btc_daily.columns = ['close']
    btc_daily = btc_daily[btc_daily['close'] > 0]

    btc_m = btc_daily['close'].resample('MS').agg(
        open='first', high='max', low='min', close='last'
    ).dropna()
    print(f'  BTC monthly: {btc_m.index[0].date()} → {btc_m.index[-1].date()}  ({len(btc_m)} rows)')

    print('Fetching Copper monthly (HG=F)...')
    copper_m = fetch_yf_monthly('HG=F')
    print(f'  Copper: {copper_m.index[0].date()} → {copper_m.index[-1].date()}  ({len(copper_m)} rows)')

    print('Fetching Gold monthly (GC=F)...')
    gold_m = fetch_yf_monthly('GC=F')
    print(f'  Gold: {gold_m.index[0].date()} → {gold_m.index[-1].date()}  ({len(gold_m)} rows)')

    # Forward-fill sparse futures gaps, align to BTC monthly index
    copper_ff = copper_m['close'].reindex(btc_m.index).ffill()
    gold_ff   = gold_m['close'].reindex(btc_m.index).ffill()

    valid_idx = copper_ff.dropna().index.intersection(gold_ff.dropna().index)
    btc_a    = btc_m.loc[valid_idx].copy()
    copper_a = copper_ff.loc[valid_idx]
    gold_a   = gold_ff.loc[valid_idx]

    cu_au = (copper_a / gold_a) * 100   # scaled ×100 for readability
    print(f'Aligned range: {valid_idx[0].date()} → {valid_idx[-1].date()} ({len(valid_idx)} months)')

    btc_ind = compute_squeeze(btc_a)
    sq      = btc_ind['squeeze_pct']
    sq_on   = btc_ind['squeeze_on']

    entry_threshold = sq.quantile(ENTRY_PCT)
    print(f'BTC entry threshold (20th pct squeeze): {entry_threshold:.1f}%')

    cu_au_ma20 = cu_au.rolling(20).mean()

    # Detect Cu/Au cycle lows
    bottom_thresh = cu_au.quantile(BOTTOM_PCT)
    roll_min      = cu_au.rolling(ROLL_MIN_WIN, center=True, min_periods=4).min()
    is_local_min  = (cu_au == roll_min) & (cu_au <= bottom_thresh)

    cycle_low_dates = []
    last_added = pd.Timestamp('1900-01-01')
    for d in cu_au.index[is_local_min]:
        if (d - last_added).days >= MIN_LOW_GAP:
            cycle_low_dates.append(d)
            last_added = d
    cycle_low_dates = pd.DatetimeIndex(cycle_low_dates)
    print(f'Detected {len(cycle_low_dates)} Cu/Au cycle lows')

    # Detect Cu/Au MA breakout signals
    above_ma    = cu_au > cu_au_ma20
    first_cross = above_ma & ~above_ma.shift(1).fillna(True) & cu_au_ma20.notna()

    breakout_dates = []
    for low_d in cycle_low_dates:
        after = first_cross[first_cross.index > low_d]
        if len(after) and after.any():
            breakout_dates.append(after.index[after][0])
    breakout_dates = pd.DatetimeIndex(dict.fromkeys(breakout_dates))
    print(f'Detected {len(breakout_dates)} Cu/Au MA breakout signals')

    return btc_a, btc_ind, sq, sq_on, cu_au, cu_au_ma20, \
           cycle_low_dates, breakout_dates, entry_threshold


# ════════════════════════════════════════════
# CHART — BTC MONTHLY BB SQUEEZE × COPPER/GOLD
# ════════════════════════════════════════════
def plot_btc_copper_gold_bb_squeeze(btc_a, btc_ind, sq, sq_on, cu_au, cu_au_ma20,
                                    cycle_low_dates, breakout_dates, entry_threshold):
    close = btc_a['close']

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.44, 0.24, 0.32],
        vertical_spacing=0.035,
        subplot_titles=[
            'BTC/USD — Monthly Price + Bollinger Bands (20, 2σ) + Keltner Channels (20, 1.5×ATR)',
            'BTC Monthly BB Squeeze %  ·  cyan = deep compression (bottom 20th pct)',
            'Copper/Gold Ratio ×100  ·  cyan ▲ = cycle low  ·  gold ◆ = MA breakout signal',
        ]
    )

    # ── Row 1: BTC Price + Bands ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=close.index, y=btc_ind['kc_upper'],
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=close.index, y=btc_ind['kc_lower'],
        fill='tonexty', fillcolor=hex_to_rgba(STEEL, 0.18),
        line=dict(color=STEEL, width=0.6),
        name='Keltner Channel', hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=close.index, y=btc_ind['bb_upper'],
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=close.index, y=btc_ind['bb_lower'],
        fill='tonexty', fillcolor=hex_to_rgba(NAVY_LIT, 0.50),
        line=dict(color=GOLD_DIM, width=0.7, dash='dot'),
        name='Bollinger Bands', hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=close.index, y=btc_ind['bb_upper'],
        line=dict(color=GOLD_DIM, width=0.7, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=close.index, y=btc_ind['bb_mid'],
        line=dict(color=STEEL, width=0.8),
        name='BB Mid (20M SMA)', hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        line=dict(color=GOLD, width=2.2),
        name='BTC/USD (monthly close)',
        hovertemplate='%{x|%b %Y}<br>BTC: $%{y:,.0f}<extra></extra>',
    ), row=1, col=1)

    # Cycle low markers on BTC panel
    if len(cycle_low_dates):
        valid_cl = [d for d in cycle_low_dates if d in close.index]
        if valid_cl:
            cl_idx = pd.DatetimeIndex(valid_cl)
            fig.add_trace(go.Scatter(
                x=cl_idx, y=close[cl_idx].values,
                mode='markers',
                marker=dict(symbol='triangle-up', size=11, color=CYAN,
                            line=dict(color=WHITE, width=1)),
                name='Cu/Au Cycle Low',
                showlegend=False,
                hovertemplate='%{x|%b %Y}<br>BTC: $%{y:,.0f}<extra></extra>',
            ), row=1, col=1)

    # Breakout markers on BTC panel
    if len(breakout_dates):
        valid_bo = [d for d in breakout_dates if d in close.index]
        if valid_bo:
            bo_idx = pd.DatetimeIndex(valid_bo)
            fig.add_trace(go.Scatter(
                x=bo_idx, y=close[bo_idx].values,
                mode='markers',
                marker=dict(symbol='diamond', size=10, color=GOLD,
                            line=dict(color=WHITE, width=1)),
                name='Cu/Au MA Breakout',
                showlegend=False,
                hovertemplate='%{x|%b %Y}<br>BTC: $%{y:,.0f}<extra></extra>',
            ), row=1, col=1)

    # ── Row 2: Squeeze % bars ─────────────────────────────────────────────────
    def bar_color(sq_val, sq_on_val):
        if sq_val <= entry_threshold: return CYAN
        if sq_on_val:                 return RED_LIT
        return GREEN_LIT

    bar_colors  = [bar_color(s, z) for s, z in zip(sq.values, sq_on.values)]
    bar_opacity = [1.0 if (z or s <= entry_threshold) else 0.50
                   for s, z in zip(sq.values, sq_on.values)]

    fig.add_trace(go.Bar(
        x=sq.index, y=sq.values,
        marker_color=bar_colors,
        marker_opacity=bar_opacity,
        name='BB Squeeze % (BTC monthly)',
        hovertemplate='%{x|%b %Y}<br>Squeeze: %{y:.1f}%<extra></extra>',
    ), row=2, col=1)

    for y_val, color, dash in [
        (entry_threshold, CYAN,  'dash'),
        (0,               STEEL, 'solid'),
    ]:
        fig.add_shape(type='line',
            x0=close.index[0], x1=close.index[-1],
            y0=y_val, y1=y_val, xref='x', yref='y2',
            line=dict(color=color, width=0.9, dash=dash))

    fig.add_annotation(
        x=close.index[-1], y=entry_threshold, xref='x', yref='y2',
        text=f'  entry zone ({entry_threshold:.1f}%)', showarrow=False,
        font=dict(color=CYAN, size=9, family='Courier New, monospace'),
        xanchor='left',
    )

    # ── Row 3: Copper/Gold Ratio ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=cu_au.index, y=cu_au.values,
        fill='tozeroy', fillcolor=hex_to_rgba(ORANGE, 0.10),
        line=dict(color=ORANGE, width=1.8),
        name='Cu/Au ×100',
        hovertemplate='%{x|%b %Y}<br>Cu/Au: %{y:.4f}<extra></extra>',
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=cu_au_ma20.index, y=cu_au_ma20.values,
        line=dict(color=MIST, width=1.0, dash='dot'),
        name='Cu/Au 20M MA',
        hovertemplate='%{x|%b %Y}<br>MA: %{y:.4f}<extra></extra>',
    ), row=3, col=1)

    if len(cycle_low_dates):
        fig.add_trace(go.Scatter(
            x=cycle_low_dates,
            y=cu_au[cycle_low_dates].values,
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=11, color=CYAN,
                        line=dict(color=WHITE, width=1)),
            text=[d.strftime('%b %Y') for d in cycle_low_dates],
            textposition='bottom center',
            textfont=dict(color=CYAN, size=8, family='Courier New, monospace'),
            name='Cu/Au Cycle Low',
            hovertemplate='%{x|%b %Y}<br>Cu/Au Low: %{y:.4f}<extra></extra>',
        ), row=3, col=1)

    if len(breakout_dates):
        valid_bo = [d for d in breakout_dates if d in cu_au.index]
        if valid_bo:
            bo_idx = pd.DatetimeIndex(valid_bo)
            fig.add_trace(go.Scatter(
                x=bo_idx, y=cu_au[bo_idx].values,
                mode='markers',
                marker=dict(symbol='diamond', size=10, color=GOLD,
                            line=dict(color=WHITE, width=1)),
                name='Cu/Au MA Breakout',
                hovertemplate='%{x|%b %Y}<br>Cu/Au Breakout: %{y:.4f}<extra></extra>',
            ), row=3, col=1)

    # ── Cyan shaded bands across all panels during BTC squeeze periods ────────
    in_entry = (sq <= entry_threshold).reindex(btc_a.index, fill_value=False)
    squeeze_periods = []
    start = None
    for d, val in in_entry.items():
        if val and start is None:
            start = d
        elif not val and start is not None:
            squeeze_periods.append((start, d))
            start = None
    if start is not None:
        squeeze_periods.append((start, in_entry.index[-1]))

    for s, e in squeeze_periods:
        fig.add_shape(
            type='rect', x0=s, x1=e, y0=0, y1=1,
            xref='x', yref='paper',
            fillcolor=hex_to_rgba(CYAN, 0.07),
            line=dict(width=0), layer='below',
        )

    # ── Axis styling ──────────────────────────────────────────────────────────
    axis_style = dict(gridcolor=STEEL, gridwidth=0.3,
                      tickfont=dict(size=9), showgrid=True)

    fig.update_yaxes(row=1, col=1, type='log', zeroline=False,
        title_text='BTC Price (log)',
        title_font=dict(size=9, color=MIST), **axis_style)
    fig.update_yaxes(row=2, col=1, zeroline=True, zerolinecolor=STEEL,
        title_text='Squeeze %',
        title_font=dict(size=9, color=MIST), **axis_style)
    fig.update_yaxes(row=3, col=1, zeroline=False, rangemode='normal',
        range=[0.1, None],
        title_text='Cu/Au ×100',
        title_font=dict(size=9, color=MIST), **axis_style)
    fig.update_xaxes(gridcolor=STEEL, gridwidth=0.3, tickfont=dict(size=9))

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font = dict(family='Courier New, monospace', size=10, color=MIST)
        ann.x = 0.01
        ann.xanchor = 'left'

    title_str = (
        'BTC Monthly BB Squeeze × Copper/Gold Ratio — Macro Cycle Convergence'
        f'<br><span style="font-family:Courier New,monospace; font-size:9px; color:{MIST};">'
        'Cyan bands = BTC BB deep compression (bottom 20th pct)  ·  '
        'cyan ▲ = Cu/Au cycle low  ·  gold ◆ = Cu/Au MA breakout</span>'
    )

    alphaline_layout(fig, title_str, height=CHART_HEIGHT)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building BTC Monthly BB Squeeze × Copper/Gold Ratio chart ===')
    btc_a, btc_ind, sq, sq_on, cu_au, cu_au_ma20, \
        cycle_low_dates, breakout_dates, entry_threshold = build_dataframe()

    fig = plot_btc_copper_gold_bb_squeeze(
        btc_a, btc_ind, sq, sq_on, cu_au, cu_au_ma20,
        cycle_low_dates, breakout_dates, entry_threshold
    )

    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
