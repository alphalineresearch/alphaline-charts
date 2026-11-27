"""
generate_btc_time_pain.py
Alphaline Research — BTC Max Pain for Time: the flat stretches before the vertical moves

Panels:
  1. BTC price (log) with the time-capitulation zones shaded: long sideways stretches
     after a crash where nothing happens and holders give up from boredom rather than
     losses. Each finished zone has a dotted line at its exit price, carried to today;
     the hover gives the share of days since then that BTC closed below it. Gold dots
     mark BTC's 20 best days (Yahoo era, Sep 2014 on).
  2. Time Pain Index: the share of the past 2 years on which BTC traded within ±35% of
     today's price. High = two years of going nowhere.

The zones are set by hand in ZONES (the last one is open-ended: it runs to today).

Data sources (no API keys required):
  - blockchain.info: BTC daily price before Yahoo's history starts (for the index's
    2-year lookback in 2014)
  - Yahoo Finance: BTC-USD daily (via yfinance)

Writes:
  docs/btc_time_pain.html

Usage:
    python scripts/generate_btc_time_pain.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alphaline_theme import (apply_theme, write_chart, hover, date_long, pct,
                             NAVY_MID, STEEL, GOLD, GOLD_LIT, WHITE, TEXT_2, TEXT_3, GREEN)
from generate_trend_charts import fetch_btc, hex_to_rgba

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT = os.path.join('docs', 'btc_time_pain.html')

# (start, end) — end None = still running
ZONES = [
    ('2014-10-01', '2016-02-29'),
    ('2018-02-01', '2020-10-31'),
    ('2022-05-01', '2023-12-31'),
    ('2025-11-15', None),
]

PAIN_WINDOW = 730            # days looked back
PAIN_BAND   = 0.35           # ±35% of today's price counts as "the same place"
PAIN_MA     = 90             # days: smoothing for the index's main line
BEST_N      = 20             # best days marked on the price
BEST_FROM   = '2014-09-17'   # Yahoo era: the best-days stats start here
CHART_FROM  = '2014-01-01'


# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def time_pain(px):
    """Share of the past PAIN_WINDOW days with a close within ±PAIN_BAND of today's."""
    lp = np.log(px.values)
    band = np.log(1 + PAIN_BAND)
    out = np.full(len(lp), np.nan)
    for i in range(PAIN_WINDOW, len(lp)):
        out[i] = np.mean(np.abs(lp[i - PAIN_WINDOW:i] - lp[i]) < band)
    return pd.Series(out, px.index)


def zone_stats(px):
    rows = []
    for a, b in ZONES:
        z = px[a:b]
        r = dict(start=z.index[0], end=z.index[-1], open=b is None, days=len(z),
                 low=z.min(), high=z.max(), exit=z.iloc[-1])
        if b is not None:
            after = px[r['end'] + pd.Timedelta(days=1):]
            fwd = px[r['end']:r['end'] + pd.Timedelta(days=365)]
            r['fwd12'] = fwd.iloc[-1] / r['exit'] - 1
            r['below'] = (after < r['exit']).mean()
        rows.append(r)
    return rows


def best_days(px):
    ret = px.pct_change()[BEST_FROM:].dropna()
    top = ret.nlargest(BEST_N)
    yrs = len(ret) / 365.25
    growth = {}
    for n in (0, 10, 20):
        rr = ret.copy()
        if n:
            rr[ret.nlargest(n).index] = 0
        growth[n] = (1 + rr).prod()
    in_zone = sum(any(pd.Timestamp(a) <= d <= (pd.Timestamp(b) if b else px.index[-1])
                      for a, b in ZONES) for d in top.index)
    return top, growth, yrs, in_zone


# ════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════
def plot(px, pain, zones, top):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.70, 0.30], vertical_spacing=0.03)
    view = px[CHART_FROM:]
    last = px.index[-1]

    # ── zone shading (both panels) + labels ──
    for z in zones:
        x1 = z['end'] + pd.Timedelta(days=1)
        fill = hex_to_rgba(GREEN, 0.20 if z['open'] else 0.13)
        for row in (1, 2):
            fig.add_vrect(x0=z['start'], x1=x1, fillcolor=fill, line_width=0, layer='below',
                          row=row, col=1)
        mid = z['start'] + (z['end'] - z['start']) / 2
        if z['open']:
            text = f'<b>Day {z["days"]}</b><br>still flat'
        else:
            text = f'<b>{z["days"]} days</b><br>then {pct(z["fwd12"] * 100, 0)} in 12m'
        fig.add_annotation(x=mid, xref='x', yref='y domain', y=0.98, yanchor='top',
                           text=text, showarrow=False, align='center',
                           font=dict(size=12, color=WHITE if z['open'] else TEXT_2))

    # ── exit levels, carried to today ──
    for i, z in enumerate(zz for zz in zones if not zz['open']):
        fig.add_trace(go.Scatter(
            x=[z['end'], last], y=[z['exit'], z['exit']], mode='lines',
            name='Zone exit price', legendgroup='exit', showlegend=i == 0,
            line=dict(color=WHITE, width=1, dash='dot'), opacity=0.6,
            hovertemplate=(f'Exit {date_long(z["end"])} · ${z["exit"]:,.0f} · closed below on '
                           f'{z["below"]:.1%} of days since<extra></extra>')), row=1, col=1)

    # ── price ──
    fig.add_trace(go.Scatter(
        x=view.index, y=view.values, mode='lines', name='BTC Price',
        line=dict(color=GOLD, width=1.6), hovertemplate=hover('BTC', '$%{y:,.0f}')), row=1, col=1)

    # ── best days ──
    rank = np.arange(1, len(top) + 1)
    fig.add_trace(go.Scatter(
        x=top.index, y=px.reindex(top.index).values, mode='markers',
        name=f'{BEST_N} best days', customdata=np.column_stack([rank, top.values * 100]),
        marker=dict(color=GOLD_LIT, size=8, line=dict(color=NAVY_MID, width=1.5)),
        hovertemplate=hover('Best day #%{customdata[0]}', '+%{customdata[1]:.1f}%',
                            ' · BTC $%{y:,.0f}')), row=1, col=1)

    # legend key for the shading
    fig.add_trace(go.Scatter(
        x=[last], y=[None], mode='lines', name='Time-capitulation zone', fill='toself',
        fillcolor=hex_to_rgba(GREEN, 0.35), line=dict(width=0), hoverinfo='skip'), row=1, col=1)

    # ── panel 2: Time Pain Index ──
    pv = pain[CHART_FROM:] * 100
    ma = pain.rolling(PAIN_MA).mean()[CHART_FROM:] * 100
    fig.add_trace(go.Scatter(
        x=pv.index, y=pv.values, mode='lines', name='Time Pain Index (daily)',
        line=dict(color=GOLD, width=0.9), opacity=0.35,
        hovertemplate=hover('Time Pain', '%{y:.0f}%',
                            f' of the past 2y within ±{PAIN_BAND:.0%} of today')), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ma.index, y=ma.values, mode='lines', name=f'Time Pain Index ({PAIN_MA}d avg)',
        line=dict(color=GOLD, width=2), fill='tozeroy', fillcolor=hex_to_rgba(GOLD, 0.08),
        hovertemplate=hover(f'Time Pain {PAIN_MA}d avg', '%{y:.0f}%')), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_width=0.8, line_color=hex_to_rgba(TEXT_3, 0.5),
                  row=2, col=1)

    apply_theme(fig)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='Time Pain', range=[0, 100], ticksuffix='%',
                     tickvals=[0, 25, 50, 75, 100], row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False,
                     range=[CHART_FROM, last + (last - pd.Timestamp(CHART_FROM)) * 0.02])
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building BTC Max Pain for Time ===')

    px = fetch_btc()
    px = px[~px.index.duplicated(keep='last')].asfreq('D').ffill()
    pain = time_pain(px)
    zones = zone_stats(px)
    top, growth, yrs, in_zone = best_days(px)

    for z in zones:
        extra = '' if z['open'] else f' | +{z["fwd12"]:.0%} 12m | below exit {z["below"]:.1%} of days since'
        print(f'  {z["start"].date()} → {z["end"].date()} | {z["days"]}d | '
              f'${z["low"]:,.0f}–${z["high"]:,.0f}{extra}')
    rec = pain[CHART_FROM:]
    ma = pain.rolling(PAIN_MA).mean()[CHART_FROM:]
    print(f'  Time Pain now {pain.iloc[-1]:.0%} | record {rec.max():.0%} on {rec.idxmax().date()}'
          f' | {PAIN_MA}d avg {ma.iloc[-1]:.0%}, record {ma.max():.0%} on {ma.idxmax().date()}')
    print(f'  Since {BEST_FROM}: {growth[0]:.0f}× | ex-10 best {growth[10]:.1f}× | '
          f'ex-20 best {growth[20]:.1f}× | {in_zone}/{BEST_N} best days inside zones')

    cur = zones[-1]
    sub = '  ·  '.join([
        f'<b>BTC ${px.iloc[-1]:,.0f}</b>',
        f'Time Pain {pain.iloc[-1]:.0%}',
        f'{PAIN_MA}d avg {ma.iloc[-1]:.0%}' + (' (record)' if ma.iloc[-1] >= ma.max() - 1e-9 else
                                               f' (record {ma.max():.0%}, {date_long(ma.idxmax())})'),
        f'Day {cur["days"]} of the current flat zone',
        date_long(px.index[-1]),
    ])
    note = (f'Time Pain = share of the past 2 years BTC spent within ±{PAIN_BAND:.0%} of today’s price (bold line: its {PAIN_MA}-day average). '
            'Shaded: the long sideways stretches after each crash; dotted lines carry each exit price '
            'forward — BTC has rarely closed back below one. '
            f'Since Sep 2014 BTC is up {growth[0]:.0f}×; miss its 10 best days and it is '
            f'{growth[10]:.0f}×, miss 20 and it is {growth[20]:.1f}×. '
            f'{in_zone} of those 20 days came inside the flat zones.')

    fig = plot(px, pain, zones, top)
    write_chart(fig, OUTPUT, title='BTC Max Pain for Time', subtitle=sub, note=note,
                source='blockchain.info · Yahoo Finance', default='All', min_plot_height=480)
    print(f'  Saved: {OUTPUT}')
