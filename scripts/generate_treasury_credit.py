"""
generate_treasury_credit.py
Alphaline Research — Treasury Credit Monitor

Strategy's preferred shares as a read on crypto stress (research: research/prefs/).

Flags
  Caution    — STRD (Strategy's most junior preferred) closes below its 50-day average
               while BTC is still above its own. STRD broke before BTC in 7 of 8 cases
               since 2025, and BTC was lower 60 days later after 11 of 12 STRD breaks.
  Late stage — STRC (variable-rate, managed to $100 par) closes under $97. STRC has been
               the last to crack in each selloff, with its low within days of BTC's.

Source: Yahoo Finance daily closes and dividends; 10-year Treasury yield (^TNX).
Writes docs/treasury_credit.html.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from alphaline_theme import apply_theme, write_chart, date_long, GOLD, WHITE, RED, TEXT_2

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

OUTPUT_PATH = os.path.join('docs', 'treasury_credit.html')
START       = '2025-02-01'
MA_DAYS     = 50
STRC_LATE   = 97.0
STRD_COUPON = 10.0
CAUTION_FILL = 'rgba(232,106,106,0.18)'
LATE_FILL    = 'rgba(212,168,67,0.20)'
SOURCE = 'Yahoo Finance · 10-year Treasury yield'


def fetch():
    closes = {}
    for t in ('STRC', 'STRD', 'BTC-USD', '^TNX'):
        h = yf.Ticker(t).history(period='max', auto_adjust=False)
        h.index = h.index.tz_localize(None).normalize()
        closes[t] = h['Close']
        print(f'  {t:8} {len(h):5} rows → {h.index[-1].date()}')
    days = closes['STRD'].loc[START:].index                     # US trading days
    df = pd.DataFrame({k: v.reindex(days).ffill(limit=3) for k, v in closes.items()})
    # BTC trades every day; take its close on each US trading day
    df['BTC-USD'] = closes['BTC-USD'].reindex(days, method='ffill')
    return df.dropna(subset=['STRD', 'BTC-USD'])


def runs(mask):
    """(start, end) dates of consecutive True days."""
    m = mask.fillna(False)
    out, start, prev = [], None, None
    for d, v in m.items():
        if v and start is None:
            start = d
        if not v and start is not None:
            out.append((start, prev)); start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def build(df):
    df = df.copy()
    df['strd_ma'] = df['STRD'].rolling(MA_DAYS).mean()
    df['btc_ma'] = df['BTC-USD'].rolling(MA_DAYS).mean()
    df['strd_spread'] = STRD_COUPON / df['STRD'] * 100 - df['^TNX']
    df['caution'] = (df['STRD'] < df['strd_ma']) & (df['BTC-USD'] > df['btc_ma'])
    df['late'] = df['STRC'] < STRC_LATE
    return df


def plot(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.42, 0.32, 0.26], vertical_spacing=0.04)
    pad = pd.Timedelta(hours=12)
    for col, fill in (('caution', CAUTION_FILL), ('late', LATE_FILL)):
        for a, z in runs(df[col]):
            for r in (1, 2, 3):
                fig.add_vrect(x0=a - pad, x1=z + pad, fillcolor=fill, line_width=0, row=r, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['BTC-USD'], name='BTC', mode='lines',
                             line=dict(color=GOLD, width=1.6),
                             hovertemplate='%{x|%Y-%m-%d}<br>BTC $%{y:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['btc_ma'], name='BTC 50-day avg', mode='lines',
                             line=dict(color=GOLD, width=0.9, dash='dot'), opacity=0.6,
                             hovertemplate='%{x|%Y-%m-%d}<br>50-day avg $%{y:,.0f}<extra></extra>'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['STRD'], name='STRD', mode='lines',
                             line=dict(color=RED, width=1.5), customdata=df['strd_spread'],
                             hovertemplate='%{x|%Y-%m-%d}<br>STRD $%{y:.2f} · %{customdata:.1f}% over the 10-year<extra></extra>'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['strd_ma'], name='STRD 50-day avg', mode='lines',
                             line=dict(color=RED, width=0.9, dash='dot'), opacity=0.7,
                             hovertemplate='%{x|%Y-%m-%d}<br>50-day avg $%{y:.2f}<extra></extra>'), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['STRC'], name='STRC', mode='lines',
                             line=dict(color=WHITE, width=1.5),
                             hovertemplate='%{x|%Y-%m-%d}<br>STRC $%{y:.2f}<extra></extra>'), row=3, col=1)
    fig.add_hline(y=100, line_color=TEXT_2, line_width=0.8, line_dash='dash', opacity=0.6, row=3, col=1)
    fig.add_hline(y=STRC_LATE, line_color=GOLD, line_width=0.8, line_dash='dot', row=3, col=1)

    # legend swatches for the two shadings
    for name, fill in (('Caution: STRD below its 50-day, BTC above its own', CAUTION_FILL.replace('0.18', '0.6')),
                       (f'Late stage: STRC under ${STRC_LATE:.0f}', LATE_FILL.replace('0.20', '0.7'))):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=name,
                                 marker=dict(symbol='square', size=10, color=fill)), row=1, col=1)

    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='BTC (log)', row=1, col=1)
    fig.update_yaxes(tickprefix='$', title_text='STRD', row=2, col=1)
    lo = min(80, np.floor(df['STRC'].min() / 5) * 5)
    fig.update_yaxes(tickprefix='$', title_text='STRC', range=[lo, 102], tickmode='array',
                     tickvals=sorted({lo, 90, STRC_LATE, 100}),
                     ticktext=[f'<span style="color:{GOLD}">${v:.0f}</span>' if v == STRC_LATE else f'${v:.0f}'
                               for v in sorted({lo, 90, STRC_LATE, 100})], row=3, col=1)
    apply_theme(fig)
    return fig


if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building Treasury Credit Monitor ===')
    df = build(fetch())
    last = df.index[-1]
    row = df.iloc[-1]

    flags = []
    if row['caution']:
        since = runs(df['caution'])[-1][0]
        flags.append(f'<b>Caution</b> since {date_long(since)}')
    if row['late']:
        since = runs(df['late'])[-1][0]
        flags.append(f'<b>Late stage</b> since {date_long(since)}')
    if not flags:
        flags.append('<b>No flags</b>')
    strd_gap = (row['STRD'] / row['strd_ma'] - 1) * 100
    btc_gap = (row['BTC-USD'] / row['btc_ma'] - 1) * 100
    side = lambda g: f'{abs(g):.1f}% {"below" if g < 0 else "above"} its 50-day'
    subtitle = (' · '.join(flags) +
                f' · STRD ${row["STRD"]:.2f}, {side(strd_gap)} ({row["strd_spread"]:.1f}% over the 10-year)'
                f' · BTC {side(btc_gap)} · STRC ${row["STRC"]:.2f} · {date_long(last)}')
    print('  ', subtitle)

    write_chart(plot(df), OUTPUT_PATH,
                title='Treasury Credit Monitor',
                subtitle=subtitle,
                note=('Strategy preferreds: STRD is the most junior (10%, non-cumulative); STRC is managed to $100 par. '
                      'Red = caution, gold = late stage'),
                source=SOURCE, default='All', end=last, min_plot_height=460)
    print(f'Saved: {OUTPUT_PATH}')
