"""
generate_treasury_preferreds.py
Alphaline Research — Treasury Preferreds vs. Par

Variable-rate preferreds issued by crypto treasury companies, each managed to trade near
its $100 stated amount by resetting its dividend rate:
  STRC — Strategy (BTC)   BMNP — BitMine (ETH)   SATA — Strive (BTC)
A discount to $100 means buyers are demanding more than the current rate. STRC under $97
has come late in each selloff since 2025, with its low within days of BTC's
(research: research/prefs/).

Source: Yahoo Finance daily closes and dividends. Writes docs/treasury_preferreds.html.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from alphaline_theme import apply_theme, write_chart, date_long, GOLD, ETH_BLUE, TEXT_2

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

OUTPUT_PATH = os.path.join('docs', 'treasury_preferreds.html')
PREFS = {                     # ticker: (issuer, colour)
    'STRC': ('Strategy', GOLD),
    'BMNP': ('BitMine', ETH_BLUE),
    'SATA': ('Strive', '#B48EE0'),
}
PAR = 100.0
STRC_LATE = 97.0
LATE_FILL = 'rgba(232,106,106,0.13)'
SOURCE = 'Yahoo Finance'


def fetch():
    px, divs = {}, {}
    for t in list(PREFS) + ['BTC-USD']:
        h = yf.Ticker(t).history(period='max', auto_adjust=False)
        h.index = h.index.tz_localize(None).normalize()
        px[t] = h['Close']
        divs[t] = h['Dividends'][h['Dividends'] > 0]
        print(f'  {t:8} {len(h):5} rows → {h.index[-1].date()}')
    days = px['STRC'].index                                   # US trading days since STRC listed
    df = pd.DataFrame({t: px[t].reindex(days) for t in PREFS})
    df['BTC'] = px['BTC-USD'].reindex(days, method='ffill')
    return df, divs


def annual_rate(d, index):
    """Dividends paid over the trailing 91 days × 4, per $100 — works for monthly,
    twice-monthly, weekly and daily payers alike."""
    paid = d.reindex(index.union(d.index)).fillna(0).sort_index()
    trailing = paid.rolling('91D').sum() * (365 / 91)
    first = d.index[0] + pd.Timedelta(days=91) if len(d) else index[-1]
    return trailing.reindex(index).where(index >= first)


def to_par(price, rate, years=1.0):
    """Total return over `years` if the price is back at $100 by then and the rate is
    unchanged: dividends plus the gain to par, over today's price. Since 2025 this
    tracked STRC's actual 3-month return closely when bought below $97 (research/prefs/ytp.py)."""
    return (rate * years + (PAR - price)) / price


def runs(mask):
    out, start, prev = [], None, None
    for d, v in mask.fillna(False).items():
        if v and start is None:
            start = d
        if not v and start is not None:
            out.append((start, prev)); start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def plot(df, rates):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.58, 0.42], vertical_spacing=0.05)
    pad = pd.Timedelta(hours=12)
    for a, z in runs(df['STRC'] < STRC_LATE):
        for r in (1, 2):
            fig.add_vrect(x0=a - pad, x1=z + pad, fillcolor=LATE_FILL, line_width=0, row=r, col=1)

    for t, (issuer, color) in PREFS.items():
        s = df[t]
        y = (rates[t] / s * 100)
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=f'{t} ({issuer})', mode='lines',
            line=dict(color=color, width=1.8 if t == 'STRC' else 1.4),
            customdata=pd.concat([rates[t], y, to_par(s, rates[t]) * 100], axis=1).values,
            hovertemplate=(f'%{{x|%Y-%m-%d}}<br>{t} $%{{y:.2f}} · rate %{{customdata[0]:.1f}}%'
                           ' · yield %{customdata[1]:.1f}% · 1-yr if back at $100 %{customdata[2]:.1f}%'
                           '<extra></extra>'),
        ), row=1, col=1)
    fig.add_hline(y=PAR, line_color=TEXT_2, line_width=0.8, line_dash='dash', opacity=0.6, row=1, col=1)
    fig.add_hline(y=STRC_LATE, line_color=GOLD, line_width=0.8, line_dash='dot', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['BTC'], name='BTC', mode='lines', line=dict(color='#F8FAFB', width=1.4),
        hovertemplate='%{x|%Y-%m-%d}<br>BTC $%{y:,.0f}<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=f'STRC under ${STRC_LATE:.0f}',
                             marker=dict(symbol='square', size=10, color='rgba(232,106,106,0.6)')), row=1, col=1)

    lo = min(85.0, (df[list(PREFS)].min().min() // 5) * 5)
    ticks = sorted({lo, 90.0, STRC_LATE, PAR})
    fig.update_yaxes(title_text='Price vs. $100 par', range=[lo, 102], tickmode='array', tickvals=ticks,
                     ticktext=[f'<span style="color:{GOLD}">${v:.0f}</span>' if v == STRC_LATE else f'${v:.0f}'
                               for v in ticks], row=1, col=1)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='BTC (log)', row=2, col=1)
    apply_theme(fig)
    return fig


if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building Treasury Preferreds vs. Par ===')
    df, divs = fetch()
    rates = {t: annual_rate(divs[t], df.index) for t in PREFS}
    last = df.index[-1]

    prices, returns = [], []
    for t in PREFS:
        s, r = df[t].dropna(), rates[t].dropna()
        txt = f'{t} ${s.iloc[-1]:.2f}'
        prices.append(f'<b>{txt}</b>' if t == 'STRC' else txt)
        if len(r):
            returns.append(f'{t} {to_par(s.iloc[-1], r.iloc[-1]) * 100:.1f}%')
    flag = (f' · STRC under ${STRC_LATE:.0f} since {date_long(runs(df["STRC"] < STRC_LATE)[-1][0])}'
            if df['STRC'].iloc[-1] < STRC_LATE else '')
    subtitle = (' · '.join(prices) + flag + ' · 1-year return if back at $100: ' + ', '.join(returns)
                + f' · BTC ${df["BTC"].iloc[-1]:,.0f} · {date_long(last)}')
    print('  ', subtitle)

    write_chart(plot(df, rates), OUTPUT_PATH,
                title='Treasury Preferreds vs. Par',
                subtitle=subtitle,
                note=('Variable-rate preferreds managed to a $100 stated amount: STRC (Strategy), BMNP (BitMine), '
                      'SATA (Strive). Rate = last 3 months of dividends, annualised. 1-year return assumes the rate '
                      'stays the same and the price is back at $100; neither is guaranteed'),
                source=SOURCE, default='All', end=last, min_plot_height=440)
    print(f'Saved: {OUTPUT_PATH}')
