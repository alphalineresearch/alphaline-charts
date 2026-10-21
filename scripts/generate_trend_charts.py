"""
generate_trend_charts.py
Alphaline Research — BTC & ETH Trend: moving averages, volatility-adjusted Mayer, trend score

Panels:
  1. Price (log), coloured by trend state (green 3–4/4, grey 2/4, red 0–1/4), with the
     200-day and 200-week moving averages. Red shading where the Mayer z is ≥ +1.5σ
     (stretched), green where it is ≤ −1.5σ (washed out).
  2. Mayer z: log(price ÷ 200-day MA) divided by its own standard deviation over the past
     4 years. The plain Mayer Multiple shrinks every cycle as volatility falls (peaks 8.2×
     in 2013, 1.85× in 2024), so fixed levels stop firing; the z keeps the same scale.

Trend score: how many of the 1, 3, 6 and 12-month returns are positive (0–4); shown by
the price line's colour and in its hover.

Research behind the zones and the score: research/trend/trend.py.

Data sources (no API keys required):
  - blockchain.info: BTC daily price before Yahoo's history starts (Aug 2010 →)
  - Kraken: ETH daily closes Aug 2015 – May 2016 (stored in data/eth_early_daily.csv)
  - Coinbase: ETH daily price before Yahoo's history starts (May 2016 →)
  - Yahoo Finance: BTC-USD and ETH-USD daily (via yfinance)

Writes:
  docs/btc_trend.html
  docs/eth_trend.html

Usage:
    python scripts/generate_trend_charts.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots

from alphaline_theme import (apply_theme, write_chart, hover, date_long, pct, sigma, NAVY_MID, STEEL,
                             GOLD, WHITE, TEXT_2, TEXT_3, GREEN, RED, ETH_BLUE)

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_BTC = os.path.join('docs', 'btc_trend.html')
OUTPUT_ETH = os.path.join('docs', 'eth_trend.html')
ETH_EARLY  = os.path.join('data', 'eth_early_daily.csv')      # Kraken, Aug 2015 – May 2016

MA_DAYS      = 200
MA_WEEK_DAYS = 1400          # 200 weeks
VOTE_MONTHS  = (1, 3, 6, 12)
Z_WINDOW     = 4 * 365       # days of history the Mayer z is scaled against
Z_MIN_DAYS   = 2 * 365       # history needed before it is drawn
Z_HOT        = 1.5           # ≥ +1.5σ: stretched
Z_COLD       = -1.5          # ≤ −1.5σ: washed out

MA200_COLOR  = WHITE
MA200W_COLOR = GOLD          # green is the uptrend colour of the price line
SCORE_WORDS  = ['Downtrend', 'Weak', 'Mixed', 'Firm', 'Uptrend']
# price line: three states so the colour doesn't flicker with every one-step change
LINE_STATES  = [('Downtrend (0–1/4)', RED), ('Mixed (2/4)', TEXT_3), ('Uptrend (3–4/4)', GREEN)]


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def fetch_yahoo(ticker):
    df = yf.download(ticker, period='max', interval='1d', progress=False, auto_adjust=True)
    s = df['Close'].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna()


def fetch_btc():
    """Yahoo from Sep 2014; blockchain.info before that, for the 2011–2013 cycles."""
    yahoo = fetch_yahoo('BTC-USD')
    try:
        r = requests.get('https://api.blockchain.info/charts/market-price'
                         '?timespan=all&format=json&sampled=false', timeout=60).json()
        v = pd.DataFrame(r['values'])
        early = pd.Series(v['y'].values, index=pd.to_datetime(v['x'], unit='s').dt.normalize())
        early = early[(early > 0) & (early.index < yahoo.index[0])]
        return pd.concat([early, yahoo])
    except Exception as e:  # noqa: BLE001
        print(f'  ! blockchain.info history unavailable ({e}); Yahoo only')
        return yahoo


def fetch_eth():
    """Yahoo from Nov 2017; Coinbase daily closes before that (ETH-USD listed May 2016);
    before that Kraken's first year, stored once in data/ (research/trend/fetch_eth_early.py)."""
    yahoo = fetch_yahoo('ETH-USD')
    kraken = pd.Series(dtype=float)
    if os.path.exists(ETH_EARLY):
        kraken = pd.read_csv(ETH_EARLY, index_col='date', parse_dates=True)['close']
    try:
        rows, start = [], pd.Timestamp('2016-05-18')
        while start < yahoo.index[0]:
            end = min(start + pd.Timedelta(days=299), yahoo.index[0])
            r = requests.get('https://api.exchange.coinbase.com/products/ETH-USD/candles',
                             params=dict(granularity=86400, start=start.isoformat(), end=end.isoformat()),
                             headers={'User-Agent': 'alphaline-charts'}, timeout=60)
            r.raise_for_status()
            rows += r.json()                      # [time, low, high, open, close, volume]
            start = end + pd.Timedelta(days=1)
        early = pd.Series({pd.Timestamp(t, unit='s').normalize(): c for t, _, _, _, c, _ in rows}).sort_index()
        early = early[early.index < yahoo.index[0]]
        return pd.concat([kraken[kraken.index < early.index[0]], early, yahoo])
    except Exception as e:  # noqa: BLE001
        print(f'  ! Coinbase history unavailable ({e}); Yahoo only')
        return yahoo


def build(px):
    """Daily frame: price, MAs, Mayer Multiple, trend score (calendar days: crypto trades daily)."""
    px = px[~px.index.duplicated(keep='last')].asfreq('D').ffill()
    d = pd.DataFrame({'close': px})
    d['ma200'] = px.rolling(MA_DAYS, min_periods=MA_DAYS).mean()
    d['ma200w'] = px.rolling(MA_WEEK_DAYS, min_periods=MA_WEEK_DAYS).mean()
    d['mayer'] = px / d['ma200']
    dist = np.log(d['mayer'])
    d['mayer_z'] = dist / dist.rolling(Z_WINDOW, min_periods=Z_MIN_DAYS).std()
    for m in VOTE_MONTHS:
        d[f'r{m}'] = px / px.shift(int(round(m * 365.25 / 12))) - 1
    votes = pd.concat([(d[f'r{m}'] > 0).where(d[f'r{m}'].notna()) for m in VOTE_MONTHS], axis=1)
    d['score'] = votes.sum(axis=1).where(votes.notna().all(axis=1))
    return d.dropna(subset=['ma200'])


def spans(mask):
    """(start, end) of each run of True days."""
    m = mask.fillna(False).astype(bool)
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    return [(g.index[0], g.index[-1] + pd.Timedelta(days=1)) for _, g in m.groupby(grp) if g.iloc[0]]


# ════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════
def plot_trend(d, ticker, price_color):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)

    # ── Panel 1: shading, price, MAs ──
    for mask, color in ((d['mayer_z'] >= Z_HOT, RED), (d['mayer_z'] <= Z_COLD, GREEN)):
        for x0, x1 in spans(mask):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=hex_to_rgba(color, 0.14), line_width=0,
                          layer='below', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d['ma200w'], mode='lines', name='200-week MA',
        line=dict(color=MA200W_COLOR, width=1.4),
        hovertemplate=hover('200-week MA', '$%{y:,.0f}')), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=d['ma200'], mode='lines', name='200-day MA',
        line=dict(color=MA200_COLOR, width=1.2, dash='dot'), opacity=0.75,
        hovertemplate=hover('200-day MA', '$%{y:,.0f}')), row=1, col=1)
    # price coloured by trend state: one trace per state, each also holding the next
    # day's point so the segments join up
    state = pd.cut(d['score'], [-1, 1, 2, 4], labels=False)          # 0 down · 1 mixed · 2 up
    score = d['score'].fillna(-1).astype(int)
    trend = np.column_stack([score.where(score >= 0, 0),
                             [SCORE_WORDS[v] if v >= 0 else '' for v in score]])
    for k, (name, color) in enumerate(LINE_STATES):
        on = (state == k) | (state.shift(1) == k)
        fig.add_trace(go.Scatter(
            x=d.index, y=d['close'].where(on), mode='lines', name=name,
            line=dict(color=color, width=1.8), connectgaps=False, customdata=trend,
            hovertemplate=hover(ticker, '$%{y:,.0f}', ' · trend %{customdata[0]}/4 %{customdata[1]}')),
            row=1, col=1)
    pre = d['score'].isna()                                          # first year: no score yet
    if pre.any():
        fig.add_trace(go.Scatter(
            x=d.index, y=d['close'].where(pre | pre.shift(1, fill_value=False)), mode='lines',
            name=f'{ticker} Price', showlegend=False, line=dict(color=price_color, width=1.6),
            hovertemplate=hover(ticker, '$%{y:,.0f}')), row=1, col=1)
    # legend keys for the shading
    for name, color in ((f'Mayer z ≥ +{Z_HOT}σ (stretched)', RED),
                        (f'Mayer z ≤ −{-Z_COLD}σ (washed out)', GREEN)):
        fig.add_trace(go.Scatter(
            x=[d.index[-1]], y=[None], mode='lines', name=name, fill='toself',
            fillcolor=hex_to_rgba(color, 0.35), line=dict(width=0), hoverinfo='skip'), row=1, col=1)

    # ── Panel 2: Mayer z ──
    for y, color, a in ((Z_HOT, RED, 0.7), (0.5, TEXT_3, 0.3), (0, TEXT_3, 0.5),
                        (-0.5, TEXT_3, 0.3), (Z_COLD, GREEN, 0.7)):
        fig.add_hline(y=y, line_dash='dot', line_width=0.8, line_color=hex_to_rgba(color, a),
                      row=2, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=d['mayer_z'], mode='lines', name='Mayer z', customdata=d['mayer'],
        line=dict(color=GOLD, width=1.2),
        hovertemplate=hover('Mayer', '%{customdata:.2f}× · %{y:+.2f}σ')), row=2, col=1)

    fig.add_annotation(**checks_box(d))
    apply_theme(fig)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='Mayer z', tickmode='array',
                     tickvals=[-3, Z_COLD, 0, Z_HOT, 3], ticktext=['−3σ', '−1.5σ', '0', '+1.5σ', '+3σ'],
                     row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def checks_box(d):
    """Today's four trend checks, bottom-right of the price panel (clear of the line in the
    default window: recent prices sit above the lows of the years before)."""
    last = d.iloc[-1]
    score = int(last['score'])
    color = LINE_STATES[0 if score <= 1 else 1 if score == 2 else 2][1]
    cells = []
    for m in VOTE_MONTHS:
        r = last[f'r{m}']
        c = GREEN if r > 0 else RED
        cells.append(f'{m}m <span style="color:{c}">{"✓" if r > 0 else "✗"} {pct(r * 100, 0)}</span>')
    text = (f'<b>Trend today</b> · <span style="color:{color}"><b>{score}/4 {SCORE_WORDS[score]}</b></span><br>'
            'Higher than ' + ' · '.join(cells))
    return dict(text=text, xref='x domain', yref='y domain', x=0.99, y=0.03,
                xanchor='right', yanchor='bottom', align='left', showarrow=False,
                font=dict(size=12, color=TEXT_2), bgcolor=hex_to_rgba(NAVY_MID, 0.85),
                bordercolor=STEEL, borderwidth=1, borderpad=6)


def subtitle(d, ticker):
    last = d.iloc[-1]
    score = int(last['score'])
    parts = [f'<b>{ticker} ${last["close"]:,.0f}</b>',
             f'Mayer {last["mayer"]:.2f}× ({sigma(last["mayer_z"], 1)})',
             f'Trend {score}/4 · {SCORE_WORDS[score]}']
    if pd.notna(last['ma200w']):
        parts.insert(2, f'{last["close"] / last["ma200w"]:.2f}× the 200-week MA')
    parts.append(date_long(d.index[-1]))
    return '  ·  '.join(parts)


NOTE = ('Line colour = trend: how many of the 1, 3, 6 and 12-month returns are positive '
        '(green 3–4, grey 2, red 0–1). Mayer z = distance from the 200-day MA in units of its usual '
        f'range over the past 4 years; shading red ≥ +{Z_HOT}σ (stretched), green ≤ −{-Z_COLD}σ (washed out)')

# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building BTC & ETH Trend Charts ===')

    for ticker, fetch, color, out, src in (
            ('BTC', fetch_btc, GOLD, OUTPUT_BTC, 'blockchain.info · Yahoo Finance'),
            ('ETH', fetch_eth, ETH_BLUE, OUTPUT_ETH, 'Kraken · Coinbase · Yahoo Finance')):
        d = build(fetch())
        print(f'  {ticker}: {d.index[0].date()} → {d.index[-1].date()} | ${d["close"].iloc[-1]:,.0f} '
              f'| Mayer {d["mayer"].iloc[-1]:.2f} ({d["mayer_z"].iloc[-1]:+.2f}σ) | trend {d["score"].iloc[-1]:.0f}/4')
        fig = plot_trend(d, ticker, color)
        write_chart(fig, out, title=f'{ticker} Trend', subtitle=subtitle(d, ticker), note=NOTE,
                    source=src, default='5Y', min_plot_height=480)
        print(f'  Saved: {out}')

    print('\nDone.')
