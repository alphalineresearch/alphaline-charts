"""
generate_btc_etf_flows.py
Alphaline Research — BTC Spot ETF Flows: are investors moving into or out of bitcoin?

Modelled on the "price + 13-week fund flows + breadth shading" layout.

Panels:
  1. BTC weekly candles (log) with swing high / low labels.
  2. 13-week rolling net flows into US spot bitcoin ETFs, $B, stacked IBIT vs. all
     other ETFs, with the total as a white line. Net flow = creations − redemptions:
     the change in BTC the funds hold, valued at that day's BTC price.

Price-panel shading: colour = sign of the 13-week net flow (green inflows, red
outflows). Depth = 60% size of the flow (full green at +$12B, full red at −$6B) +
40% breadth agreeing with it (share of days in the same direction, 4-week average).

Header also reports flow breadth: the share of the past 20 trading days with net inflows.

Data sources (no API keys required):
  - bitbo.io (BitcoinTreasuries) ETF flow page: daily net BTC flows, total and by issuer
  - Yahoo Finance: BTC-USD daily OHLC (via yfinance)

Writes:
  docs/btc_etf_flows.html

Usage:
    python scripts/generate_btc_etf_flows.py
"""

import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots

from alphaline_theme import (apply_theme, write_chart, date_long, NAVY_MID, STEEL,
                             GOLD, GOLD_DIM, WHITE, TEXT_2, TEXT_3, GREEN, RED, RED_DEEP)

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_PATH = os.path.join('docs', 'btc_etf_flows.html')

FLOWS_URL    = 'https://bitbo.io/treasuries/etf-flows/'
FIT_START    = '2023-12-01'      # price history pulled; the chart starts at the ETF launch
ROLL_WEEKS   = 13
CB_MIN_BTC   = 50_000           # cost basis drawn once the ETFs hold this much net BTC
BREADTH_DAYS = 20                # trading days in the breadth window
SHADE_FULL_IN   = 12.0           # $B of 13-wk inflows for full-strength green
SHADE_FULL_OUT  = 6.0            # $B of 13-wk outflows for full-strength red
SHADE_BASE_ALPHA = 0.07          # every week: regime tint
SHADE_MAX_ALPHA  = 0.25          # added on top at full depth
SHADE_SMOOTH_WK  = 4             # weeks the breadth is averaged over for the depth
PIVOT_SWING  = 0.25              # 25% move between labelled highs and lows

IBIT_COLOR   = GOLD
OTHER_COLOR  = '#6F8FB8'         # muted steel-blue: the rest of the field
NET_COLOR    = WHITE

# Per-asset settings; generate_eth_etf_flows.py passes its own (ETH_CFG).
BTC_CFG = dict(
    asset='BTC', unit='BTC', lead='IBIT (BlackRock)', cb_min=CB_MIN_BTC,
    cb_method='net',                 # 'net': net dollars in ÷ net coins; 'avg': average cost
    swing=PIVOT_SWING,
    full_in=SHADE_FULL_IN, full_out=SHADE_FULL_OUT,
    ticks=[10_000, 15_000, 20_000, 30_000, 50_000, 70_000, 100_000, 150_000, 200_000, 300_000],
)

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/130.0 Safari/537.36')


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


def usd_b(v, d=1):
    """$4.2B / ($3.1B) — finance-style negatives."""
    return f'(${abs(v):,.{d}f}B)' if v < 0 else f'${v:,.{d}f}B'


def units_k(v, unit='BTC'):
    return f'({abs(v)/1e3:,.1f}k {unit})' if v < 0 else f'+{v/1e3:,.1f}k {unit}'


def price_label(v):
    """$126k for BTC-sized prices, $4,950 for ETH-sized ones."""
    return f'${v/1e3:,.0f}k' if v >= 1e4 else f'${v:,.0f}'


# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def _get(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, headers={'User-Agent': UA}, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if i == tries - 1:
                raise
            print(f'  retry {i+1}: {e}')
            time.sleep(3 * (i + 1))


_ENTRY = re.compile(r'getPreviousBusinessDay\((\d+)\),\s*truncate\((-?[\d.]+)')
_ENDS  = ('const merged', 'aum_etf_', 'function ')


def _prev_business_day(ts):
    """The page dates each row by when the holding change was reported; the flow
    happened the business day before (same rule as the page's own JS)."""
    dow = ts.dayofweek                       # Mon=0 … Sun=6
    return ts - pd.Timedelta(days=2 if dow == 6 else 3 if dow == 0 else 1)


def _series(page, start_marker):
    i = page.index(start_marker)
    ends = [k for k in (page.find(m, i + len(start_marker)) for m in _ENDS) if k > 0]
    rows = _ENTRY.findall(page[i:min(ends)])
    if not rows:
        return pd.Series(dtype=float)
    d = pd.DataFrame(rows, columns=['ms', 'btc']).astype(float)
    d['date'] = pd.to_datetime(d['ms'], unit='ms').map(_prev_business_day).dt.normalize()
    return d.groupby('date')['btc'].sum()


def fetch_flows():
    """Daily net flows in BTC: total of all US spot ETFs and BlackRock's IBIT."""
    print('  Fetching ETF flows (bitbo.io)...')
    page = _get(FLOWS_URL)
    total = _series(page, 'const historyBtc')
    ibit  = _series(page, 'aum_etf_blackrock')
    if len(total) < 250:
        raise RuntimeError(f'only {len(total)} days of ETF flows parsed — page layout changed?')
    df = pd.DataFrame({'total': total, 'lead': ibit}).fillna(0.0).sort_index()
    df['other'] = df['total'] - df['lead']
    print(f'    {len(df)} days, {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}, '
          f'net {df["total"].sum():,.0f} BTC')
    return df


def fetch_price(ticker='BTC-USD', start=FIT_START):
    print(f'  Fetching {ticker} (Yahoo)...')
    px = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[['Open', 'High', 'Low', 'Close']].dropna()
    px.columns = ['open', 'high', 'low', 'close']
    return px


# ════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════
def pivots(wk, swing=PIVOT_SWING):
    """Alternating weekly highs/lows that are at least `swing` apart (zigzag)."""
    out, mode = [], None
    hi_i, lo_i = wk.index[0], wk.index[0]
    for d, r in wk.iterrows():
        if r['high'] > wk.at[hi_i, 'high']:
            hi_i = d
        if r['low'] < wk.at[lo_i, 'low']:
            lo_i = d
        if mode != 'down' and r['low'] <= wk.at[hi_i, 'high'] * (1 - swing):
            out.append((hi_i, wk.at[hi_i, 'high'], 'high'))
            mode, lo_i = 'down', d
        elif mode != 'up' and r['high'] >= wk.at[lo_i, 'low'] * (1 + swing):
            out.append((lo_i, wk.at[lo_i, 'low'], 'low'))
            mode, hi_i = 'up', d
    return out


def average_cost(units, usd_b):
    """Average cost of the coins the ETFs hold: inflows add coins at that day's price,
    outflows remove coins at the running average cost (so redemptions at low prices
    don't push the average up, as they do in the net-dollar version)."""
    held = cost = 0.0
    out = {}
    for d, q in units.items():
        if q > 0:
            held += q
            cost += usd_b[d] * 1e9
        elif q < 0 and held > 0:
            q_out = min(-q, held)
            cost -= q_out * cost / held
            held -= q_out
        out[d] = cost / held if held > 0 else np.nan
    return pd.Series(out)


def build_dataframe():
    return build_from_flows(fetch_flows(), fetch_price('BTC-USD'), BTC_CFG)


def build_from_flows(flows, px, cfg, usd=None):
    """flows: daily net flows in coin units, columns total / lead / other.
    usd: the same in $B, when the source reports dollars (ETH); otherwise each day's
    coin flow is valued at that day's close (BTC)."""
    if usd is None:
        close = px['close'].reindex(flows.index, method='ffill')
        usd = flows.mul(close, axis=0) / 1e9                     # $B

    # Weekly candles (crypto week ends Sunday)
    launch = flows.index.min()
    wk = px[px.index >= launch].resample('W-SUN').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

    # 13-week rolling flows, on the same weekly grid
    usd_w = usd.resample('W-SUN').sum()
    btc_w = flows.resample('W-SUN').sum()
    roll = usd_w.rolling(ROLL_WEEKS, min_periods=1).sum()     # first 12 weeks: since launch
    roll_btc = btc_w.rolling(ROLL_WEEKS, min_periods=1).sum()

    # Flow breadth: share of the last 20 trading days with net inflows
    breadth = (flows['total'] > 0).astype(float).rolling(BREADTH_DAYS).mean()
    breadth_w = breadth.resample('W-SUN').last()

    # ETF cost basis
    if cfg.get('cb_method', 'net') == 'avg':
        cost_basis = average_cost(flows['total'], usd['total'])
    else:   # cumulative net dollars in ÷ cumulative net coins held
        cost_basis = usd['total'].cumsum() * 1e9 / flows['total'].cumsum()
    cost_basis = cost_basis.where(flows['total'].cumsum() >= cfg['cb_min'])
    cost_basis[cost_basis.notna().cummax() == False] = np.nan   # start once, don't flicker
    cost_basis_w = cost_basis.resample('W-SUN').last()

    return dict(px=px, wk=wk, roll=roll, roll_btc=roll_btc, cost_basis=cost_basis, cost_basis_w=cost_basis_w,
                breadth=breadth, breadth_w=breadth_w, flows=flows, usd=usd)


# ════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════
def flow_shade(v, br, full_in=SHADE_FULL_IN, full_out=SHADE_FULL_OUT):
    """Colour for one week. Hue from the 13-week net flow ($B): green for inflows, red
    for outflows. Every week gets a base tint so regimes read as blocks; depth on top of
    that blends how big the flow is with how broad it is (flow breadth agreeing with the
    direction: share of days with inflows for green, with outflows for red)."""
    if pd.isna(v):
        return None
    inflow = v > 0
    size = min(abs(v) / (full_in if inflow else full_out), 1.0)
    if pd.isna(br):
        agree = 0.5
    else:
        agree = br if inflow else 1 - br
    agree = min(max((agree - 0.3) / 0.5, 0.0), 1.0)    # 30% → none, 80%+ → full
    depth = (0.6 * size + 0.4 * agree) ** 0.8
    return hex_to_rgba(GREEN if inflow else RED_DEEP,
                       round(SHADE_BASE_ALPHA + SHADE_MAX_ALPHA * depth, 3))


def plot_btc_etf_flows(d):
    return plot_etf_flows(d, BTC_CFG)


def plot_etf_flows(d, cfg):
    A, U = cfg['asset'], cfg['unit']
    wk = d['wk']
    roll, roll_btc = d['roll'].dropna(), d['roll_btc'].dropna()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04)

    # ── Background (price panel only): flow regime + graded depth ────────────
    br_sm = d['breadth'].resample('W-SUN').mean().rolling(SHADE_SMOOTH_WK, min_periods=1).mean()
    for end, v in d['roll']['total'].items():
        c = flow_shade(v, br_sm.get(end, np.nan), cfg['full_in'], cfg['full_out'])
        if c:
            fig.add_vrect(x0=end - pd.Timedelta(days=7), x1=end, fillcolor=c,
                          line_width=0, layer='below', row=1, col=1)

    # ── Row 1: weekly candles ──────────────────────────────────────────────────
    chg = wk['close'].pct_change() * 100
    hover_txt = [
        f'Week to {dt:%b} {dt.day}, {dt.year}<br>O ${o:,.0f}<br>H ${h:,.0f}<br>L ${l:,.0f}<br>C ${c:,.0f}'
        + (f'<br>Week {"(" + format(abs(p), ".1f") + "%)" if p < 0 else "+" + format(p, ".1f") + "%"}'
           if pd.notna(p) else '')
        for dt, o, h, l, c, p in zip(wk.index, wk['open'], wk['high'], wk['low'], wk['close'], chg)
    ]
    fig.add_trace(go.Candlestick(
        x=wk.index, open=wk['open'], high=wk['high'], low=wk['low'], close=wk['close'],
        increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED_DEEP, width=1), fillcolor=RED_DEEP),
        whiskerwidth=0, name=f'{A} Price (weekly)', text=hover_txt, hoverinfo='text'),
        row=1, col=1)

    # ETF holders' average cost basis
    cb = d['cost_basis_w'].dropna()
    cb_hov = [f'{dt:%b} {dt.day}, {dt.year}' for dt in cb.index]
    fig.add_trace(go.Scatter(
        x=cb.index, y=cb, name='ETF Cost Basis', line=dict(color=GOLD, width=2),
        customdata=np.c_[cb_hov],
        hovertemplate='%{customdata[0]} · ETF cost basis $%{y:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_annotation(x=cb.index[-1], y=np.log10(cb.iloc[-1]), xref='x', yref='y',
                       text=(f'<b>${cb.iloc[-1]/1e3:,.1f}k</b>' if cb.iloc[-1] >= 1e4
                             else f'<b>${cb.iloc[-1]:,.0f}</b>'), showarrow=False, xanchor='left',
                       xshift=6, font=dict(color=GOLD, size=12))

    # Swing high / low labels, like the reference chart
    for dt, v, kind in pivots(wk, cfg.get('swing', PIVOT_SWING)):
        if dt == wk.index[0]:
            continue
        fig.add_annotation(x=dt, y=np.log10(v), xref='x', yref='y',
                           text=price_label(v),
                           showarrow=False, yshift=12 if kind == 'high' else -12,
                           font=dict(color=TEXT_2, size=12))

    # ── Row 2: 13-week rolling net flows ──────────────────────────────────────
    hov = [(f'13 wks to ' if i >= ROLL_WEEKS - 1 else 'Since launch to ') + f'{dt:%b} {dt.day}, {dt.year}'
           for i, dt in enumerate(roll.index)]
    for col, name, color in (('lead', f'{cfg["lead"]} 13-wk flow', IBIT_COLOR),
                             ('other', 'All other ETFs 13-wk flow', OTHER_COLOR)):
        fig.add_trace(go.Bar(
            x=roll.index, y=roll[col], name=name, marker=dict(color=color, opacity=0.85),
            customdata=np.c_[hov, roll_btc[col] / 1e3],
            hovertemplate='%{customdata[0]} · ' + name.split(' 13')[0]
                          + f' $%{{y:,.1f}}B (%{{customdata[1]:+,.1f}}k {U})<extra></extra>'),
            row=2, col=1)
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll['total'], name='Net ETF Flow (13-wk)',
        line=dict(color=NET_COLOR, width=1.6), customdata=np.c_[hov, roll_btc['total'] / 1e3],
        hovertemplate=f'%{{customdata[0]}} · Net $%{{y:,.1f}}B (%{{customdata[1]:+,.1f}}k {U})<extra></extra>'),
        row=2, col=1)
    fig.add_hline(y=0, line=dict(color=TEXT_3, width=1, dash='dot'), row=2, col=1)

    # Legend-only keys for the background shading
    for name, c in (('13-wk net inflows', GREEN),
                    ('13-wk net outflows', RED_DEEP)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=name,
                                 marker=dict(symbol='square', size=12, color=hex_to_rgba(c, 0.45))),
                      row=1, col=1)

    apply_theme(fig)
    fig.update_layout(barmode='relative', bargap=0.08, xaxis_rangeslider_visible=False)
    ticks = cfg['ticks']
    fig.update_yaxes(type='log', title_text=f'{A} (log)', tickvals=ticks,
                     ticktext=[price_label(t) for t in ticks], row=1, col=1)
    fig.update_yaxes(title_text='13-wk net flow ($B)', tickprefix='$', ticksuffix='B',
                     tickformat=',.0f', row=2, col=1)
    fig.update_xaxes(range=[wk.index[0] - pd.Timedelta(days=7),
                            wk.index[-1] + pd.Timedelta(days=35)])

    # Reading for the header
    last_px = d['px']['close'].iloc[-1]
    r, rb = roll.iloc[-1], roll_btc.iloc[-1]
    br = d['breadth'].dropna().iloc[-1]
    n_in = int(round(br * BREADTH_DAYS))
    state = ('Investors adding' if r['total'] > 0 and br >= 0.5 else
             'Investors leaving' if r['total'] < 0 and br < 0.5 else 'Mixed')
    cb_now = d['cost_basis'].dropna().iloc[-1]
    gap = (last_px / cb_now - 1) * 100
    gap_txt = f'price {abs(gap):.0f}% {"below" if gap < 0 else "above"}'
    subtitle = (f'{A} ${last_px:,.0f} · ETF cost basis ${cb_now:,.0f} ({gap_txt}) · 13-wk net ETF flow {usd_b(r["total"])} '
                f'= {units_k(rb["total"], U)} · Flow breadth {br:.0%} '
                f'({n_in}/{BREADTH_DAYS} days inflows) · {state} · '
                f'Flows to {date_long(d["flows"].index[-1])}')
    return fig, subtitle


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building BTC ETF Flows ===')
    data = build_dataframe()
    fig, subtitle = plot_btc_etf_flows(data)
    write_chart(fig, OUTPUT_PATH,
                title='BTC Spot ETF Flows',
                subtitle=subtitle,
                note='Bars: 13-week rolling net creations − redemptions in US spot bitcoin ETFs (since-launch total until Apr 2024), '
                     'each day\'s BTC flow valued at that day\'s price. '
                     'Shading: green while 13-week net flow is positive, red when negative; '
                     'deeper when the flow is both large and broad (most days in the same direction). '
                     'Flow breadth: share of the past 20 trading days with net inflows. '
                     'Gold line: ETF cost basis, cumulative net dollars in ÷ cumulative net BTC held (flows since launch).',
                source='bitbo.io (BitcoinTreasuries) · Yahoo Finance',
                default='All', min_plot_height=520)
    print(f'Saved: {OUTPUT_PATH}')
