"""
generate_rsi_confluence.py
Alphaline Research — RSI Bottom Confluence: past dates where Daily, Weekly, Monthly and
Annual RSI were all in the washout zone together, for BTC, ETH and MSTR.

Match rule (distance to the zone, each leg scaled by how far it naturally moves):
    dist = sqrt( ((D-25)/8)^2 + ((W-34)/6)^2 + ((M-43)/5)^2 + ((A-44.9)/3)^2 )
    (MSTR uses a deeper zone: 23 / 32 / 40 / 43)
    match ≤ 1.0 · near miss ≤ 1.3

Data sources (no API keys required):
  - CoinMetrics community API: BTC (from 2010) and ETH (from 2015) daily prices
  - Yahoo Finance: MSTR daily closes (via yfinance)

Writes three files to docs/:
  docs/btc_rsi_confluence.html
  docs/eth_rsi_confluence.html
  docs/mstr_rsi_confluence.html

Usage:
    python generate_rsi_confluence.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from alphaline_theme import apply_theme, write_chart, date_long, pct

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUT_DIR        = 'docs'
RSI_PERIOD     = 14
RSI_1YR_PERIOD = 52       # Annual RSI = RSI(52) on weekly candles, as in generate_rsi_charts.py
LEGS           = ['Daily', 'Weekly', 'Monthly', 'Annual']
SETUP          = {'Daily': 25, 'Weekly': 34, 'Monthly': 43, 'Annual': 44.9}   # the washout zone
# MSTR swings harder than BTC, so its RSIs pass through BTC's zone early in a decline;
# it gets a deeper zone of its own
SETUP_BY_ASSET = {'MSTR': {'Daily': 23, 'Weekly': 32, 'Monthly': 40, 'Annual': 43}}
SCALE          = {'Daily': 8, 'Weekly': 6, 'Monthly': 5, 'Annual': 3}
MATCH_DIST     = 1.0
NEAR_DIST      = 1.3
EPISODE_GAP    = 30       # days between qualifying days that start a new episode
FWD_DAYS       = [('3M', 91), ('6M', 182), ('1Y', 365)]

ASSETS = [
    # source, id, label, weekly candle, price colour, first date to search (None = all history)
    ('coinmetrics', 'btc',  'BTC',  'W-SUN', '#D4A843', None),
    ('coinmetrics', 'eth',  'ETH',  'W-SUN', '#8FAEE6', None),
    ('yahoo',       'MSTR', 'MSTR', 'W-FRI', '#F8FAFB', '2020-08-11'),   # first BTC purchase announced
]

# ════════════════════════════════════════════
# BRAND COLORS
# ════════════════════════════════════════════
GOLD      = '#D4A843'
TEXT_2    = '#B4C2CD'
STEEL     = '#374D61'
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GREEN_LIT = '#2ABF7A'
WHITE     = '#F8FAFB'

def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'

# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════
def rma(series: pd.Series, period: int):
    """Wilder average gain / loss, same smoothing as compute_rsi in generate_rsi_charts.py."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return gain, loss

def rsi_from(gain, loss):
    return 100 - (100 / (1 + gain / loss))

def rsi_open_candle(close: pd.Series, rule: str, period: int) -> pd.Series:
    """RSI on `rule` candles, read every day with that day's candle still open —
    so a past date is measured the same way as today's unfinished candle."""
    bucket  = close.index.to_period(rule)
    closes  = close.groupby(bucket).last()
    gain, loss = rma(closes, period)
    prev_c  = closes.shift(1).reindex(bucket).to_numpy()
    prev_g  = gain.shift(1).reindex(bucket).to_numpy()
    prev_l  = loss.shift(1).reindex(bucket).to_numpy()
    change  = close.to_numpy() - prev_c
    g = (prev_g * (period - 1) + np.clip(change, 0, None)) / period
    l = (prev_l * (period - 1) + np.clip(-change, 0, None)) / period
    return pd.Series(rsi_from(g, l), index=close.index)

def build_episodes(dates, gap_days=EPISODE_GAP):
    episodes = []
    for dt in dates:
        if episodes and (dt - episodes[-1][-1]).days <= gap_days:
            episodes[-1].append(dt)
        else:
            episodes.append([dt])
    return episodes

# ════════════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════════════
def _get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))

def fetch_coinmetrics(asset: str) -> pd.Series:
    url = ('https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
           f'?assets={asset}&metrics=PriceUSD&frequency=1d&page_size=10000&paging_from=start')
    rows = []
    while url:
        j = _get(url)
        rows += j['data']
        url = j.get('next_page_url')
    s = pd.Series({pd.Timestamp(r['time'][:10]): float(r['PriceUSD']) for r in rows if r.get('PriceUSD')})
    return s.sort_index()

def fetch_yahoo_chart(ticker: str) -> pd.Series:
    """Yahoo chart API directly — fallback when yfinance returns nothing."""
    r = requests.get(f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}'
                     '?period1=0&period2=9999999999&interval=1d&events=split,div',
                     headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    adj = res['indicators']['adjclose'][0]['adjclose']
    idx = pd.to_datetime(res['timestamp'], unit='s').normalize()
    return pd.Series(adj, index=idx, dtype=float).dropna().groupby(level=0).last()

def fetch_yahoo(ticker: str) -> pd.Series:
    df = yf.download(ticker, period='max', interval='1d', auto_adjust=True, progress=False)
    if df.empty:
        return fetch_yahoo_chart(ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df['close'].dropna()

def build_dataframe(close: pd.Series, weekly_rule: str) -> pd.DataFrame:
    gain, loss = rma(close, RSI_PERIOD)
    return pd.DataFrame({
        'close':   close,
        'Daily':   rsi_from(gain, loss),
        'Weekly':  rsi_open_candle(close, weekly_rule, RSI_PERIOD),
        'Monthly': rsi_open_candle(close, 'M', RSI_PERIOD),
        'Annual':  rsi_open_candle(close, weekly_rule, RSI_1YR_PERIOD),
    }).dropna()


def zone_distance(d: pd.DataFrame, setup) -> pd.Series:
    return np.sqrt(sum(((d[c] - setup[c]) / SCALE[c]) ** 2 for c in LEGS))

def find_matches(df: pd.DataFrame, setup, since=None):
    """Episodes whose closest day sits within MATCH_DIST (match) or NEAR_DIST (near miss)
    of the washout zone, with forward returns and the lowest point in the next year.
    The episode running into today is left out of the history and returned as `current`."""
    d    = df if since is None else df.loc[since:]
    dist = zone_distance(d, setup)
    episodes = build_episodes(list(d.index[dist <= NEAR_DIST]))
    current = None
    if episodes and episodes[-1][-1] >= d.index[-1] - pd.Timedelta(days=EPISODE_GAP):
        ep = episodes.pop()
        current = dict(start=ep[0], end=d.index[-1], match=min(dist[dt] for dt in ep) <= MATCH_DIST)

    rows = []
    for ep in episodes:
        best  = min(ep, key=lambda dt: dist[dt])    # the day deepest inside the zone
        t_end = best + pd.Timedelta(days=365)
        row = dict(start=ep[0], end=ep[-1], best=best, close=d.close[best], dist=dist[best],
                   match=dist[best] <= MATCH_DIST,
                   low=(df.close.loc[best:t_end].min() / d.close[best] - 1) * 100,
                   **{c: d.loc[best, c] for c in LEGS})
        for lbl, n in FWD_DAYS:
            t = best + pd.Timedelta(days=n)
            row[lbl] = (df.close.asof(t) / d.close[best] - 1) * 100 if t <= df.index[-1] else np.nan
        rows.append(row)
    return d, dist, pd.DataFrame(rows), current

# ════════════════════════════════════════════
# CHART — PRICE + "ALL FOUR IN ZONE" STRIP, RSI DETAIL BEHIND A TOGGLE
# ════════════════════════════════════════════
# y-axis domains: price (y), strip (y2), RSI panels (y3..y6)
DOMAINS_COLLAPSED = [[0.17, 1.0], [0.0, 0.13]] + [[0.0, 0.001]] * 4
DOMAINS_EXPANDED  = [[0.53, 1.0], [0.42, 0.50],
                     [0.30, 0.39], [0.20, 0.29], [0.10, 0.19], [0.0, 0.09]]

def _status(dist_now):
    if dist_now <= MATCH_DIST:
        return 'match'
    return 'near' if dist_now <= NEAR_DIST else 'none'

def _marker(status, size):
    if status == 'match':
        return dict(color=GREEN_LIT, size=size, line=dict(color=NAVY, width=2))
    if status == 'near':
        return dict(color=NAVY, size=size - 1, line=dict(color=GREEN_LIT, width=2))
    return dict(color=STEEL, size=size, line=dict(color=TEXT_2, width=1.5))

def _hover_text(e, label):
    kind = 'Signal' if e.match else 'Near miss'
    return '<br>'.join([
        f'<b>{kind} · {date_long(e.best)}</b> · distance {e.dist:.2f}',
        f'{label} ${e.close:,.2f}  ·  ' + '  ·  '.join(f'{c} {e[c]:.1f}' for c in LEGS),
        'After: ' + '  ·  '.join(f'{lbl} {pct(e[lbl]) if pd.notna(e[lbl]) else "n/a"}'
                                 for lbl, _ in FWD_DAYS),
        f'Lowest point in the next year: {pct(e.low)}',
    ])

def plot_confluence(d, dist, eps, current, label, price_color, setup):
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.0)
    today, dist_now = d.iloc[-1], dist.iloc[-1]
    status = _status(dist_now)

    # ── Episodes shaded full height: solid = signal, faint = near miss ──
    spans = [(e.start, e.end, e.match) for _, e in eps.iterrows()]
    if current:
        spans.append((current['start'], current['end'], current['match']))
    for start, end, is_match in spans:
        a = 0.16 if is_match else 0.07
        fig.add_shape(type='rect', xref='x', yref='paper', y0=0, y1=1, layer='below',
                      x0=start - pd.Timedelta(days=1), x1=end + pd.Timedelta(days=1),
                      fillcolor=hex_to_rgba(GREEN_LIT, a), line_width=0)

    # ── Row 1: price, past signals, today ──────────────────────────
    fig.add_trace(go.Scatter(
        x=d.index, y=d['close'], mode='lines', name=f'{label} price',
        line=dict(color=price_color, width=1.6),
        hovertemplate='%{x|%b %-d, %Y} · ' + label + ' $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)

    for is_match, name in [(True, f'Signal (distance ≤ {MATCH_DIST:.1f})'),
                           (False, f'Near miss (≤ {NEAR_DIST:.1f})')]:
        sub = eps[eps['match'] == is_match] if not eps.empty else eps
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['best'], y=sub['close'], mode='markers', name=name,
            marker=_marker('match' if is_match else 'near', 10), meta='legend-dot',
            hovertext=[_hover_text(e, label) for _, e in sub.iterrows()],
            hovertemplate='%{hovertext}<extra></extra>'
        ), row=1, col=1)

    now_word = {'match': 'Signal on', 'near': 'Near miss', 'none': 'No signal'}[status]
    fig.add_trace(go.Scatter(
        x=[d.index[-1]], y=[today['close']], mode='markers', name='Today',
        marker=_marker(status, 10), showlegend=False,
        hovertemplate=(f'<b>Today · {now_word}</b> · distance {dist_now:.2f}<br>'
                       + label + ' $%{y:,.2f}  ·  '
                       + '  ·  '.join(f'{c} {today[c]:.1f}' for c in LEGS) + '<extra></extra>')
    ), row=1, col=1)

    # ── Row 2: one lit row per RSI when it sits inside its zone ────
    z_leg   = np.array([((d[c] - setup[c]) / SCALE[c]).abs().to_numpy() for c in LEGS])
    in_zone = (z_leg <= 1).astype(float)
    all_four_no_signal = (in_zone.sum(axis=0) == 4) & (dist > MATCH_DIST).to_numpy()
    in_zone[:, all_four_no_signal] = 0.6            # each in zone, combined distance just outside
    fig.add_trace(go.Heatmap(
        x=d.index, y=LEGS[::-1], z=in_zone[::-1],
        customdata=np.array([d[c].to_numpy() for c in LEGS[::-1]]),
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.6, hex_to_rgba(GREEN_LIT, 0.45)], [1, GREEN_LIT]],
        zmin=0, zmax=1, showscale=False, ygap=2, name='In zone',
        hovertemplate='%{x|%b %-d, %Y} · %{y} RSI %{customdata:.1f}<extra></extra>'
    ), row=2, col=1)

    # ── Rows 3-6: RSI detail with the zone band (hidden until toggled) ──
    rsi_trace_ids = []
    first_rsi_shape = len(fig.layout.shapes)
    for r, c in enumerate(LEGS, start=3):
        fig.add_hrect(y0=setup[c] - SCALE[c], y1=setup[c] + SCALE[c],
                      fillcolor=hex_to_rgba(GOLD, 0.14), line_width=0, row=r, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color=STEEL, line_width=0.8, row=r, col=1)
        rsi_trace_ids.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=d.index, y=d[c], mode='lines', name=f'{c} RSI',
            line=dict(color=TEXT_2, width=1.1), showlegend=False,
            hovertemplate='%{x|%b %-d, %Y} · ' + c + ' RSI %{y:.1f}<extra></extra>'
        ), row=r, col=1)
    rsi_shapes = range(first_rsi_shape, len(fig.layout.shapes))

    apply_theme(fig)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='Price (log)', row=1, col=1)
    fig.update_yaxes(showgrid=False, fixedrange=True, row=2, col=1)
    titles = {'Daily': 'Daily', 'Weekly': 'Weekly', 'Monthly': 'Monthly', 'Annual': 'Annual (52w)'}
    for r, c in enumerate(LEGS, start=3):
        lo, hi = d[c].min(), d[c].max()
        pad = (hi - lo) * 0.06
        fig.update_yaxes(title_text=titles[c], range=[lo - pad, hi + pad], nticks=4, row=r, col=1)
    fig.update_xaxes(rangeslider_visible=False, type='date')   # bottom axis's traces start hidden

    # ── Toggle: collapsed by default ───────────────────────────────
    def state(expanded):
        vis = [True] * len(fig.data)
        for i in rsi_trace_ids:
            vis[i] = expanded
        doms = DOMAINS_EXPANDED if expanded else DOMAINS_COLLAPSED
        lay = {f'yaxis{"" if i == 0 else i + 1}.domain': dom for i, dom in enumerate(doms)}
        lay.update({f'yaxis{i}.visible': expanded for i in range(3, 7)})
        lay.update({f'shapes[{i}].visible': expanded for i in rsi_shapes})
        return [{'visible': vis}, lay]

    collapsed_traces, collapsed_layout = state(False)
    for i, v in enumerate(collapsed_traces['visible']):
        fig.data[i].visible = v
    fig.update_layout(collapsed_layout)
    fig.update_layout(
        xaxis_range=[d.index[0], d.index[-1] + (d.index[-1] - d.index[0]) * 0.03],
        updatemenus=[dict(
            type='buttons', direction='right', x=0.0, y=1.0, xanchor='left', yanchor='top',
            pad=dict(l=6, t=6), showactive=False,
            bgcolor=NAVY_MID, bordercolor=GOLD, borderwidth=1, font=dict(color=TEXT_2, size=12),
            buttons=[dict(label='Show RSI detail', method='update',
                          args=state(True), args2=state(False))],
        )],
    )
    return fig

def subtitle_for(d, dist, eps, label):
    today, dist_now = d.iloc[-1], dist.iloc[-1]
    word = {'match': 'SIGNAL ON', 'near': 'NEAR MISS', 'none': 'NO SIGNAL'}[_status(dist_now)]
    m = eps[eps['match']] if not eps.empty else eps
    sub = (f'<b>{word}</b> · distance {dist_now:.2f}  ·  '
           + '  ·  '.join(f'{c} {today[c]:.1f}' for c in LEGS)
           + f'  ·  {len(m)} past signals')
    yr = m['1Y'].dropna() if not m.empty else pd.Series(dtype=float)
    if len(yr):
        sub += (f'  ·  1Y later: {(yr > 0).sum()} of {len(yr)} higher, '
                f'median {pct(yr.median(), 0)}')
    return sub

# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=== Building RSI Bottom Confluence Charts ===')

    def note_for(setup):
        zone = ' · '.join(f'{c} {setup[c]:g} ±{SCALE[c]}' for c in LEGS)
        return (f'Signal when Daily, Weekly, Monthly and Annual RSI sit in the washout zone together ({zone}), '
                f'combined as one distance: solid green ≤ {MATCH_DIST:.1f}, hollow = near miss ≤ {NEAR_DIST:.1f}. '
                'Strip rows light when each RSI is in its zone. Weekly, monthly and annual read the open candle; '
                'annual = RSI(52) on weekly candles. Hover a dot for what followed.')

    for source, ident, label, weekly_rule, color, since in ASSETS:
        print(f'\nFetching {label} ({source})...')
        close = fetch_coinmetrics(ident) if source == 'coinmetrics' else fetch_yahoo(ident)
        df = build_dataframe(close, weekly_rule)
        setup = SETUP_BY_ASSET.get(label, SETUP)
        d, dist, eps, current = find_matches(df, setup, since)
        print(f'  {label}: today distance {dist.iloc[-1]:.2f} ({_status(dist.iloc[-1])}) | '
              + ', '.join(f'{c} {d[c].iloc[-1]:.1f}' for c in LEGS))
        for _, e in eps.iterrows():
            print(f'    {"SIGNAL" if e.match else "near  "} {e.best.date()} d={e.dist:.2f}  ' + '  '.join(
                f'{lbl} {e[lbl]:+.0f}%' if pd.notna(e[lbl]) else f'{lbl} n/a' for lbl, _ in FWD_DAYS)
                + f'  low {e.low:.0f}%')

        fig = plot_confluence(d, dist, eps, current, label, color, setup)
        title = f'{label} RSI Bottom Confluence' + (' (since BTC treasury, Aug 2020)' if since else '')
        source_txt = 'CoinMetrics' if source == 'coinmetrics' else 'Yahoo Finance'
        path = os.path.join(OUT_DIR, f'{label.lower()}_rsi_confluence.html')
        write_chart(fig, path, title=title, subtitle=subtitle_for(d, dist, eps, label),
                    note=note_for(setup), source=source_txt, default='All', min_plot_height=520)
        print(f'  Saved: {path}')

    print('\nDone.')
