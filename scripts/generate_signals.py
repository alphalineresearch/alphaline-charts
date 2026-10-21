"""
generate_signals.py
Alphaline Research — homepage Signal Monitor feed

Reads the chart HTML files the other scripts just wrote to docs/, pulls the
latest readings out of them, adds treasury-company mNAV, and writes
docs/signals.json. The homepage (index.html) fetches that file on load; if it
is missing or broken, the page falls back to the numbers baked into it.

Run AFTER the chart scripts in the daily workflow:
    python scripts/generate_signals.py

Weekly upkeep: update TREASURIES below from each company's Monday release.
"""

import base64
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DOCS = 'docs'
OUTPUT_PATH = os.path.join(DOCS, 'signals.json')

# ════════════════════════════════════════════
# BRAND / STATUS COLOURS (match the site)
# ════════════════════════════════════════════
GREEN   = '#2ABF7A'
RED     = '#E86A6A'
NEUTRAL = '#8FA2B2'
# BTC Opportunity Signal recency buckets — same colours as the risk-score chart
RECENCY = [(30, '#2ABF7A'), (90, '#D4A843'), (180, '#8A6B25'), (365, '#D64444'), (None, '#7A8F9F')]
SIGNAL_THRESHOLD = 40

# ════════════════════════════════════════════
# TREASURY COMPANIES — update weekly
#   mNAV (net) = market cap ÷ (crypto × price + cash − debt − preferred)
#   cash = cash + marketable securities + other investments
# ════════════════════════════════════════════
TREASURIES = [
    # ticker, asset, coins held, shares (basic, all classes), cash, debt, preferred, as-of / source
    dict(t='MSTR', asset='BTC', coins=848_517,   shares=422_353_257, cash=5_280_000_000,
         debt=6_714_000_000, pref=13_556_668_500, asof='2026-10-19 · strategy.com'),
    dict(t='ASST', asset='BTC', coins=36_818,    shares=124_060_000, cash=366_850_000,
         debt=0, pref=1_403_017_600, asof='2026-10-16 · 8-K (SATS $700M settles 10-20: add to pref/cash)'),
    dict(t='BMNR', asset='ETH', coins=6_041_481, shares=603_230_000, cash=852_000_000,
         debt=0, pref=0, asof='2026-10-18 · PR (cash/securities $582M + Beast $180M + ORBS $74M + 215 BTC)'),
    dict(t='SBET', asset='ETH', coins=893_956,   shares=217_508_000, cash=50_000_000,
         debt=0, pref=0, asof='2026-10-19 · sharplink.com dashboard'),
]
PRICE_TICKERS = {'BTC': 'BTC-USD', 'ETH': 'ETH-USD'}


# ════════════════════════════════════════════
# PLOTLY HTML READER
# ════════════════════════════════════════════
_DTYPES = {'f8': '<f8', 'f4': '<f4', 'i1': 'i1', 'i2': '<i2', 'i4': '<i4', 'i8': '<i8',
           'u1': 'u1', 'u2': '<u2', 'u4': '<u4', 'u8': '<u8'}

def _decode(v):
    """Plotly 6+ writes numeric arrays as {'dtype': 'f8', 'bdata': base64}."""
    if isinstance(v, dict) and 'bdata' in v:
        arr = np.frombuffer(base64.b64decode(v['bdata']), dtype=_DTYPES[v['dtype']])
        if 'shape' in v:
            shape = [int(s) for s in str(v['shape']).split(',')]
            arr = arr.reshape(shape)
        return arr.astype(float)
    return np.array(v if v is not None else [], dtype=object)


def read_traces(filename):
    """Return {trace name: pd.Series indexed by date} from a Plotly HTML file."""
    path = os.path.join(DOCS, filename)
    with open(path, encoding='utf-8') as fh:
        html = fh.read()
    start = html.index('Plotly.newPlot(')
    data, _ = json.JSONDecoder().raw_decode(html, html.index('[', start))
    out = {}
    for tr in data:
        name = tr.get('name')
        if not name or 'x' not in tr or 'y' not in tr:
            continue
        x = _decode(tr['x'])
        y = _decode(tr['y'])
        if len(x) != len(y) or not len(x):
            continue
        try:
            idx = pd.to_datetime(pd.Series(x).astype(str), errors='coerce')
        except Exception:
            continue
        s = pd.Series(pd.to_numeric(pd.Series(y), errors='coerce').values, index=idx)
        s = s[s.index.notna()].dropna()
        s = s[~s.index.duplicated(keep='last')].sort_index()
        if len(s):
            out[name] = s
    return out


def pick(traces, *names):
    """First trace whose name matches (exact, then prefix) one of `names`."""
    for n in names:
        if n in traces:
            return traces[n]
    for n in names:
        for k, v in traces.items():
            if k.startswith(n):
                return v
    raise KeyError(f'none of {names} in {list(traces)}')


def value_days_ago(s, days):
    """Value on or just before (last date − days)."""
    target = s.index[-1] - pd.Timedelta(days=days)
    prior = s.loc[:target]
    return prior.iloc[-1] if len(prior) else np.nan


# ════════════════════════════════════════════
# FORMATTING
# ════════════════════════════════════════════
def pct(v, d=1):
    """+4.5% / (3.9%) — finance-style negatives."""
    if v is None or not np.isfinite(v):
        return '—'
    return f'({abs(v):.{d}f}%)' if v < 0 else f'+{v:.{d}f}%'

def usd(v):
    return f'${v:,.0f}'

def bn(v):
    return f'${v:,.1f}B'

def date_long(ts):
    return f'{ts:%b} {ts.day}, {ts.year}'


# ════════════════════════════════════════════
# TILES
# ════════════════════════════════════════════
def tile_btc_signal():
    tr = read_traces('btc_risk_score_history.html')
    score = pick(tr, 'Composite Score (14d avg)', 'Composite Score')
    now = float(score.iloc[-1])
    on = now >= SIGNAL_THRESHOLD
    fired = score[score >= SIGNAL_THRESHOLD]
    if on:
        # start of the current run above the threshold
        below = score[score < SIGNAL_THRESHOLD]
        run_start = score.loc[below.index[-1]:].index[1] if len(below) else score.index[0]
        days_since = 0
        sub = f'Score {now:.0f} · fires at {SIGNAL_THRESHOLD}+ · on since {date_long(run_start)}'
    else:
        days_since = (score.index[-1] - fired.index[-1]).days if len(fired) else None
        last = f'last on {date_long(fired.index[-1])}' if len(fired) else 'not triggered yet'
        sub = f'Score {now:.0f} · fires at {SIGNAL_THRESHOLD}+ · {last}'
    bucket = len(RECENCY) - 1
    if days_since is not None:
        for i, (limit, _) in enumerate(RECENCY):
            if limit is None or days_since <= limit:
                bucket = i
                break
    return dict(id='btc-risk', name='BTC Opportunity Signal', val='On' if on else 'Off', unit='',
                state='', color=RECENCY[bucket][1], sub=sub,
                viz=dict(type='recency', bucket=bucket)), score.index[-1]


def tile_btc_hashrate():
    tr = read_traces('btc_price_hashrate.html')
    hr = pick(tr, 'Hash Rate')
    ath = float(pick(tr, 'Hash Rate ATH').iloc[-1]) if 'Hash Rate ATH' in tr else float(hr.max())
    now = float(hr.iloc[-1])
    c7 = (now / value_days_ago(hr, 7) - 1) * 100
    c30 = (now / value_days_ago(hr, 30) - 1) * 100
    if c7 > 0 and c30 > 0:
        state, color = 'Expanding', GREEN
    elif c7 < 0 and c30 < 0:
        state, color = 'Contracting', RED
    else:
        state, color = 'Mixed', NEUTRAL
    below = (1 - now / ath) * 100
    high = 'at its high' if below < 0.5 else f'{below:.0f}% below its high'
    return dict(id='btc-hash', name='BTC Hashrate', val=pct(c7), unit='7d', state=state, color=color,
                sub=f'{now:,.0f} EH/s · {high}'), hr.index[-1]


def tile_btc_cost():
    tr = read_traces('btc_price_vs_production_cost.html')
    price = pick(tr, 'BTC Price')
    cost = pick(tr, 'Production Cost (hist)', 'Production Cost')
    both = pd.concat([price, cost], axis=1, keys=['p', 'c']).dropna()
    p, c = float(both['p'].iloc[-1]), float(both['c'].iloc[-1])
    prem = (p / c - 1) * 100
    above = prem >= 0
    return dict(id='btc-cost', name='BTC vs. Production Cost', val=pct(prem, 0), unit='',
                state='Above cost' if above else 'Below cost', color=NEUTRAL if above else GREEN,
                sub=f'{usd(p)} price vs. {usd(c)} cost to mine',
                viz=dict(type='scale', min=-50, max=150, v=round(prem, 1),
                         zones=[[-50, 0, 'opp']], ticks=[0])), both.index[-1]


def tile_eth_model():
    tr = read_traces('eth_combined_tvl.html')
    price = pick(tr, 'ETH Price')
    model = pick(tr, 'Model (combined TVL)', 'Model')
    both = pd.concat([price, model], axis=1, keys=['p', 'm']).dropna()
    p, m = float(both['p'].iloc[-1]), float(both['m'].iloc[-1])
    gap = (p / m - 1) * 100
    if gap < 0:
        state, color = f'ETH {abs(gap):.0f}% below', GREEN
    else:
        state, color = f'ETH {gap:.0f}% above', RED if gap > 25 else NEUTRAL
    return dict(id='eth-model', name='ETH Model Price', val=usd(m), unit='', state=state, color=color,
                sub=f'ETH trades at {usd(p)}',
                viz=dict(type='scale', min=-50, max=50, v=round(max(-50, min(50, gap)), 1),
                         zones=[[-50, -20, 'opp'], [20, 50, 'str']], ticks=[0])), both.index[-1]


def tile_eth_secured():
    tr = read_traces('eth_stacked_tvl.html')
    parts = [('Stablecoins', pick(tr, 'Stablecoins')),
             ('DeFi', pick(tr, 'DeFi TVL', 'DeFi')),
             ('Tokenized assets', pick(tr, 'RWA TVL', 'RWA'))]
    total = pd.concat([s for _, s in parts], axis=1).dropna().sum(axis=1)   # $B
    t_now, t_7 = float(total.iloc[-1]), value_days_ago(total, 7)
    chg = round((t_now / t_7 - 1) * 100, 1)
    rows = []
    for label, s in parts:
        s = s.loc[:total.index[-1]]
        rows.append([label, bn(float(s.iloc[-1])), round((s.iloc[-1] / value_days_ago(s, 7) - 1) * 100, 1)])
    return dict(id='eth-secured', name='Secured on Ethereum', val=bn(t_now), unit='', chg=chg,
                state='Up this week' if chg >= 0 else 'Down this week', color=GREEN if chg >= 0 else RED,
                sub='', viz=dict(type='rows', rows=rows)), total.index[-1]


def fetch_last_close(ticker):
    import yfinance as yf
    df = yf.download(ticker, period='10d', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return float(df['close'].dropna().iloc[-1])


def tile_treasuries(price_fn=fetch_last_close):
    coin_px = {a: price_fn(t) for a, t in PRICE_TICKERS.items()}
    rows = []
    for c in TREASURIES:
        try:
            mcap = price_fn(c['t']) * c['shares']
            net = c['coins'] * coin_px[c['asset']] + c['cash'] - c['debt'] - c['pref']
            val = f'{mcap / net:.2f}×' if net > 0 else 'n/m'
        except Exception as e:
            print(f'  {c["t"]}: {e}', file=sys.stderr)
            val = '—'
        rows.append([f'{c["t"]} · {c["asset"]}', val, None])
    return dict(id='dat', name='Treasury Companies · mNAV', val='', unit='', state='', color=NEUTRAL,
                sub='Market cap ÷ net assets (crypto + cash − debt − preferred)',
                viz=dict(type='rows', rows=rows)), None


# ════════════════════════════════════════════
# HOMEPAGE MINI-CHARTS ("More charts" cards)
# Small, always-current line charts drawn by the homepage as SVG —
# replaces the static PNG thumbnails. ~3 years, ~150 points per line.
# ════════════════════════════════════════════
GOLD, MIST_L, GREEN_L = '#D4A843', '#B4C2CD', '#2ABF7A'
THUMB_YEARS = 3
THUMB_POINTS = 150


def _thin(s, start):
    s = s.loc[start:].dropna()
    if len(s) > THUMB_POINTS:
        idx = np.linspace(0, len(s) - 1, THUMB_POINTS).round().astype(int)
        s = s.iloc[idx]
    return s


def _pts(s):
    return [float(f'{v:.4g}') for v in s.values]


def _thumb(id_, title, g, series, note, log=False, ymin=None, ymax=None, ref=None):
    end = max(s.index[-1] for _, s, _ in series)
    start = end - pd.DateOffset(years=THUMB_YEARS)
    return dict(id=id_, t=title, g=g, href=f'charts.html#{id_}', log=log, ymin=ymin, ymax=ymax,
                ref=ref or [], note=note, start=start.strftime('%Y-%m'),
                series=[dict(n=n, c=c, pts=_pts(_thin(s, start))) for n, s, c in series])


def thumb_btc_signal():
    sc = pick(read_traces('btc_risk_score_history.html'), 'Composite Score (14d avg)', 'Composite Score')
    v = sc.iloc[-1]
    return _thumb('btc-risk', 'BTC Opportunity Signal', 'btc', [('Score', sc, GOLD)],
                  f'Score {v:.0f} · signal {"on" if v >= SIGNAL_THRESHOLD else "off"} (fires at {SIGNAL_THRESHOLD}+)',
                  ymin=0, ymax=100, ref=[SIGNAL_THRESHOLD])

def thumb_btc_cost():
    tr = read_traces('btc_price_vs_production_cost.html')
    p, c = pick(tr, 'BTC Price'), pick(tr, 'Production Cost (hist)', 'Production Cost')
    return _thumb('btc-cost', 'BTC Price vs. Production Cost', 'btc',
                  [('BTC price', p, GOLD), ('Cost to mine', c, GREEN_L)],
                  f'{usd(p.iloc[-1])} price vs. {usd(c.iloc[-1])} cost', log=True)

def thumb_eth_model():
    tr = read_traces('eth_combined_tvl.html')
    p, m = pick(tr, 'ETH Price'), pick(tr, 'Model (combined TVL)', 'Model')
    return _thumb('eth-combined', 'ETH Model Price', 'eth',
                  [('ETH price', p, GOLD), ('Model price', m, MIST_L)],
                  f'ETH {usd(p.iloc[-1])} · model {usd(m.iloc[-1])}', log=True)

def thumb_eth_secured():
    tr = read_traces('eth_stacked_tvl.html')
    st = pick(tr, 'Stablecoins')
    tot = pd.concat([st, pick(tr, 'DeFi TVL', 'DeFi'), pick(tr, 'RWA TVL', 'RWA')], axis=1).dropna().sum(axis=1)
    return _thumb('eth-stacked-tvl-mcap', 'Value Secured on Ethereum', 'eth',
                  [('Total secured', tot, GOLD), ('Stablecoins', st, MIST_L)],
                  f'{bn(tot.iloc[-1])} total · {bn(st.iloc[-1])} stablecoins', ymin=0)

def thumb_mstr():
    s = pick(read_traces('mstr_pnav_heatmap.html'), 'mNAV (z-score heatmap)', 'mNAV', 'P/NAV')
    return _thumb('mstr-pnav', 'MSTR mNAV', 'btc', [('mNAV', s, GOLD)],
                  f'{s.iloc[-1]:.2f}× net of debt & preferred (14d avg)', ymin=0, ref=[1])

def thumb_btc_rsi():
    s = pick(read_traces('btc_rsi_weekly.html'), 'RSI(14)')
    return _thumb('btc-rsi-weekly', 'BTC Weekly RSI(14)', 'btc', [('RSI', s, GOLD)],
                  f'RSI {s.iloc[-1]:.0f}', ymin=0, ymax=100, ref=[30, 70])

THUMBS = [thumb_btc_signal, thumb_btc_cost, thumb_eth_model, thumb_eth_secured, thumb_mstr, thumb_btc_rsi]


def build_thumbs():
    out = []
    for fn in THUMBS:
        try:
            out.append(fn())
        except Exception as e:
            print(f'  thumb {fn.__name__} skipped: {e}', file=sys.stderr)
    return out


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
TILE_ORDER = [tile_btc_signal, tile_btc_hashrate, tile_btc_cost,
              tile_eth_model, tile_eth_secured, tile_treasuries]


def build(previous=None):
    prev_items = {i['id']: i for i in (previous or {}).get('items', [])}
    items, dates, failed = [], [], []
    for fn in TILE_ORDER:
        try:
            item, as_of = fn()
            items.append(item)
            if as_of is not None:
                dates.append(pd.Timestamp(as_of))
            print(f'  ok   {item["id"]:<12} {item["val"] or item["viz"]["rows"]}')
        except Exception as e:
            failed.append(fn.__name__)
            print(f'  FAIL {fn.__name__}: {e}', file=sys.stderr)
    if failed and prev_items:
        # keep yesterday's tile rather than drop it
        ids = [i['id'] for i in items]
        order = ['btc-risk', 'btc-hash', 'btc-cost', 'eth-model', 'eth-secured', 'dat']
        for pid in order:
            if pid not in ids and pid in prev_items:
                items.insert(min(order.index(pid), len(items)), prev_items[pid])
    # "Updated" = the most recent date across the chart data
    as_of = max(dates).strftime('%Y-%m-%d') if dates else datetime.now(timezone.utc).strftime('%Y-%m-%d')
    thumbs = build_thumbs()
    if len(thumbs) < 4 and previous and previous.get('thumbs'):
        thumbs = previous['thumbs']
    return dict(asOf=as_of, generated=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
                items=items, thumbs=thumbs), failed


if __name__ == '__main__':
    print('=== Building homepage signals.json ===')
    previous = None
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, encoding='utf-8') as fh:
                previous = json.load(fh)
        except Exception:
            previous = None
    payload, failed = build(previous)
    if len(payload['items']) < 3:
        # don't overwrite a good file with a mostly-empty one
        sys.exit(f'Only {len(payload["items"])} tiles built — leaving {OUTPUT_PATH} unchanged.')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f'Saved: {OUTPUT_PATH} (as of {payload["asOf"]}{", failed: " + ", ".join(failed) if failed else ""})')
