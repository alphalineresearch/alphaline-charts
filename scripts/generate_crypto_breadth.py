"""
generate_crypto_breadth.py
Alphaline Research — Crypto Breadth: short-term strong, long-term weak

Weekly breadth of the crypto market from CoinMarketCap's historical top-500 snapshots
(Sundays). Each week's universe is that week's top 100 by market cap, so coins that
later died count while they were large. Stablecoins, wrapped / liquid-staked copies
and tokenized stocks or commodities are removed.

Signal: ≥ 60% of the top 100 above their 30-week average (short-term strong) while
≤ 25% are above their 100-week average (long-term weak). Episodes are merged across
gaps of ≤ 2 weeks; after an episode ends the signal stays "fresh" for 4 weeks, then
"recent" to 13 weeks and "fading" to 26 weeks.

Research and tests: research/breadth/ (sslw.py, multi_ma.py).

Data store (committed): data/breadth/snapshots_YYYY.csv[.gz] (date, id, price, mcap)
and data/breadth/coins.csv. Each run fetches only the Sundays not yet stored.

Usage:
    python scripts/generate_crypto_breadth.py
    python scripts/generate_crypto_breadth.py --seed research/breadth/cache/raw   # one-off
"""

import glob
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

from alphaline_theme import apply_theme, write_chart, date_long, GOLD, GREEN, WHITE

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_PATH = os.path.join('docs', 'crypto_breadth.html')
STORE_DIR   = os.path.join('data', 'breadth')
COINS_PATH  = os.path.join(STORE_DIR, 'coins.csv')
CMC_URL     = ('https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical'
               '?date={}&limit=500&start=1&convertId=2781')
FIRST_WEEK  = '2013-04-28'
CHART_START = '2016-01-01'

TOP_N       = 100
SHORT_MA    = 30            # weeks
LONG_MA     = 100           # weeks
MIN_COINS   = 30            # coins with enough history before a reading counts
SHORT_MIN   = 0.60          # ≥ this share above the 30-week average
LONG_MAX    = 0.25          # ≤ this share above the 100-week average
MERGE_GAP   = 2             # weeks off that still count as the same episode
RECENCY = [                 # weeks since the episode ended → state
    (4,  'Fresh (ended ≤ 4 wk)'),
    (13, 'Recent (4–13 wk)'),
    (26, 'Fading (13–26 wk)'),
]
STATE_COLORS = {            # green → teal as the signal ages; breadth lines keep gold / white
    'On':                   GREEN,
    'Fresh (ended ≤ 4 wk)': '#9BE3BF',
    'Recent (4–13 wk)':     '#3FB6C7',
    'Fading (13–26 wk)':    '#2E7A8C',
    'Off':                  '#7A8F9F',
}

RWA_TAGS = {'tokenized-stock', 'tokenized-etfs', 'tokenized-gold', 'tokenized-commodities',
            'tokenized-treasury-bonds'}
NOT_RWA  = {1975, 28299}                 # Chainlink, TokenFi: tagged tokenized-stock, ordinary tokens
PEG_REFS = {1: 'BTC', 1027: 'ETH', 5426: 'SOL', 1839: 'BNB'}
SOURCE   = 'CoinMarketCap weekly historical snapshots'


# ════════════════════════════════════════════
# DATA STORE
# ════════════════════════════════════════════
def _rows_from_snapshot(day, data):
    rows, meta = [], []
    for c in data:
        q = (c.get('quotes') or [{}])[0]
        price, mcap = q.get('price'), q.get('marketCap')
        if not price or not mcap or price <= 0 or mcap <= 0:
            continue
        rows.append((day, c['id'], float(f'{price:.6g}'), int(mcap)))
        tags = set(c.get('tags') or [])
        meta.append((c['id'], c.get('symbol'), c.get('slug'),
                     int(bool(RWA_TAGS & tags) and c['id'] not in NOT_RWA)))
    return rows, meta


def _fetch_snapshot(day):
    for attempt in range(5):
        try:
            r = requests.get(CMC_URL.format(day), timeout=30,
                             headers={'User-Agent': 'Mozilla/5.0 (alphaline-charts)'})
            data = r.json().get('data')
            if r.ok and isinstance(data, list):
                return data
        except Exception as e:
            print(f'  {day}: {e}')
        time.sleep(10 * 2 ** attempt)
    return None


def _year_path(year, gz=False):
    return os.path.join(STORE_DIR, f'snapshots_{year}.csv' + ('.gz' if gz else ''))


def load_store():
    parts = []
    for p in sorted(glob.glob(os.path.join(STORE_DIR, 'snapshots_*.csv*'))):
        parts.append(pd.read_csv(p, parse_dates=['date']))
    if not parts:
        return pd.DataFrame(columns=['date', 'id', 'price', 'mcap'])
    df = pd.concat(parts, ignore_index=True)
    return drop_bad_prints(df.drop_duplicates(['date', 'id'], keep='last'))


def drop_bad_prints(df):
    """CoinMarketCap's snapshots hold a few bad readings (e.g. IOV at a $2T market cap on
    Oct 3, 2021; ION at $71B in Apr 2022). Drop a coin's week when its market cap is over
    20× both neighbouring weeks, when it shows up for one isolated week above $10B, or
    when anything other than BTC is worth more than BTC."""
    s = df.sort_values(['id', 'date'])
    g = s.groupby('id')['mcap']
    prev, nxt = g.shift(1), g.shift(-1)
    near = pd.concat([prev, nxt], axis=1).max(axis=1)
    spike = near.notna() & (s['mcap'] > 20 * near) & (s['mcap'] > 1e9)
    orphan = prev.isna() & nxt.isna() & (s['mcap'] > 1e10)
    btc = s.loc[s['id'] == 1].set_index('date')['mcap']
    above_btc = (s['id'] != 1) & (s['mcap'] > s['date'].map(btc))
    bad = spike | orphan | above_btc
    if bad.any():
        print(f'  dropped {int(bad.sum())} bad snapshot readings')
    return s[~bad].sort_values(['date', 'id']).reset_index(drop=True)


def _append(rows, meta):
    if not rows:
        return
    new = pd.DataFrame(rows, columns=['date', 'id', 'price', 'mcap'])
    for year, g in new.groupby(new['date'].str[:4]):
        path = _year_path(year)
        if os.path.exists(_year_path(year, gz=True)):
            path = _year_path(year, gz=True)
            g = pd.concat([pd.read_csv(path), g]).drop_duplicates(['date', 'id'], keep='last')
            g.sort_values(['date', 'id']).to_csv(path, index=False, compression='gzip')
        else:
            g.sort_values(['date', 'id']).to_csv(path, mode='a', index=False,
                                                 header=not os.path.exists(path))
    m = pd.DataFrame(meta, columns=['id', 'symbol', 'slug', 'rwa'])
    if os.path.exists(COINS_PATH):
        old = pd.read_csv(COINS_PATH)
        m = pd.concat([old, m])
        rwa = m.groupby('id')['rwa'].max()
        m = m.drop_duplicates('id', keep='last').set_index('id')
        m['rwa'] = rwa
        m = m.reset_index()
    else:
        m = m.drop_duplicates('id', keep='last')
    m.sort_values('id').to_csv(COINS_PATH, index=False)


def seed_from_raw(raw_dir):
    """One-off: build the store from research/breadth/cache/raw/*.json.gz"""
    os.makedirs(STORE_DIR, exist_ok=True)
    rows, meta = [], []
    for fn in sorted(os.listdir(raw_dir)):
        with gzip.open(os.path.join(raw_dir, fn), 'rt', encoding='utf-8') as f:
            r, m = _rows_from_snapshot(fn[:10], json.load(f))
        rows += r
        meta += m
    _append(rows, meta)
    this_year = str(pd.Timestamp.utcnow().year)
    for p in glob.glob(os.path.join(STORE_DIR, 'snapshots_*.csv')):
        if not p.endswith(f'_{this_year}.csv'):       # finished years are compressed
            pd.read_csv(p).to_csv(p + '.gz', index=False, compression='gzip')
            os.remove(p)
    print(f'seeded {len(rows):,} rows')


def update_store():
    os.makedirs(STORE_DIR, exist_ok=True)
    have = load_store()
    last = have['date'].max() if len(have) else pd.Timestamp(FIRST_WEEK) - pd.Timedelta(days=7)
    yesterday = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
    todo = pd.date_range(last + pd.Timedelta(days=7), yesterday, freq='W-SUN')
    print(f'Snapshot store: {have["date"].nunique()} weeks to {last.date()}; fetching {len(todo)} new')
    for d in todo:
        day = d.strftime('%Y-%m-%d')
        data = _fetch_snapshot(day)
        if not data:
            print(f'  {day}: not available yet')
            break
        rows, meta = _rows_from_snapshot(day, data)
        _append(rows, meta)
        print(f'  {day}: {len(rows)} coins')
        time.sleep(1.2)


# ════════════════════════════════════════════
# BREADTH
# ════════════════════════════════════════════
def pegged_ids(px, coins):
    """Coins that are really another asset: tokenized stocks / commodities (tag), or prices
    that track $1, BTC, ETH, SOL or BNB week to week."""
    rwa = set(coins.loc[coins['rwa'] == 1, 'id'])
    refs = {'USD': pd.Series(1.0, index=px.index)}
    refs.update({name: px[i] for i, name in PEG_REFS.items() if i in px})
    n_obs = px.notna().sum()
    pegged = set()
    for r in refs.values():
        lr = np.log(px.div(r, axis=0))
        wobble = lr.diff().abs().median()
        level = lr.median().abs()
        pegged |= set(wobble[(wobble < 0.01) & (level < np.log(2)) & (n_obs >= 3)].index)
    return (rwa | pegged) - set(PEG_REFS)


def compute(store, coins):
    px_all = store.pivot_table(index='date', columns='id', values='price')
    drop = pegged_ids(px_all, coins)
    clean = store[~store['id'].isin(drop)].copy()
    clean['rank'] = clean.groupby('date')['mcap'].rank(ascending=False, method='first')
    px = clean.pivot_table(index='date', columns='id', values='price')
    rk = clean.pivot_table(index='date', columns='id', values='rank').reindex(px.index)
    uni = rk <= TOP_N

    out = pd.DataFrame(index=px.index)
    for n, col in ((SHORT_MA, 'short'), (LONG_MA, 'long')):
        ma = px.rolling(n, min_periods=int(n * 0.85)).mean()
        ok = uni & ma.notna()
        cnt = ok.sum(axis=1)
        out[col] = (((px > ma) & ok).sum(axis=1) / cnt.replace(0, np.nan)).where(cnt >= MIN_COINS)
        out[f'n_{col}'] = cnt
    out['btc'] = px[1]
    out['on'] = (out['short'] >= SHORT_MIN) & (out['long'] <= LONG_MAX)
    return out, len(drop)


def episodes(on):
    """(start, end) of runs of signal weeks, merging gaps of ≤ MERGE_GAP weeks."""
    idx, eps, start, last = on.index, [], None, None
    for i, v in enumerate(on.values):
        if not v:
            continue
        if start is None:
            start = i
        elif i - last > MERGE_GAP + 1:
            eps.append((idx[start], idx[last]))
            start = i
        last = i
    if start is not None:
        eps.append((idx[start], idx[last]))
    return eps


def states(index, eps):
    """Signal state for every week: On, then Fresh / Recent / Fading after an episode ends."""
    st = pd.Series('Off', index=index)
    for a, z in eps:
        st.loc[a:z] = 'On'
    ends = pd.Series([z for _, z in eps])
    for d in index:
        if st[d] == 'On':
            continue
        prior = ends[ends < d]
        if len(prior):
            wk = (d - prior.iloc[-1]).days // 7
            for limit, label in RECENCY:
                if wk <= limit:
                    st[d] = label
                    break
    return st


# ════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════
def plot_breadth(df, st, eps):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.05)

    # Signal windows on both panels
    for a, z in eps:
        if z < df.index[0]:
            continue
        for r in (1, 2):
            fig.add_vrect(x0=a - pd.Timedelta(days=3.5), x1=z + pd.Timedelta(days=3.5),
                          fillcolor='rgba(42,191,122,0.14)', line_width=0, row=r, col=1)

    # Panel 1: BTC coloured by signal state (one trace per run of the same state)
    shown = set()
    runs = (st != st.shift()).cumsum()
    for _, seg in df.groupby(runs):
        label = st[seg.index[0]]
        end = df.index.get_loc(seg.index[-1])
        seg = df.iloc[df.index.get_loc(seg.index[0]):min(end + 2, len(df))]   # join to next run
        fig.add_trace(go.Scatter(
            x=seg.index, y=seg['btc'], mode='lines', name=label, legendgroup=label,
            showlegend=label not in shown, line=dict(color=STATE_COLORS[label], width=1.8),
            hovertemplate='%{x|%Y-%m-%d}<br>BTC $%{y:,.0f}  |  ' + label + '<extra></extra>',
        ), row=1, col=1)
        shown.add(label)

    # Panel 2: breadth
    fig.add_trace(go.Scatter(
        x=df.index, y=df['short'] * 100, mode='lines', name='% above 30-week avg',
        line=dict(color=GOLD, width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Above 30-week avg: %{y:.0f}%<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['long'] * 100, mode='lines', name='% above 100-week avg',
        line=dict(color=WHITE, width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Above 100-week avg: %{y:.0f}%<extra></extra>',
    ), row=2, col=1)
    fig.add_hline(y=SHORT_MIN * 100, line_color=GOLD, line_width=0.8, line_dash='dot', row=2, col=1)
    fig.add_hline(y=LONG_MAX * 100, line_color=WHITE, line_width=0.8, line_dash='dot',
                  opacity=0.6, row=2, col=1)

    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='BTC (log)', row=1, col=1)
    # signal levels labelled on the axis in their line's colour (off the data)
    lo, hi = round(LONG_MAX * 100), round(SHORT_MIN * 100)
    fig.update_yaxes(title_text='Share of top 100', range=[0, 100], tickmode='array',
                     tickvals=[0, lo, hi, 100],
                     ticktext=['0%', f'<span style="color:{WHITE}">{lo}%</span>',
                               f'<span style="color:{GOLD}">{hi}%</span>', '100%'], row=2, col=1)
    apply_theme(fig)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    if '--seed' in sys.argv:
        seed_from_raw(sys.argv[sys.argv.index('--seed') + 1])
    else:
        update_store()

    store = load_store()
    coins = pd.read_csv(COINS_PATH)
    full, n_drop = compute(store, coins)
    eps = episodes(full['on'])
    st_full = states(full.index, eps)
    df = full.loc[CHART_START:]
    st = st_full.loc[CHART_START:]
    print(f'Breadth: {len(full)} weeks, {n_drop} pegged coins removed, {len(eps)} signal episodes')
    for a, z in eps:
        print(f'  {a.date()} → {z.date()}')

    last = df.index[-1]
    state = st.iloc[-1]
    if state == 'On':
        a = eps[-1][0]
        status = f'<b>Signal on</b> since {date_long(a)}'
    elif state == 'Off':
        status = '<b>Signal off</b>' + (f' · last ended {date_long(eps[-1][1])}' if eps else '')
    else:
        status = f'<b>Signal {state.split(" (")[0].lower()}</b> · ended {date_long(eps[-1][1])}'
    subtitle = (f'{status} · {df["short"].iloc[-1]:.0%} above 30-week avg · '
                f'{df["long"].iloc[-1]:.0%} above 100-week avg · week of {date_long(last)}')

    fig = plot_breadth(df, st, eps)
    write_chart(fig, OUTPUT_PATH,
                title='Crypto Breadth: Short-Term Strong, Long-Term Weak',
                subtitle=subtitle,
                note=(f'Signal = ≥ {SHORT_MIN:.0%} of the top 100 above their 30-week average (gold) while '
                      f'≤ {LONG_MAX:.0%} are above their 100-week average (white) · ex stablecoins and pegged tokens'),
                source=SOURCE, default='All', end=last, min_plot_height=420)
    print(f'Saved: {OUTPUT_PATH}')
