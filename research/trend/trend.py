"""
Trend and risk/reward research: moving averages for BTC, ETH and the crypto market.

Part A — systematic trend rules vs buy-and-hold
    rules (standard settings, not tuned): price > 200-day MA, 50 > 200 golden cross,
    time-series momentum vote (share of 1/3/6/12-month returns > 0), and their average.
    Signal at the close, position from the next bar, 10 bp per unit of turnover, cash earns 0.
    Sharpe difference vs buy-and-hold with a moving-block bootstrap; period split; MA-length
    sweep to show whether results sit on a plateau or a spike.

Part B — forward returns by zone
    Mayer Multiple (price / 200-day MA) and price / 200-week MA, fixed zone edges.
    Forward 3/6/12-month return: median, % positive, median worst drawdown inside the window,
    observations and roughly independent episodes (obs / horizon).

Data: BTC daily from blockchain.info (Aug 2010→), ETH daily from Yahoo Finance (Nov 2017→),
weekly total market / TOTAL3 from the CoinMarketCap snapshots in data/breadth (pegged assets,
stablecoins and tokenized RWAs removed, as for the breadth chart).

    python research/trend/trend.py
"""
import os
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(HERE, 'cache')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

FEE = 0.001                 # per unit of turnover
rng = np.random.default_rng(7)
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def _cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + '.csv')
    if os.path.exists(p) and pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(p), unit='s') < pd.Timedelta('20h'):
        return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    s = fn()
    s.to_csv(p)
    return s


def btc_daily():
    def get():
        r = requests.get('https://api.blockchain.info/charts/market-price'
                         '?timespan=all&format=json&sampled=false', timeout=60).json()
        v = pd.DataFrame(r['values'])
        s = pd.Series(v['y'].values, index=pd.to_datetime(v['x'], unit='s').dt.normalize(), name='btc')
        return s[s > 0]
    return _cached('btc', get)


def eth_daily():
    """Same history as the ETH Trend chart: Kraken (Aug 2015) → Coinbase → Yahoo."""
    def get():
        import generate_trend_charts as G
        cwd = os.getcwd()
        os.chdir(ROOT)                     # data/eth_early_daily.csv is repo-relative
        try:
            s = G.fetch_eth()
        finally:
            os.chdir(cwd)
        return s[~s.index.duplicated(keep='last')].rename('eth')
    return _cached('eth_full', get)


def market_weekly():
    """Weekly total crypto market cap and TOTAL3 (ex BTC, ETH), pegs / RWAs removed."""
    import generate_crypto_breadth as G
    cwd = os.getcwd()
    os.chdir(ROOT)                         # the store paths are relative to the repo
    try:
        store = G.load_store()
        coins = pd.read_csv(G.COINS_PATH)
    finally:
        os.chdir(cwd)
    px = store.pivot_table(index='date', columns='id', values='price')
    drop = G.pegged_ids(px, coins)
    clean = store[~store['id'].isin(drop)]
    total = clean.groupby('date')['mcap'].sum()
    total3 = clean[~clean['id'].isin([1, 1027])].groupby('date')['mcap'].sum()
    return pd.DataFrame({'total': total, 'total3': total3}).sort_index()


# ════════════════════════════════════════════
# PART A — TREND RULES
# ════════════════════════════════════════════
def rules(px, per_year):
    """Position in [0, 1] for each rule, known at the close of each bar."""
    # crypto trades every day: a 200-day MA is 200 daily bars; weekly analogues 7 / 30 weeks
    bars = (lambda n: n) if per_year == 365 else {50: 7, 200: 30}.get
    ma = lambda n: px.rolling(bars(n), min_periods=bars(n)).mean()
    ma50, ma200 = ma(50), ma(200)
    out = pd.DataFrame(index=px.index)
    out['P>200d'] = (px > ma200).astype(float).where(ma200.notna())
    out['50>200'] = (ma50 > ma200).astype(float).where(ma200.notna())
    look = [int(round(m * per_year / 12)) for m in (1, 3, 6, 12)]
    votes = pd.concat([(px / px.shift(n) > 1).astype(float).where(px.shift(n).notna()) for n in look], axis=1)
    out['TSMOM vote'] = votes.mean(axis=1).where(votes.notna().all(axis=1))
    out['Blend'] = out[['P>200d', '50>200', 'TSMOM vote']].mean(axis=1).where(out.notna().all(axis=1))
    return out


def backtest(px, pos):
    r = px.pct_change()
    w = pos.shift(1)                       # trade on the next bar
    return (w * r - FEE * w.diff().abs()).dropna()


def stats(r, per_year, pos=None):
    eq = (1 + r).cumprod()
    yrs = len(r) / per_year
    dd = eq / eq.cummax() - 1
    out = dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1,
               Vol=r.std() * np.sqrt(per_year),
               Sharpe=r.mean() / r.std() * np.sqrt(per_year) if r.std() else np.nan,
               MaxDD=dd.min(),
               Calmar=(eq.iloc[-1] ** (1 / yrs) - 1) / -dd.min() if dd.min() < 0 else np.nan)
    if pos is not None:
        p = pos.reindex(r.index)
        out['InMkt'] = p.mean()
        out['Trades/yr'] = (p.diff().abs() > 0).sum() / yrs
    return out


def sharpe_diff_boot(rs, rb, per_year, block, n=2000):
    """Moving-block bootstrap of Sharpe(strategy) − Sharpe(buy-and-hold) on paired returns."""
    a, b = rs.values, rb.reindex(rs.index).values
    m = len(a)
    sh = lambda x: x.mean() / x.std() * np.sqrt(per_year)
    diffs = []
    for _ in range(n):
        starts = rng.integers(0, m - block, size=m // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:m]
        diffs.append(sh(a[idx]) - sh(b[idx]))
    diffs = np.array(diffs)
    return np.percentile(diffs, 5), np.percentile(diffs, 95), (diffs <= 0).mean()


def part_a(name, px, per_year, splits, block):
    pos = rules(px, per_year)
    start = pos['Blend'].first_valid_index()
    px, pos = px[start:], pos[start:]
    bh = px.pct_change().dropna()
    print(f'\n── {name}: {start:%b %Y} → {px.index[-1]:%b %Y} ' + '─' * 40)
    rows = {'Buy & hold': stats(bh, per_year)}
    boots = {}
    for c in pos:
        r = backtest(px, pos[c])
        rows[c] = stats(r, per_year, pos[c])
        boots[c] = sharpe_diff_boot(r, bh, per_year, block)
    t = pd.DataFrame(rows).T
    t['ΔSharpe 90%'] = [''] + [f'{lo:+.2f} … {hi:+.2f}' for lo, hi, _ in boots.values()]
    t['P(Δ≤0)'] = [np.nan] + [p for *_, p in boots.values()]
    print(_fmt(t))

    print(f'\n  {name} by period (Sharpe / MaxDD)')
    rows = {}
    for lo, hi in splits:
        seg = px[lo:hi]
        row = {'Buy & hold': _sd(stats(seg.pct_change().dropna(), per_year))}
        for c in pos:
            row[c] = _sd(stats(backtest(seg, pos[c][lo:hi]), per_year))
        rows[f'{lo[:4]}–{hi[:4]}'] = row
    print(pd.DataFrame(rows).to_string())
    return px, pos


def _sd(s):
    return f"{s['Sharpe']:.2f} / {s['MaxDD']:.0%}"


def ma_sweep(name, px, per_year):
    """P > n-day MA for a range of n: is 200 on a plateau?"""
    rows = {}
    s = px.index[365]                      # common start so every length is judged on the same years
    for n in (50, 100, 150, 200, 250, 300, 365):
        ma = px.rolling(n, min_periods=n).mean()
        pos = (px > ma).astype(float).where(ma.notna())
        rows[f'{n}d'] = stats(backtest(px[s:], pos[s:]), per_year, pos[s:])
    rows['Buy & hold'] = stats(px[s:].pct_change().dropna(), per_year)
    print(f'\n  {name}: price > n-day MA, n swept (same start {s:%b %Y})')
    print(_fmt(pd.DataFrame(rows).T))


def _fmt(t):
    t = t.copy()
    for c in ('CAGR', 'Vol', 'MaxDD', 'InMkt'):
        if c in t:
            t[c] = t[c].map(lambda v: f'{v:.0%}' if pd.notna(v) else '')
    for c in ('Sharpe', 'Calmar', 'Trades/yr', 'P(Δ≤0)'):
        if c in t:
            t[c] = t[c].map(lambda v: f'{v:.2f}' if pd.notna(v) else '')
    return t.to_string()


# ════════════════════════════════════════════
# PART B — FORWARD RETURNS BY ZONE
# ════════════════════════════════════════════
def fwd(px, n):
    """Return over the next n bars and the worst drawdown from entry inside that window."""
    ret = px.shift(-n) / px - 1
    lo = px[::-1].rolling(n, min_periods=n).min()[::-1].shift(-1)
    return ret, (lo / px - 1).clip(upper=0)


def zone_table(name, px, ratio, edges, labels, horizons, per_year):
    print(f'\n── {name} ' + '─' * 50)
    z = pd.cut(ratio, edges, labels=labels, right=False)
    for months in horizons:
        n = int(round(months * per_year / 12))
        ret, dd = fwd(px, n)
        df = pd.DataFrame({'z': z, 'ret': ret, 'dd': dd}).dropna()
        rows = {}
        for lab in labels + ['All']:
            g = df if lab == 'All' else df[df.z == lab]
            if len(g) == 0:
                continue
            rows[lab] = {'Median': g.ret.median(), 'Mean': g.ret.mean(), '% up': (g.ret > 0).mean(),
                         'Worst 10%': g.ret.quantile(0.1), 'Med DD': g.dd.median(),
                         'Obs': len(g), '~Indep': len(g) / n,
                         '% time': len(g) / len(df)}
        t = pd.DataFrame(rows).T
        for c in ('Median', 'Mean', '% up', 'Worst 10%', 'Med DD', '% time'):
            t[c] = t[c].map(lambda v: f'{v:+.0%}' if c in ('Median', 'Mean', 'Worst 10%', 'Med DD') else f'{v:.0%}')
        t['Obs'] = t['Obs'].astype(int)
        t['~Indep'] = t['~Indep'].map(lambda v: f'{v:.1f}')
        print(f'\n  forward {months} months')
        print(t.to_string())


def part_b(btc, eth, mkt):
    for name, px in (('BTC', btc), ('ETH', eth)):
        mayer = px / px.rolling(200, min_periods=200).mean()
        zone_table(f'{name} — Mayer Multiple (price / 200-day MA)', px, mayer,
                   [0, 0.8, 1.0, 1.5, 2.4, np.inf], ['<0.8', '0.8–1.0', '1.0–1.5', '1.5–2.4', '≥2.4'],
                   (3, 6, 12), 365)
    for name, px in (('BTC', btc), ('ETH', eth)):
        dist = np.log(px / px.rolling(200, min_periods=200).mean())
        z = dist / dist.rolling(4 * 365, min_periods=2 * 365).std()
        zone_table(f'{name} — Mayer z (log distance from 200-day MA ÷ its 4-year std)', px, z,
                   [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf], ['≤−1.5', '−1.5…−0.5', '±0.5', '0.5…1.5', '≥1.5'],
                   (3, 6, 12), 365)
    w200 = btc / btc.rolling(1400, min_periods=1400).mean()
    zone_table('BTC — price / 200-week MA', btc, w200,
               [0, 1.0, 1.5, 2.5, 4.0, np.inf], ['<1.0', '1.0–1.5', '1.5–2.5', '2.5–4.0', '≥4.0'],
               (6, 12), 365)
    for col in ('total', 'total3'):
        px = mkt[col]
        r = px / px.rolling(30, min_periods=30).mean()
        zone_table(f'{col.upper()} — price / 30-week MA (weekly)', px, r,
                   [0, 0.8, 1.0, 1.5, 2.4, np.inf], ['<0.8', '0.8–1.0', '1.0–1.5', '1.5–2.4', '≥2.4'],
                   (3, 6, 12), 52)


# ════════════════════════════════════════════
if __name__ == '__main__':
    btc, eth = btc_daily(), eth_daily()
    mkt = market_weekly()
    print(f'BTC {btc.index[0]:%Y-%m-%d}→{btc.index[-1]:%Y-%m-%d} · ETH {eth.index[0]:%Y-%m-%d}→ · '
          f'market weekly {mkt.index[0]:%Y-%m-%d}→{mkt.index[-1]:%Y-%m-%d}')

    print('\n═════ PART A — trend rules vs buy-and-hold (10 bp per trade, next-bar execution) ═════')
    part_a('BTC daily', btc, 365, [('2011', '2014'), ('2015', '2018'), ('2019', '2022'), ('2023', '2026')], 60)
    ma_sweep('BTC', btc, 365)
    part_a('ETH daily', eth, 365, [('2016', '2018'), ('2019', '2022'), ('2023', '2026')], 60)
    part_a('TOTAL weekly', mkt['total'], 52, [('2014', '2018'), ('2019', '2022'), ('2023', '2026')], 9)
    part_a('TOTAL3 weekly', mkt['total3'], 52, [('2014', '2018'), ('2019', '2022'), ('2023', '2026')], 9)

    print('\n═════ PART B — forward returns by zone ═════')
    part_b(btc, eth, mkt)
