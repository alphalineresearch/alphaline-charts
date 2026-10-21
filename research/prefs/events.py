"""Stress / recovery events in crypto-treasury preferreds and what BTC, ETH, MSTR and the
preferred itself did next. Daily data, Feb 2025 → today: only a handful of events."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 250)
p = pd.read_csv(os.path.join(HERE, 'cache', 'prices.csv'), index_col=0, parse_dates=True)
dv = pd.read_csv(os.path.join(HERE, 'cache', 'dividends.csv'), index_col=0, parse_dates=True).fillna(0)
mkt = p.dropna(subset=['MSTR']).loc['2025-02-01':]          # US trading days
px = mkt[['STRC', 'STRK', 'STRF', 'STRD', 'BMNP', 'SATA', 'MSTR', 'BTC-USD', 'ETH-USD']].ffill(limit=3)


def total_return_index(t):
    """Price + dividends reinvested on the ex-date."""
    s = px[t].dropna()
    d = dv[t].reindex(s.index).fillna(0) if t in dv else 0
    return (s + d).div(s.shift()).fillna(1).cumprod() * s.iloc[0]


TR = {t: total_return_index(t) for t in ['STRC', 'STRK', 'STRF', 'STRD']}
H = [5, 20, 60, 120]


def fwd(series, d0, h):
    s = series.dropna()
    i = s.index.searchsorted(d0)
    return s.iloc[i + h] / s.iloc[i] - 1 if i + h < len(s) else np.nan


def worst_ahead(series, d0, h=60):
    s = series.dropna().loc[d0:].iloc[:h + 1]
    return s.min() / s.iloc[0] - 1


def first_crosses(series, below, rearm, cooldown=40):
    """First close below `below` after having closed at or above `rearm`."""
    s = series.dropna()
    armed, out, last = True, [], None
    for d, v in s.items():
        if v >= rearm:
            armed = True
        if armed and v < below and (last is None or (d - last).days > cooldown):
            out.append(d); last = d; armed = False
    return out


def first_recover(series, above, after_below, cooldown=40):
    s = series.dropna()
    seen_low, out = False, []
    for d, v in s.items():
        if v < after_below:
            seen_low = True
        if seen_low and v >= above:
            out.append(d); seen_low = False
    return out


def table(name, dates):
    rows = []
    for d in dates:
        r = {'date': d.date(), 'STRC': px.at[d, 'STRC'], 'BTC': px.at[d, 'BTC-USD']}
        for h in H:
            r[f'BTC {h}d'] = fwd(px['BTC-USD'], d, h)
        r['BTC worst 60d'] = worst_ahead(px['BTC-USD'], d)
        r['ETH 60d'] = fwd(px['ETH-USD'], d, 60)
        r['MSTR 60d'] = fwd(px['MSTR'], d, 60)
        r['STRC TR 60d'] = fwd(TR['STRC'], d, 60)
        r['STRD TR 60d'] = fwd(TR['STRD'], d, 60)
        rows.append(r)
    t = pd.DataFrame(rows).set_index('date')
    fm = {c: (lambda v: f'{v:+.0%}' if pd.notna(v) else '–') for c in t.columns if c not in ('STRC', 'BTC')}
    fm['STRC'] = lambda v: f'${v:.1f}'
    fm['BTC'] = lambda v: f'${v:,.0f}'
    print(f'\n=== {name} ({len(t)}) ===')
    print(t.to_string(formatters=fm))
    return t


if __name__ == '__main__':
    base = {h: px['BTC-USD'].pct_change(h).shift(-h).median() for h in H}
    print('BTC median forward return, all US trading days since Feb 2025:',
          {f'{h}d': f'{v:+.1%}' for h, v in base.items()})
    table('STRC first close < $97 (re-armed above $99)', first_crosses(px['STRC'], 97, 99))
    table('STRC first close < $92 (re-armed above $97)', first_crosses(px['STRC'], 92, 97))
    table('STRC back to ≥ $98 after trading < $95', first_recover(px['STRC'], 98, 95))
    for t in ('STRK', 'STRD'):
        ma = px[t].rolling(50).mean()
        below = (px[t] < ma) & (px[t].shift() >= ma.shift())
        table(f'{t} closes below its 50-day average', list(below[below].index))
    # Yield context now
    ann = {'STRK': 8, 'STRF': 10, 'STRD': 10, 'STRC': 12}
    last = px.index[-1]
    print('\ncurrent yields:', {t: f'{ann[t] / px.at[last, t]:.2%}' for t in ann}, '| 10y', p['^TNX'].dropna().iloc[-1])
