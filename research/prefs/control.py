"""Do preferred-stock trend breaks say more about BTC than BTC's own trend break?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import events as E

px = E.px
b = px['BTC-USD']


def cross_down(s, n=50):
    ma = s.rolling(n).mean()
    x = (s < ma) & (s.shift() >= ma.shift())
    return list(x[x].index)


def summ(name, dates, h=60):
    v = [x for x in (E.fwd(b, d, h) for d in dates) if pd.notna(x)]
    w = [E.worst_ahead(b, d) for d in dates]
    print(f'{name:40} n={len(v):2}  BTC {h}d median {np.median(v):+.0%}  lower in {sum(x < 0 for x in v)}/{len(v)}'
          f'   median worst-60d {np.median(w):+.0%}')


if __name__ == '__main__':
    summ('ALL days (baseline)', list(px.index[:-60]))
    summ('BTC closes below its 50d avg', cross_down(b))
    for t in ('MSTR', 'STRK', 'STRD', 'STRF'):
        summ(f'{t} closes below its 50d avg', cross_down(px[t]))
    ma = b.rolling(50).mean()
    early = [x for x in cross_down(px['STRD']) if b[x] > ma[x]]
    summ('STRD break while BTC still above 50d', early)
    print('   ', [x.date().isoformat() for x in early])
    bb = cross_down(b)
    for t in ('STRK', 'STRD', 'MSTR'):
        lags = []
        for x in cross_down(px[t]):
            near = [(y - x).days for y in bb if abs((y - x).days) <= 30]
            if near:
                lags.append(min(near, key=abs))
        print(f'{t} break → nearest BTC break (days, + = BTC later): {lags}')
    for t in ('BTC-USD', 'STRD', 'STRK', 'MSTR'):
        s = px[t]
        print(f'now {t}: {s.iloc[-1] / s.rolling(50).mean().iloc[-1] - 1:+.1%} vs 50d avg')
