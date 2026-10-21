"""
Where to park money when the crypto trend rule says "out": cash, QQQ, or QQQ only while
QQQ is itself above its 200-day MA (else cash).

Crypto weight w from the trend rule (signal at the close, applied next day); the rest
(1 − w) goes to the parking asset. 10 bp per unit of turnover on each leg. QQQ is total
return (Yahoo adjusted), flat on weekends.

    python research/trend/rotate.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trend as T


def qqq_daily():
    def get():
        import yfinance as yf
        df = yf.download('QQQ', period='max', interval='1d', progress=False, auto_adjust=True)
        s = df['Close'].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s.rename('qqq').dropna()
    return T._cached('qqq', get)


def run(px, w, park_r, park_w):
    """px: crypto price; w: crypto weight known at close; park_w: weight of the rest in the
    parking asset (1 = all, 0 = cash)."""
    r = px.pct_change()
    w = w.shift(1)
    p = ((1 - w) * park_w.shift(1))
    out = w * r + p * park_r - T.FEE * (w.diff().abs() + p.diff().abs())
    return out.dropna()


def table(name, px, q):
    qd = q.reindex(px.index).ffill()
    q_r = qd.pct_change()
    q_ma = q.rolling(200, min_periods=200).mean()                  # 200 trading days, as for stocks
    q_trend = (q > q_ma).astype(float).where(q_ma.notna()).reindex(px.index).ffill()
    pos = T.rules(px, 365)
    start = max(pos['Blend'].first_valid_index(), q_trend.first_valid_index())
    px, q_r, q_trend, pos = px[start:], q_r[start:], q_trend[start:], pos[start:]

    rows = {'Buy & hold crypto': T.stats(px.pct_change().dropna(), 365),
            'Buy & hold QQQ': T.stats(q_r.dropna(), 365)}
    for rule in ('P>200d', 'TSMOM vote'):
        w = pos[rule]
        one = pd.Series(1.0, index=px.index)
        rows[f'{rule} → cash'] = T.stats(run(px, w, q_r, one * 0), 365)
        rows[f'{rule} → QQQ'] = T.stats(run(px, w, q_r, one), 365)
        rows[f'{rule} → QQQ if QQQ trend up'] = T.stats(run(px, w, q_r, q_trend), 365)
    print(f'\n── {name}: {start:%b %Y} → {px.index[-1]:%b %Y} ' + '─' * 40)
    print(T._fmt(pd.DataFrame(rows).T))

    # what QQQ did while the crypto rule was out
    out = pos['P>200d'].shift(1) == 0
    print(f'  while out of {name} (P>200d): QQQ {(1 + q_r[out]).prod() - 1:+.0%} total over '
          f'{out.mean():.0%} of days; corr(crypto, QQQ) daily {px.pct_change().corr(q_r):.2f}')

    # by period
    rows = {}
    for lo, hi in (('2011', '2014'), ('2015', '2018'), ('2019', '2022'), ('2023', '2026')):
        if px[lo:hi].empty:
            continue
        w = pos['P>200d'][lo:hi]
        one = pd.Series(1.0, index=w.index)
        rows[f'{lo}–{hi}'] = {
            '→ cash': T._sd(T.stats(run(px[lo:hi], w, q_r[lo:hi], one * 0), 365)),
            '→ QQQ': T._sd(T.stats(run(px[lo:hi], w, q_r[lo:hi], one), 365)),
            '→ QQQ if up': T._sd(T.stats(run(px[lo:hi], w, q_r[lo:hi], q_trend[lo:hi]), 365))}
    print(f'\n  {name} P>200d by period (Sharpe / MaxDD)')
    print(pd.DataFrame(rows).to_string())


if __name__ == '__main__':
    q = qqq_daily()
    table('BTC', T.btc_daily(), q)
    table('ETH', T.eth_daily(), q)
