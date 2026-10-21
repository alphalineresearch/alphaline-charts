"""
Is the Mayer Multiple decaying, and does a volatility-adjusted version fix it?

Mayer      = price / 200-day MA
Mayer z    = log(price / 200-day MA) / rolling 4-year std of that same log distance
             (no look-ahead: the std uses only the past)
Pctile 4y  = percentile of today's log distance within the trailing 4 years

Per-cycle peaks and troughs, then forward 12-month returns by zone split early / late,
to see which measure keeps the same meaning over time.

    python research/trend/oscillator.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trend as T

WIN = 4 * 365


def measures(px):
    px = px[~px.index.duplicated(keep='last')].asfreq('D').ffill()
    dist = np.log(px / px.rolling(200, min_periods=200).mean())
    sd = dist.rolling(WIN, min_periods=2 * 365).std()
    return pd.DataFrame({'px': px, 'mayer': np.exp(dist), 'z': dist / sd,
                         'pctile': dist.rolling(WIN, min_periods=2 * 365).rank(pct=True)}).dropna(subset=['mayer'])


def cycles(df, spans):
    rows = {}
    for name, (lo, hi) in spans.items():
        seg = df[lo:hi]
        rows[name] = {'Mayer max': seg.mayer.max(), 'z max': seg.z.max(), 'Mayer min': seg.mayer.min(), 'z min': seg.z.min()}
    return pd.DataFrame(rows).T.round(2)


def zones(df, col, edges, labels, split):
    fwd = df.px.shift(-365) / df.px - 1
    out = {}
    for half, seg in (('early', df[:split]), ('late', df[split:])):
        f = fwd.reindex(seg.index)
        z = pd.cut(seg[col], edges, labels=labels)
        g = pd.DataFrame({'z': z, 'f': f}).dropna().groupby('z', observed=False)['f']
        out[(half, 'median')] = g.median()
        out[(half, '% up')] = g.apply(lambda s: (s > 0).mean())
        out[(half, '% time')] = g.size() / g.size().sum()
    t = pd.DataFrame(out)
    return t.map(lambda v: f'{v:+.0%}' if pd.notna(v) else '')


if __name__ == '__main__':
    btc = measures(T.btc_daily())
    print('BTC cycle extremes')
    print(cycles(btc, {'2011': ('2011', '2011'), '2013': ('2013', '2013'), '2017': ('2017', '2018-01'),
                       '2019': ('2019', '2019'), '2021': ('2021', '2021'), '2024': ('2024', '2025'),
                       'bear 2015': ('2014-06', '2015-12'), 'bear 2018': ('2018', '2019-03'),
                       'bear 2022': ('2022', '2022'), '2026': ('2026', '2026')}).to_string())
    print(f"\nnow: Mayer {btc.mayer.iloc[-1]:.2f} · z {btc.z.iloc[-1]:+.2f} · 4y pctile {btc.pctile.iloc[-1]:.0%}")

    print('\nBTC forward 12m by zone — early (≤2018) vs late (2019→)')
    print('Mayer:')
    print(zones(btc, 'mayer', [0, 0.8, 1.0, 1.5, 2.4, 99], ['<0.8', '0.8–1', '1–1.5', '1.5–2.4', '≥2.4'], '2018-12-31').to_string())
    print('Mayer z:')
    print(zones(btc, 'z', [-99, -1.5, -0.5, 0.5, 1.5, 99], ['≤-1.5', '-1.5–-0.5', '±0.5', '0.5–1.5', '≥1.5'], '2018-12-31').to_string())
    print('4y percentile:')
    print(zones(btc, 'pctile', [0, .1, .3, .7, .9, 1.01], ['≤10%', '10–30%', '30–70%', '70–90%', '≥90%'], '2018-12-31').to_string())

    eth = measures(T.eth_daily())
    print('\nETH cycle extremes (Yahoo history from 2017)')
    print(cycles(eth, {'2021': ('2021', '2021'), '2024': ('2024', '2024'), '2025': ('2025', '2025'),
                       'bear 2022': ('2022', '2022')}).to_string())
    print(f"now: Mayer {eth.mayer.iloc[-1]:.2f} · z {eth.z.iloc[-1]:+.2f}")
