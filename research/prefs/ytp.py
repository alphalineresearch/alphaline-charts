"""Yield-to-par check: predicted return if STRC is back at $100 after h months (rate unchanged)
vs. the total return STRC actually delivered."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import events as E

s = E.px['STRC'].dropna()
tr = E.TR['STRC'].dropna()
d = E.dv['STRC'][E.dv['STRC'] > 0]
paid = d.reindex(s.index.union(d.index)).fillna(0).sort_index()
rate = (paid.rolling('91D').sum() * 365 / 91).reindex(s.index)
rate[s.index < d.index[0] + pd.Timedelta(days=91)] = np.nan
rate = rate.fillna(d.iloc[0] * 12)                           # first months: monthly × 12

rows = []
for months, days in ((3, 63), (6, 126)):
    pred = (rate * months / 12 + (100 - s)) / s
    act = tr.shift(-days) / tr - 1
    x = pd.DataFrame({'price': s, 'pred': pred, 'act': act}).dropna()
    x['bucket'] = pd.cut(x.price, [0, 90, 95, 97, 99, 101], labels=['<$90', '$90–95', '$95–97', '$97–99', '$99+'])
    g = x.groupby('bucket', observed=True).agg(days=('act', 'size'), pred=('pred', 'median'), act=('act', 'median'),
                                               act_min=('act', 'min'))
    print(f'\n=== {months}-month horizon: predicted (back at par, rate unchanged) vs actual STRC total return ===')
    print(g.map(lambda v: f'{v:+.1%}' if isinstance(v, float) else v).to_string())
    print(f'   all days: corr(pred, actual) = {x.pred.corr(x.act):.2f};  actual ≥ 80% of predicted on {(x.act >= 0.8 * x.pred).mean():.0%} of days')
last = s.index[-1]
for m in (6, 12):
    print(f'now: {m}-month return if back at $100 = {(rate.iloc[-1] * m / 12 + 100 - s.iloc[-1]) / s.iloc[-1]:+.1%}  (price {s.iloc[-1]:.2f}, rate {rate.iloc[-1]:.2f}%)')
