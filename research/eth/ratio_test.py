"""Does secured value / ETH market cap say anything about forward ETH returns?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'breadth'))
import numpy as np, pandas as pd
import generate_eth_stacked_tvl_mcap as G
from robust import block_boot_rho

pd.set_option('display.width', 220)
here = os.path.dirname(os.path.abspath(__file__))
cache = os.path.join(here, 'ratio_df.csv')
if os.path.exists(cache):
    df = pd.read_csv(cache, index_col=0, parse_dates=True)
else:
    df = G.build_dataframe()
    df.to_csv(cache)
d = df.dropna(subset=['eth_price', 'total_to_mcap'])
d = d[d['total_secured_usd'] > 1e9].copy()
d['r30'] = d['total_to_mcap'].rolling(30, min_periods=7).mean()
print('range:', d.index[0].date(), '→', d.index[-1].date(), 'ETH supply const', G.ETH_SUPPLY)
print(d['r30'].resample('YE').agg(['min', 'median', 'max']).round(2).T.to_string())

px = d['eth_price']
H = (30, 90, 180, 365)
F = {h: np.log(px.shift(-h) / px) for h in H}
x = d.loc['2019-01-01':]
mid = x.index[len(x) // 2]
print(f'\nSpearman rho: r30 vs forward log ETH return (block bootstrap 90%); halves split at {mid.date()}')
for h in H:
    f = F[h].reindex(x.index)
    rho, lo, hi, _ = block_boot_rho(x['r30'], f, h, 800)
    a = block_boot_rho(x.loc[:mid, 'r30'], f.loc[:mid], h, 300)[0]
    b = block_boot_rho(x.loc[mid:, 'r30'], f.loc[mid:], h, 300)[0]
    print(f'  {h:3d}d  rho {rho:+.2f} [{lo:+.2f},{hi:+.2f}]   1st half {a:+.2f}   2nd half {b:+.2f}')

# percentile of the ratio vs its own trailing 3-year history (no look-ahead)
pct = x['r30'].rolling(365 * 3, min_periods=365).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
cat = pd.cut(pct, [0, .2, .4, .6, .8, 1.0001], right=False, labels=['0–20', '20–40', '40–60', '60–80', '80–100'])
tb = {}
for h in (90, 180, 365):
    f = F[h].reindex(x.index)
    tb[f'{h}d median'] = f.groupby(cat, observed=True).median().pipe(np.expm1)
    tb[f'{h}d up'] = (f > 0).groupby(cat, observed=True).mean()
tb['days'] = cat.value_counts().sort_index()
t = pd.DataFrame(tb)
base = {f'{h}d median': np.expm1(F[h].reindex(x.index).dropna().median()) for h in (90, 180, 365)}
print('\nratio percentile vs its trailing 3y → forward ETH:')
print(t.map(lambda v: f'{v:+.0%}' if isinstance(v, float) and abs(v) < 20 else v).to_string())
print('baseline:', {k: f'{v:+.0%}' for k, v in base.items()})
print('now: r30', round(d['r30'].iloc[-1], 2), ' pct vs trailing 3y', round(pct.iloc[-1], 2))

# overlap with the combined-TVL model z-score
import generate_signals as S
S.DOCS = os.path.join(here, '..', '..', 'docs')
tr = S.read_traces('eth_model_zscore.html')
z = next((v for k, v in tr.items() if 'z' in k.lower() and 'ma' not in k.lower()), None)
print('\nz-score traces:', list(tr)[:8])
if z is not None:
    j = pd.concat([x['r30'], z.rename('z')], axis=1).dropna()
    print('corr(ratio, model z) spearman:', round(j.corr(method='spearman').iloc[0, 1], 2))
