"""Numbers and series for the research page → cache/page_data.json"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth as B

b = pd.read_csv(os.path.join(B.HERE, 'cache', 'breadth.csv'), index_col=0, parse_dates=True)
df = b.loc['2016-01-01':].copy()
alts_btc = b['alt_cw_index'] / b['btc']          # holdable cap-weighted top-100 alt basket vs BTC
alts_btc_ew = b['alt_ewp_index'] / b['btc']      # holdable equal-weight basket, rebalanced weekly
typical_btc = b['alt_ew_index'] / b['btc']       # the typical (median-like) alt
T = {'btc': b['btc'], 'eth': b['eth'], 'alts_btc': alts_btc}
F = {(k, h): B.forward(v, h).reindex(df.index) for k, v in T.items() for h in (4, 13, 26, 52)}
ex = lambda v: None if pd.isna(v) else round(float(np.expm1(v)), 4)

out = {}
out['series'] = dict(
    date=[d.strftime('%Y-%m-%d') for d in df.index],
    btc=df['btc'].round(2).tolist(),
    pct_above_ma=(df['pct_above_ma'] * 100).round(1).tolist(),
    net_hi_lo=(df['net_hi_lo'] * 100).round(1).tolist(),
    btc_dom=(df['btc_dom'] * 100).round(1).tolist(),
    alts_btc=(alts_btc.loc[df.index] / alts_btc.loc[df.index].iloc[0]).round(5).tolist(),
    alts_btc_ew=(alts_btc_ew.loc[df.index] / alts_btc_ew.loc[df.index].iloc[0]).round(5).tolist(),
    total3_btc=((b['total3'] / b['btc']).loc[df.index] / (b['total3'] / b['btc']).loc[df.index].iloc[0]).round(5).tolist(),
)
# typical alt vs BTC
for h in (13, 26, 52):
    f = F[('alts_btc', h)].dropna()
    out[f'alts_btc_up_share_{h}'] = round(float((f > 0).mean()), 3)
    out[f'alts_btc_med_{h}'] = ex(f.median())
out['alts_btc_total'] = round(float(alts_btc.loc[df.index].iloc[-1] / alts_btc.loc[df.index].iloc[0] - 1), 4)
out['alts_btc_ew_total'] = round(float(alts_btc_ew.loc[df.index].iloc[-1] / alts_btc_ew.loc[df.index].iloc[0] - 1), 4)
out['typical_btc_total'] = round(float(typical_btc.loc[df.index].iloc[-1] / typical_btc.loc[df.index].iloc[0] - 1), 6)
t3 = b['total3'] / b['btc']
t3x = b['total3'] * (1 - b['stable_share'] * b['total'] / b['total3']) / b['btc']   # ex stablecoins
out['total3x_btc_total'] = round(float(t3x.loc[df.index].iloc[-1] / t3x.loc[df.index].iloc[0] - 1), 4)
out['series']['total3x_btc'] = (t3x.loc[df.index] / t3x.loc[df.index].iloc[0]).round(5).tolist()
out['total3_btc_total'] = round(float(t3.loc[df.index].iloc[-1] / t3.loc[df.index].iloc[0] - 1), 4)

# buckets of % above MA → forward medians, by period
bins, lbl = [0, .2, .4, .6, .8, 1.0001], ['0–20', '20–40', '40–60', '60–80', '80–100']
cat = pd.cut(df['pct_above_ma'], bins, labels=lbl, include_lowest=True, right=False)
bk = {}
for per, sl in [('all', slice(None)), ('2016–20', slice('2016', '2020')), ('2021–26', slice('2021', '2026'))]:
    rows = []
    for k in lbl:
        idx = cat.loc[sl][cat.loc[sl] == k].index
        r = {'bucket': k, 'weeks': int(len(idx))}
        for t in ('btc', 'eth', 'alts_btc'):
            for h in (13, 26):
                f = F[(t, h)].reindex(idx).dropna()
                r[f'{t}_{h}'] = ex(f.median()) if len(f) else None
        rows.append(r)
    base = {}
    for t in ('btc', 'eth', 'alts_btc'):
        for h in (13, 26):
            base[f'{t}_{h}'] = ex(F[(t, h)].loc[sl].dropna().median())
    bk[per] = dict(rows=rows, base=base)
out['buckets'] = bk

# events
def events(mask, cooldown=26):
    res, last = [], None
    for d in df.index[mask.fillna(False)]:
        if last is None or (d - last).days > cooldown * 7:
            res.append(d); last = d
    return res
near_hi = (b['btc'] >= 0.9 * b['btc'].rolling(52).max()).reindex(df.index)
chg8 = df['pct_above_ma'].diff(8)
sigs = {
    'divergence': near_hi & (df['pct_above_ma'] < 0.5) & (chg8 < -0.15),
    'washout': df['pct_above_ma'] <= 0.10,
    'thrust': (df['pct_above_ma'] >= 0.6) & (df['pct_above_ma'].rolling(7).min() <= 0.2),
}
ev = {}
for k, m in sigs.items():
    rows = []
    for d in events(m):
        r = {'date': d.strftime('%Y-%m-%d'), 'breadth': round(float(df.at[d, 'pct_above_ma']) * 100)}
        for t in ('btc', 'eth', 'alts_btc'):
            for h in (4, 13, 26):
                r[f'{t}_{h}'] = ex(F[(t, h)].get(d))
        rows.append(r)
    ev[k] = rows
out['events'] = ev
out['baseline'] = {f'{t}_{h}': ex(F[(t, h)].dropna().median()) for t in ('btc', 'eth', 'alts_btc') for h in (4, 13, 26)}
out['baseline_up'] = {f'{t}_{h}': round(float((F[(t, h)].dropna() > 0).mean()), 3) for t in ('btc', 'eth', 'alts_btc') for h in (4, 13, 26)}

rho = pd.read_csv(os.path.join(B.HERE, 'cache', 'robust_rho.csv'))
out['rho'] = rho.round(3).to_dict(orient='records')

last = df.iloc[-1]
out['latest'] = dict(date=df.index[-1].strftime('%Y-%m-%d'),
                     pct_above_ma=round(float(last.pct_above_ma) * 100),
                     pct_above_ma_4w_ago=round(float(df['pct_above_ma'].iloc[-5]) * 100),
                     pct_above_ma_peak_8w=round(float(df['pct_above_ma'].iloc[-8:].max()) * 100),
                     pct_up_yoy=round(float(last.pct_up_yoy) * 100),
                     net_hi_lo=round(float(last.net_hi_lo) * 100),
                     btc_dom=round(float(last.btc_dom) * 100, 1),
                     btc=round(float(last.btc)),
                     btc_off_high=round(float(last.btc / b['btc'].rolling(52).max().iloc[-1] - 1) * 100),
                     n_coins=int(last.n_ma))
out['meta'] = dict(weeks=len(df), start=df.index[0].strftime('%Y-%m-%d'),
                   coins_total=int(pd.read_csv(B.SNAP, usecols=['id']).id.nunique()))
# ── Multi-window breadth (multi_ma.py) ──
for n in (30, 50, 100, 200):
    v = (b[f'ma{n}'] * 100).where(b[f'n{n}'] >= 30).reindex(df.index).round(1)
    out['series'][f'ma{n}'] = [None if pd.isna(x) else float(x) for x in v]
    out['latest'][f'ma{n}'] = round(float(b[f'ma{n}'].iloc[-1]) * 100)
    out['latest'][f'n{n}'] = int(b[f'n{n}'].iloc[-1])
mr = pd.read_csv(os.path.join(B.HERE, 'cache', 'multi_ma_rho.csv'))
out['multi_rho'] = mr[mr.target.isin(['BTC', 'ETH']) & mr.h.isin([13, 52])].round(2).to_dict(orient='records')
an = (b['ma30'] >= 0.6) & (b['ma100'] <= 0.25)
rows, prev = [], None
for d0 in b.loc['2016':].index[an.loc['2016':].fillna(False)]:
    if prev is None or (d0 - prev).days > 12 * 7:
        r = {'date': d0.strftime('%Y-%m-%d'), 'ma30': round(float(b.at[d0, 'ma30']) * 100), 'ma100': round(float(b.at[d0, 'ma100']) * 100)}
        for h in (13, 26, 52):
            r[f'btc_{h}'] = ex(B.forward(b['btc'], h).get(d0))
        rows.append(r)
    prev = d0
out['analogs'] = rows
out['baseline']['btc_52'] = ex(B.forward(b['btc'], 52).reindex(df.index).dropna().median())

json.dump(out, open(os.path.join(B.HERE, 'cache', 'page_data.json'), 'w'), separators=(',', ':'))
print(json.dumps({k: v for k, v in out.items() if k not in ('series', 'rho', 'buckets', 'events')}, indent=1))
print(json.dumps(out['buckets']['all'], indent=0)[:1500])
print(json.dumps(out['events']['divergence'], indent=0)[:1200])
