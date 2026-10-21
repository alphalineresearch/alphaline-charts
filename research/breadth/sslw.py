"""
Short-term strong, long-term weak (SSLW): ≥X% of the top 100 above their 30-week average
while ≤Y% are above their 100-week average.

1. threshold grid      — does the result depend on the exact 60 / 25 cut-offs?
2. episodes            — merged runs (gaps ≤ 2 weeks), with forward returns and drawdowns
3. recency             — forward returns from weeks 0–4, 4–13, 13–26 weeks after an episode ended
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth as B

pd.set_option('display.width', 220)
b = pd.read_csv(os.path.join(B.HERE, 'cache', 'breadth.csv'), index_col=0, parse_dates=True)
b = b.loc['2016-01-01':]
btc = b['btc']
fwd = {h: B.forward(btc, h) for h in (13, 26, 52)}
base = {h: np.expm1(fwd[h].dropna().median()) for h in fwd}


def episodes(mask, merge_gap=2):
    """Runs of True weeks, joining runs separated by ≤ merge_gap False weeks."""
    on = mask.fillna(False).values
    idx = mask.index
    eps, start, last = [], None, None
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            elif i - last > merge_gap + 1:
                eps.append((start, last)); start = i
            last = i
    if start is not None:
        eps.append((start, last))
    return [(idx[s], idx[e]) for s, e in eps]


def max_dd_after(d0, weeks=52):
    p = btc.loc[d0:].iloc[:weeks + 1]
    return float((p / p.cummax() - 1).min()), float(p.min() / p.iloc[0] - 1)


def signal(st, lt, long_col='ma100'):
    return (b['ma30'] >= st) & (b[long_col] <= lt)


if __name__ == '__main__':
    # 1. Threshold grid: episodes and their 52w forward from start
    print('=== GRID: episodes (merged) | BTC 52w from episode start: median, # up | 26w median ===')
    rows = []
    for long_col in ('ma100', 'ma200'):
        for st in (0.5, 0.55, 0.6, 0.65, 0.7):
            for lt in (0.15, 0.2, 0.25, 0.3, 0.35):
                eps = episodes(signal(st, lt, long_col))
                s52 = [fwd[52].get(a) for a, _ in eps]
                s52 = [x for x in s52 if pd.notna(x)]
                s26 = [x for x in (fwd[26].get(a) for a, _ in eps) if pd.notna(x)]
                rows.append(dict(long=long_col, st=st, lt=lt, n=len(eps), n52=len(s52),
                                 up52=sum(x > 0 for x in s52),
                                 med52=np.expm1(np.median(s52)) if s52 else np.nan,
                                 med26=np.expm1(np.median(s26)) if s26 else np.nan,
                                 weeks_on=int(signal(st, lt, long_col).sum())))
    g = pd.DataFrame(rows)
    g['hit'] = g.up52.astype(str) + '/' + g.n52.astype(str)
    for long_col in ('ma100', 'ma200'):
        sub = g[g.long == long_col]
        print(f'\n{long_col}: cell = hits/episodes-with-52w-data, median 52w')
        print(sub.pivot(index='st', columns='lt', values='hit').to_string())
        print(sub.pivot(index='st', columns='lt', values='med52').map(lambda v: f'{v:+.0%}' if pd.notna(v) else '–').to_string())
    print(f'\nbaseline BTC median: 26w {base[26]:+.0%}, 52w {base[52]:+.0%}')

    # 2. Episodes at the chosen thresholds
    for st, lt in [(0.6, 0.25), (0.55, 0.3)]:
        eps = episodes(signal(st, lt))
        print(f'\n=== EPISODES 30w ≥ {st:.0%}, 100w ≤ {lt:.0%} ===')
        for a, z in eps:
            dd, low = max_dd_after(a)
            f = {h: fwd[h].get(a) for h in fwd}
            fe = {h: fwd[h].get(z) for h in fwd}
            print(f'  {a.date()} → {z.date()} ({(z - a).days // 7 + 1:2d}w) | from start: 13w {np.expm1(f[13]):+.0%} 26w {np.expm1(f[26]):+.0%} 52w {np.expm1(f[52]):+.0%}'
                  f' | max drawdown in next 52w {dd:+.0%}, lowest vs entry {low:+.0%}'
                  f' | from END: 26w {np.expm1(fe[26]):+.0%} 52w {np.expm1(fe[52]):+.0%}')

    # 3. Recency: every week, by weeks since the last episode ended
    st, lt = 0.6, 0.25
    eps = episodes(signal(st, lt))
    state = pd.Series('off', index=b.index)
    for a, z in eps:
        state.loc[a:z] = 'on'
    ends = [z for _, z in eps]
    since = pd.Series(np.nan, index=b.index)
    for d in b.index:
        prior = [z for z in ends if z < d]
        if prior and state[d] != 'on':
            since[d] = (d - prior[-1]).days // 7
    bucket = pd.Series('off (>26w or never)', index=b.index)
    bucket[state == 'on'] = 'on'
    bucket[(since >= 1) & (since <= 4)] = 'fresh (ended ≤4w)'
    bucket[(since > 4) & (since <= 13)] = 'recent (4–13w)'
    bucket[(since > 13) & (since <= 26)] = 'fading (13–26w)'
    print('\n=== RECENCY buckets (all weeks) → BTC forward median / % up ===')
    out = []
    for k in ['on', 'fresh (ended ≤4w)', 'recent (4–13w)', 'fading (13–26w)', 'off (>26w or never)']:
        w = bucket[bucket == k].index
        r = {'bucket': k, 'weeks': len(w)}
        for h in (13, 26, 52):
            f = fwd[h].reindex(w).dropna()
            r[f'{h}w med'] = f'{np.expm1(f.median()):+.0%}' if len(f) else '–'
            r[f'{h}w up'] = f'{(f > 0).mean():.0%}' if len(f) else '–'
        out.append(r)
    print(pd.DataFrame(out).set_index('bucket').to_string())
    print('\ncurrent:', bucket.iloc[-1], '| weeks since end:', since.iloc[-1], '| last episode:', eps[-1][0].date(), '→', eps[-1][1].date())
    bucket.to_csv(os.path.join(B.HERE, 'cache', 'sslw_bucket.csv'))


def page_data():
    """Episodes, recency buckets and the threshold grid for the research page."""
    eps = episodes(signal(0.6, 0.25))
    ep_rows = []
    for a, z in eps:
        dd, low = max_dd_after(a)
        ep_rows.append(dict(start=a.strftime('%Y-%m-%d'), end=z.strftime('%Y-%m-%d'),
                            weeks=(z - a).days // 7 + 1,
                            **{f'btc_{h}': (None if pd.isna(fwd[h].get(a)) else float(np.expm1(fwd[h].get(a)))) for h in (13, 26, 52)},
                            max_dd=dd if a < btc.index[-30] else None, low_vs_entry=low if a < btc.index[-30] else None))
    bucket = pd.read_csv(os.path.join(B.HERE, 'cache', 'sslw_bucket.csv'), index_col=0, parse_dates=True).iloc[:, 0]
    rec = []
    for k in ['on', 'fresh (ended ≤4w)', 'recent (4–13w)', 'fading (13–26w)', 'off (>26w or never)']:
        w = bucket[bucket == k].index
        r = {'bucket': k, 'weeks': int(len(w))}
        for h in (13, 26, 52):
            f = fwd[h].reindex(w).dropna()
            r[f'med_{h}'] = float(np.expm1(f.median())) if len(f) else None
            r[f'up_{h}'] = float((f > 0).mean()) if len(f) else None
        rec.append(r)
    grid = []
    for st in (0.5, 0.55, 0.6, 0.65, 0.7):
        for lt in (0.15, 0.2, 0.25, 0.3, 0.35):
            e = episodes(signal(st, lt))
            s52 = [x for x in (fwd[52].get(a) for a, _ in e) if pd.notna(x)]
            grid.append(dict(st=st, lt=lt, n=len(s52), up=int(sum(x > 0 for x in s52)),
                             med=float(np.expm1(np.median(s52))) if s52 else None))
    return dict(episodes=ep_rows, recency=rec, grid=grid)


if __name__ == '__main__' and '--page' in sys.argv:
    import json
    json.dump(page_data(), open(os.path.join(B.HERE, 'cache', 'sslw_page.json'), 'w'))
    print('wrote sslw_page.json')
