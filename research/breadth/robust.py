"""
Robustness checks for the breadth results:
  - significance with overlapping returns handled (block-bootstrap of Spearman rho)
  - detrended / change-based versions of dominance and breadth
  - period split for every measure
  - what breadth says about alts vs BTC (ETH/BTC, equal-weight alts/BTC)
  - event studies against the unconditional baseline over the same period
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth as B

START = '2016-01-01'
rng = np.random.default_rng(7)
pd.set_option('display.width', 220)


def block_boot_rho(x, y, h, n_boot=2000):
    """Spearman rho with a moving-block bootstrap (block = horizon) for the 90% interval
    and the share of resamples with the opposite sign."""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3 * h:
        return np.nan, np.nan, np.nan, np.nan
    r = d.rank().values
    rho = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
    n, blk = len(d), max(h, 4)
    starts_n = n - blk + 1
    k = int(np.ceil(n / blk))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + blk) for s in rng.integers(0, starts_n, k)])[:n]
        rr = r[idx]
        vals[i] = np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
    lo, hi = np.percentile(vals, [5, 95])
    p_flip = (vals < 0).mean() if rho > 0 else (vals > 0).mean()
    return rho, lo, hi, p_flip


if __name__ == '__main__':
    b = pd.read_csv(os.path.join(B.HERE, 'cache', 'breadth.csv'), index_col=0, parse_dates=True)
    df = b.loc[START:].copy()
    df['ethbtc'] = b['eth'] / b['btc']
    df['alts_btc'] = b['alt_ew_index'] / b['btc']
    # Change-based measures (remove slow time trends)
    df['dom_chg26'] = b['btc_dom'].diff(26)
    df['dom_vs_1y'] = b['btc_dom'] - b['btc_dom'].rolling(52).mean()
    df['ma_chg8'] = b['pct_above_ma'].diff(8)
    df['stable_chg13'] = b['stable_share'].diff(13)
    df['alts_vs_btc_13w'] = np.log(df['alts_btc'] / df['alts_btc'].shift(13))
    targets = {
        'BTC': b['btc'], 'ETH': b['eth'], 'ETH/BTC': b['eth'] / b['btc'],
        'Alts(EW)/BTC': b['alt_ew_index'] / b['btc'],
    }
    measures = ['pct_above_ma', 'ma_chg8', 'net_hi_lo', 'pct_up_yoy',
                'btc_dom', 'dom_vs_1y', 'dom_chg26', 'stable_share', 'stable_chg13', 'alts_vs_btc_13w']

    rows = []
    for m in measures:
        for tname, ts in targets.items():
            for h in [13, 26]:
                fwd = B.forward(ts, h).reindex(df.index)
                for per, sl in [('all', slice(None)), ('2016–20', slice('2016', '2020')), ('2021–26', slice('2021', '2026'))]:
                    x, y = df.loc[sl, m], fwd.loc[sl]
                    rho, lo, hi, pf = block_boot_rho(x, y, h, n_boot=600 if per != 'all' else 1500)
                    rows.append(dict(measure=m, target=tname, h=h, period=per, rho=rho, lo90=lo, hi90=hi, p_flip=pf))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(B.HERE, 'cache', 'robust_rho.csv'), index=False)

    piv = res.pivot_table(index=['measure', 'target', 'h'], columns='period', values='rho').round(2)
    ci = res[res.period == 'all'].set_index(['measure', 'target', 'h'])[['lo90', 'hi90', 'p_flip']].round(2)
    out = piv.join(ci)
    # "consistent": same sign in both halves, |rho| ≥ 0.15 in each, and 90% interval excludes 0
    out['consistent'] = ((np.sign(out['2016–20']) == np.sign(out['2021–26']))
                         & (out[['2016–20', '2021–26']].abs().min(axis=1) >= 0.15)
                         & ((out['lo90'] > 0) | (out['hi90'] < 0)))
    print(out.to_string())
    print('\nCONSISTENT RELATIONSHIPS:')
    print(out[out.consistent].to_string())

    # Event studies vs baseline over the same years
    def ev(mask, cooldown=26):
        res, last = [], None
        for d in df.index[mask.fillna(False)]:
            if last is None or (d - last).days > cooldown * 7:
                res.append(d)
                last = d
        return res

    near_hi = b['btc'] >= 0.9 * b['btc'].rolling(52).max()
    sigs = {
        'washout ≤10% above MA': df['pct_above_ma'] <= 0.10,
        'thrust ≤20%→≥60% in 6w': (df['pct_above_ma'] >= 0.6) & (df['pct_above_ma'].rolling(7).min() <= 0.2),
        'divergence: BTC near high, breadth <50% & −15pts/8w': near_hi.reindex(df.index) & (df['pct_above_ma'] < 0.5) & (df['ma_chg8'] < -0.15),
        'BTC near high, breadth ≥70%': near_hi.reindex(df.index) & (df['pct_above_ma'] >= 0.7),
    }
    print('\nEVENTS vs baseline (median forward return; baseline = all weeks 2016+)')
    rows = []
    for name, mask in sigs.items():
        dates = ev(mask)
        for tname in ['BTC', 'ETH', 'Alts(EW)/BTC']:
            for h in [4, 13, 26]:
                f = B.forward(targets[tname], h).reindex(df.index)
                e = f.reindex(dates).dropna()
                rows.append(dict(signal=name, n=len(e), target=tname, h=h,
                                 med=np.expm1(e.median()), up=(e > 0).mean(),
                                 base_med=np.expm1(f.median()), base_up=(f > 0).mean()))
        print(f'  {name}: {[d.date().isoformat() for d in dates]}')
    t = pd.DataFrame(rows)
    for c in ['med', 'up', 'base_med', 'base_up']:
        t[c] = t[c].map(lambda v: f'{v:+.0%}')
    print(t.to_string(index=False))
