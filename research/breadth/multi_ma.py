"""% of the week's top-100 above their own N-week average, for several N, and what
each says about forward BTC / ETH / alt-basket-vs-BTC returns."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth as B, robust as R

WINDOWS = [30, 50, 100, 200]
pd.set_option('display.width', 220)


def pct_above(p, n, top_n=B.TOP_N):
    px, rk = p['px'], p['rk']
    uni = rk <= top_n
    need = int(n * 0.85)
    ma = px.rolling(n, min_periods=need).mean()
    ok = uni & ma.notna()
    return ((px > ma) & ok).sum(axis=1) / ok.sum(axis=1).replace(0, np.nan), ok.sum(axis=1)


if __name__ == '__main__':
    p = B.build_panel()
    b = pd.read_csv(os.path.join(B.HERE, 'cache', 'breadth.csv'), index_col=0, parse_dates=True)
    for n in WINDOWS:
        b[f'ma{n}'], b[f'n{n}'] = pct_above(p, n)
    b.to_csv(os.path.join(B.HERE, 'cache', 'breadth.csv'))
    print('eligible coins (median per year):')
    print(b[[f'n{n}' for n in WINDOWS]].loc['2015':].resample('YE').median().astype(int).T.to_string())
    print('\ncorrelation between windows:'); print(b[[f'ma{n}' for n in WINDOWS]].loc['2018':].corr().round(2))

    T = {'BTC': b['btc'], 'ETH': b['eth'], 'Alts/BTC': b['alt_cw_index'] / b['btc']}
    rows = []
    for n in WINDOWS:
        start = b.index[b[f'n{n}'] >= 30][0]            # need 30+ coins with that much history
        df = b.loc[max(start, pd.Timestamp('2016-01-01')):]
        mid = df.index[len(df) // 2]
        for t, ts in T.items():
            for h in [4, 13, 26, 52]:
                f = B.forward(ts, h).reindex(df.index)
                rho, lo, hi, _ = R.block_boot_rho(df[f'ma{n}'], f, h, 800)
                r1 = R.block_boot_rho(df.loc[:mid, f'ma{n}'], f.loc[:mid], h, 300)[0]
                r2 = R.block_boot_rho(df.loc[mid:, f'ma{n}'], f.loc[mid:], h, 300)[0]
                rows.append(dict(ma=n, start=df.index[0].date(), target=t, h=h, rho=rho, lo=lo, hi=hi,
                                 first_half=r1, second_half=r2))
    res = pd.DataFrame(rows)
    res['consistent'] = ((np.sign(res.first_half) == np.sign(res.second_half))
                         & (res[['first_half', 'second_half']].abs().min(axis=1) >= 0.15)
                         & ((res.lo > 0) | (res.hi < 0)))
    res.to_csv(os.path.join(B.HERE, 'cache', 'multi_ma_rho.csv'), index=False)
    print('\n', res.round(2).to_string(index=False))

    # bucket view at long horizons
    for n in WINDOWS:
        start = b.index[b[f'n{n}'] >= 30][0]
        df = b.loc[max(start, pd.Timestamp('2016-01-01')):]
        cat = pd.cut(df[f'ma{n}'], [0, .2, .4, .6, .8, 1.001], right=False,
                     labels=['0–20', '20–40', '40–60', '60–80', '80–100'])
        out = {}
        for t in ['BTC', 'ETH']:
            for h in [13, 26, 52]:
                f = B.forward(T[t], h).reindex(df.index)
                out[f'{t} {h}w'] = f.groupby(cat, observed=True).median().pipe(np.expm1)
        out['weeks'] = cat.value_counts().sort_index()
        tb = pd.DataFrame(out)
        tb.loc['ALL'] = [np.expm1(B.forward(T[c.split()[0]], int(c.split()[1][:-1])).reindex(df.index).median()) if c != 'weeks' else len(df) for c in tb.columns]
        print(f'\n=== % above {n}w average (from {df.index[0].date()}) → median forward return ===')
        print(tb.map(lambda v: f'{v:+.0%}' if isinstance(v, float) and abs(v) < 20 else v).to_string())
    print('\nlatest:', {f'ma{n}': round(b[f"ma{n}"].iloc[-1] * 100) for n in WINDOWS},
          {f'n{n}': int(b[f"n{n}"].iloc[-1]) for n in WINDOWS})
